// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/receiver.hpp"

#include "distributed/mesh_device_impl.hpp"
#include <tt-metalium/mesh_device.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>
#include <numeric>
#include <sys/prctl.h>
#include <numa.h>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/io_window_config.hpp>

#include "context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include "llrt/zone_meta.hpp"
#include "impl/streaming_profiler/spsc_packet.h"

namespace tt::tt_metal::streaming_profiler {

namespace {

// Credits go back about once per relay push rather than per poll.
constexpr uint32_t kAckBatchPages = 8 * profiler::kSpscMaxFramePages;
constexpr uint32_t kPageWords = kernel_profiler::SPSC_SPAN_PAGE_WORDS;
constexpr uint32_t kPageBytes = kPageWords * 4;
// Idle probe period: credits are returned as soon as pages are walked, so a sleep only delays the frames that land
// during it, never a credit the device is short of (the FIFO holds over a millisecond of egress).
constexpr uint32_t kProbeSleepCapUs = 100;
constexpr size_t kSyncBatchBytes = size_t{kBatchFrames} * profiler::kSpscMaxFrameWords * sizeof(uint32_t);
// The PCIe tile's SII register block in its own address space, and the timer inside it that counts from reset (tables 4
// and 12 of the Blackhole PCIE_SS spec).
constexpr uint64_t kSiiBase = 0xFFFFFFFFF0000000ull;
constexpr uint32_t kCfrLo = 0xA8, kCfrHi = 0xAC;

constexpr uint64_t join(uint32_t hi, uint32_t lo) { return (uint64_t{hi} << 32) | lo; }

}  // namespace

RootRefclkWindow::RootRefclkWindow(tt::Cluster& cluster, uint32_t chip_id) :
    numa_node_(static_cast<int>(cluster.get_numa_node_for_device(chip_id))) {
    const auto pcie =
        cluster.get_driver()->get_soc_descriptor(chip_id).get_cores(CoreType::PCIE, CoordSystem::TRANSLATED);
    // Reads are uncached under WC, and the fences in tsc_now() order them.
    window_ = cluster.get_driver()->create_io_window(
        chip_id,
        pcie.front(),
        kSiiBase,
        tt::umd::HostIoWindowConfig{.mapping = tt::umd::HostMemoryCaching::WC, .size = kCfrHi + sizeof(uint32_t)});
    // Reading LO latches HI, and this is the only reader, so the pair is consistent.
    const uint32_t lo = window_->read32(kCfrLo);
    const uint32_t hi = window_->read32(kCfrHi);
    bases_ = {.root_refclk = static_cast<int64_t>(join(hi, lo)), .tsc = tsc_now()};
    // The first call sleeps while it measures the rate, so it is made here rather than on the sync thread.
    static_cast<void>(ns_per_tsc_tick());
}

RootRefclkWindow::~RootRefclkWindow() = default;

RefclkBurst RootRefclkWindow::read_burst() {
    // Reads from the other CPU socket take about 90 ns longer in one direction, which would shift a burst's midpoint by
    // tens of ns.
    if (numa_node_ >= 0 && numa_available() != -1) {
        numa_run_on_node(numa_node_);
    }
    // LO wraps every 86 s and bursts can be further apart, so HI is re-read each burst.
    RefclkBurst reads;
    uint32_t hi = 0, lo_last = 0;
    for (uint32_t i = 0; i < kRefclkBurstReads; i++) {
        const int64_t before = tsc_now();
        const uint32_t lo = window_->read32(kCfrLo);
        const int64_t after = tsc_now();
        if (i == 0) {
            hi = window_->read32(kCfrHi);
        } else if (lo < lo_last) {
            hi++;
        }
        lo_last = lo;
        reads[i] = {.mid = std::midpoint(before, after), .rtt = after - before, .refclk = join(hi, lo)};
    }
    return reads;
}

Receiver::Receiver(tt::Cluster& cluster, std::unique_ptr<DevicePrograms> programs) :
    programs_(std::move(programs)),
    root_refclk_(cluster, programs_->capture_context().devices[CaptureContext::kRootDevice].chip_id),
    clock_solver_(programs_->capture_context(), root_refclk_.bases()) {
    for (const CapturedSocket& captured : programs_->sockets()) {
        auto stream = std::make_unique<Stream>();
        stream->captured = captured;
        stream->fifo = captured.socket->host_fifo();
        stream->capacity = stream->fifo.size() / kPageBytes;
        stream->marks = std::vector<std::atomic<uint64_t>>(stream->fifo.size() / kMarkBytes);
        for (auto& mark : stream->marks) {
            mark.store(UINT64_MAX, std::memory_order_relaxed);
        }
        streams_view_.push_back(ReceiverStream{
            .fifo = stream->fifo,
            .walked = &stream->walked_bytes,
            .dev = captured.device_index,
            .marks = stream->marks,
            .sync = captured.sync});
        streams_.push_back(std::move(stream));
    }
}

std::unique_ptr<Receiver> Receiver::create(const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto& mc = MetalContext::instance(mesh_device->impl().get_context_id());
    auto programs = std::make_unique<DevicePrograms>();
    if (!programs->boot(mesh_device)) {
        return nullptr;
    }
    service().register_builtin_consumers(mc.rtoptions());
    std::unique_ptr<Receiver> receiver(new Receiver(mc.get_cluster(), std::move(programs)));
    service().attach_receiver(*receiver);
    for (uint32_t d = 0; d < receiver->capture_context().devices.size(); d++) {
        std::vector<Stream*> owned;
        for (auto& s : receiver->streams_) {
            if (s->captured.device_index == d) {
                owned.push_back(s.get());
            }
        }
        if (!owned.empty()) {
            receiver->ingest_threads_.emplace_back(&Receiver::ingest_thread, receiver.get(), std::move(owned));
        }
    }
    receiver->sync_thread_ = std::jthread([r = receiver.get()](std::stop_token stop) { r->sync_thread(stop); });
    receiver->programs_->start();
    return receiver;
}

Receiver::~Receiver() {
    programs_->quiesce([this](uint32_t device_index, uint32_t socket_index, RelayState state) {
        stream(device_index, socket_index).relay.store(state, std::memory_order_release);
    });
    for (auto& t : ingest_threads_) {
        t.join();
    }
    sync_thread_.request_stop();
    wake_sync_thread();
    sync_thread_.join();
    service().detach_receiver(*this);
    for (uint32_t d = 0; d < capture_context().devices.size(); d++) {
        programs_->verify_completeness(d);
    }
    log_report();
    const uint64_t foreign = llrt::ZoneMetaRegistry::instance().foreign_sections();
    const uint64_t collisions = llrt::ZoneMetaRegistry::instance().collisions();
    if (collisions != 0 || foreign != 0) {
        log_warning(
            tt::LogMetal,
            "[streaming profiler] zone names: {} id collisions, {} foreign metadata sections ignored (the JIT "
            "cache holds ELFs from a different .tt_zone_meta layout)",
            collisions,
            foreign);
    }
}

bool Receiver::poll(Stream& s) {
    // pages_available counts from the last ack, so it includes the pages already seen
    const uint64_t arrived = s.acked + s.captured.socket->pages_available();
    if (arrived == s.arrived) {
        return false;
    }
    s.arrived = arrived;
    s.arrived_bytes.store(arrived * kPageBytes, std::memory_order_release);
    s.fullest = std::max(s.fullest, arrived - s.acked);
    if (s.consumed < arrived) {
        __builtin_prefetch(s.page(s.consumed));
    }
    return true;
}

// Frames are walked one per stream per round so the header loads of a device's streams are in flight together: a
// frame's header is a dependent DRAM miss, and walked back to back they would serialize at that latency. Every
// landed page below `arrived` is a frame header or inside the frame before it: the relay notifies only bytes the
// PCIe tile has acknowledged.
bool Receiver::walk_frame(Stream& s) {
    namespace kp = kernel_profiler;
    if (s.consumed >= s.arrived) {
        return false;
    }
    const uint32_t* page = reinterpret_cast<const uint32_t*>(s.page(s.consumed));
    const uint32_t w0 = page[0];
    const uint32_t w1 = page[1];
    TT_FATAL(
        pp_is_bulkspan(w0) && w1 >= kp::SPSC_SPAN_WIRE_CTRL_WORDS && w1 <= profiler::kSpscMaxPayloadWords,
        "streaming profiler: device {} socket {} page {} is not a frame header ({:#010x} {:#010x}); {} of {} pages "
        "landed",
        s.captured.device_index,
        s.captured.socket_index,
        s.consumed,
        w0,
        w1,
        s.arrived - s.acked,
        s.capacity);
    const uint32_t fw = kp::spsc_span_frame_words(w1);
    const uint64_t frame_pages = fw / kPageWords;
    if (s.arrived - s.consumed < frame_pages) {
        return false;
    }
    s.frames++;
    const uint64_t start = s.consumed * kPageBytes;
    if (start / kMarkBytes != s.mark_block) {
        s.mark_block = start / kMarkBytes;
        s.marks[s.mark_block % s.marks.size()].store(start, std::memory_order_release);
    }
    s.consumed += frame_pages;
    // The next header is known; the ones after are guessed at the same stride, which a relay shipping a core per
    // sweep hits every time, so several of the walk's dependent misses are in flight instead of one. Never past
    // `arrived`: the device rewrites those pages before the walk reaches them.
    for (uint64_t p = s.consumed; p < s.arrived && p <= s.consumed + 3 * frame_pages; p += frame_pages) {
        __builtin_prefetch(s.page(p));
    }
    // Credits go back as the walk earns them: a pass over eight sockets can run long once any of them is behind, and
    // credits held until its end would let the others fill meanwhile.
    if (s.consumed - s.acked >= kAckBatchPages) {
        s.captured.socket->pop(static_cast<uint32_t>(s.consumed - s.acked), true);
        s.acked = s.consumed;
    }
    return true;
}

bool Receiver::publish(Stream& s) {
    const uint64_t walked = s.consumed * kPageBytes;
    if (walked == s.walked_bytes.load(std::memory_order_relaxed)) {
        return false;
    }
    s.walked_bytes.store(walked, std::memory_order_release);
    return true;
}

// Credits only ever follow the walk, and a drained relay reports done only when every byte it sent is credited, so
// done means the walk has consumed every landed page.
bool Receiver::settle(Stream& s) {
    const bool published = publish(s);
    const RelayState relay = s.relay.load(std::memory_order_acquire);
    if (relay == RelayState::Running) {
        return published;
    }
    if (s.acked < s.consumed) {
        s.captured.socket->pop(static_cast<uint32_t>(s.consumed - s.acked), true);
        s.acked = s.consumed;
    }
    s.retired = relay == RelayState::Done;
    return published;
}

uint64_t Receiver::live_head(uint32_t stream) const {
    const Stream& s = *streams_[stream];
    return widen_head(s.arrived_bytes.load(std::memory_order_acquire), s.captured.socket->bytes_sent());
}

void Receiver::ingest_thread(std::vector<Stream*> streams) {
    std::string name = "sp-ingest:";
    for (Stream* s : streams) {
        name += std::to_string(s->captured.device_index) + "." + std::to_string(s->captured.socket_index) + ",";
    }
    name.pop_back();
    set_thread_name(name);
    prctl(PR_SET_TIMERSLACK, 1000);  // default 50 us slack would round every probe sleep up to it
    // The sockets bind their FIFOs to the device's node; walked from the other node, the headers' dependent misses
    // run at half the rate and the FIFOs fill.
    if (const int node = streams.front()->captured.numa_node; node >= 0 && numa_available() != -1) {
        numa_run_on_node(node);
    }
    IdleBackoff backoff(kProbeSleepCapUs, 0);
    // Frames published since the consumers, and since the sync thread, were last woken.
    uint64_t unwoken = 0, unwoken_sync = 0;
    while (true) {
        bool any = false;
        for (Stream* s : streams) {
            any |= poll(*s);
        }
        uint64_t frames_before = 0, sync_frames_before = 0;
        for (const Stream* s : streams) {
            (s->captured.sync ? sync_frames_before : frames_before) += s->frames;
        }
        for (bool progress = true; progress;) {
            progress = false;
            for (Stream* s : streams) {
                progress |= walk_frame(*s);
            }
        }
        for (Stream* s : streams) {
            settle(*s);
            (s->captured.sync ? unwoken_sync : unwoken) += s->frames;
        }
        unwoken -= frames_before;
        unwoken_sync -= sync_frames_before;
        // A wake goes out once a batch of frames is ready, or once a pass finds nothing new so none are held back.
        if (unwoken != 0 && (unwoken >= kBatchFrames || !any)) {
            service().wake_walkers();
            unwoken = 0;
        }
        if (unwoken_sync != 0 && (unwoken_sync >= kBatchFrames || !any)) {
            wake_sync_thread();
            unwoken_sync = 0;
        }
        std::erase_if(streams, [](const Stream* stream) { return stream->retired; });
        if (streams.empty()) {
            break;
        }
        if (any) {
            backoff.reset();
            continue;
        }
        backoff.idle();
    }
}

void Receiver::wake_sync_thread() {
    sync_wake_.fetch_add(1, std::memory_order_release);
    sync_wake_.notify_one();
}

void Receiver::sync_thread(std::stop_token stop) {
    set_thread_name("sp-sync");
    struct SyncStream {
        uint32_t index = 0;
        uint64_t cursor = 0, dropped = 0;
    };
    std::vector<SyncStream> sync_streams;
    for (uint32_t index = 0; index < streams_.size(); index++) {
        if (streams_[index]->captured.sync) {
            sync_streams.push_back({.index = index});
        }
    }
    std::array<uint32_t, kBatchFrames> frame_words{};
    const auto frames = std::make_unique_for_overwrite<std::byte[]>(kSyncBatchBytes);
    const auto take_batch = [&](SyncStream& sync) {
        const Stream& s = *streams_[sync.index];
        const Walked walked = walk_frames(
            s.fifo,
            sync.cursor,
            s.walked_bytes.load(std::memory_order_acquire),
            s.marks,
            std::span<std::byte>(frames.get(), kSyncBatchBytes),
            frame_words,
            [&] { return live_head(sync.index); });
        if (walked.cursor == sync.cursor) {
            return false;
        }
        sync.cursor = walked.cursor;
        sync.dropped += walked.dropped;
        const uint32_t dev = s.captured.device_index;
        const CoreTable& core_of_xy = capture_context().devices[dev].core_of_xy;
        const std::byte* frame_bytes = frames.get();
        for (uint32_t i = 0; i < walked.frames; i++) {
            const uint32_t* frame = reinterpret_cast<const uint32_t*>(frame_bytes);
            const uint32_t record_count = frame[kernel_profiler::SPSC_PREFIX_SYNC_RECORD_COUNT];
            const uint32_t core = core_of_xy.find(frame[kernel_profiler::SPSC_PREFIX_XY]);
            TT_FATAL(
                kernel_profiler::SPSC_SPAN_PREFIX_WORDS + record_count * kernel_profiler::kSyncRecordWords <=
                    frame_words[i],
                "streaming profiler: a sync frame of chip {} from core {:#x} holds {} records in {} words",
                capture_context().devices[dev].chip_id,
                frame[kernel_profiler::SPSC_PREFIX_XY],
                record_count,
                frame_words[i]);
            const auto* records =
                reinterpret_cast<const kernel_profiler::SyncRecord*>(frame + kernel_profiler::SPSC_SPAN_PREFIX_WORDS);
            for (uint32_t r = 0; r < record_count; r++) {
                clock_solver_.on_record(dev, core, records[r]);
            }
            frame_bytes += size_t{frame_words[i]} * sizeof(uint32_t);
        }
        if (clock_solver_.on_batch_end()) {
            service().wake_walkers();
        }
        return true;
    };
    const auto refclk_burst = [&] {
        const bool host_line = clock_solver_.on_refclk_burst(root_refclk_.read_burst());
        service().steady().sample();
        return host_line;
    };
    std::chrono::steady_clock::time_point next_burst{};
    while (true) {
        const uint32_t seen = sync_wake_.load(std::memory_order_acquire);
        // The ingest threads have published every frame before the stop, so a pass after it that finds nothing has
        // drained the streams.
        const bool stopping = stop.stop_requested();
        bool any = false;
        for (SyncStream& sync : sync_streams) {
            any |= take_batch(sync);
        }
        if (const auto now = std::chrono::steady_clock::now(); now >= next_burst) {
            next_burst = now + kRefclkBurstPeriod;
            refclk_burst();
        }
        if (any) {
            continue;
        }
        if (stopping) {
            break;
        }
        sync_wake_.wait(seen, std::memory_order_acquire);
    }
    // The closing bursts extend the host series past the capture's last records.
    while (!refclk_burst()) {
    }
    clock_solver_.on_capture_end();
    for (const SyncStream& sync : sync_streams) {
        if (sync.dropped != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] the clock solver missed {} bytes of chip {}'s sync stream",
                sync.dropped,
                capture_context().devices[streams_[sync.index]->captured.device_index].chip_id);
        }
    }
}

Receiver::Stream& Receiver::stream(uint32_t device_index, uint32_t socket_index) {
    for (auto& s : streams_) {
        if (s->captured.device_index == device_index && s->captured.socket_index == socket_index) {
            return *s;
        }
    }
    TT_THROW("streaming profiler: no stream for device {} socket {}", device_index, socket_index);
}

void Receiver::log_report() const {
    uint64_t pages = 0, frames = 0;
    std::string fill;
    for (const auto& s : streams_) {
        fill += fmt::format("{}{}%", fill.empty() ? "" : " ", s->fullest * 100 / s->capacity);
        pages += s->arrived;
        frames += s->frames;
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] capture: {} frames, {:.1f} MB from {} device(s); FIFO high-water marks {}",
        frames,
        pages * static_cast<double>(kPageBytes) / 1e6,
        capture_context().devices.size(),
        fill);
}

}  // namespace tt::tt_metal::streaming_profiler
