// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/service.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <thread>
#include <utility>
#include <pthread.h>

#include <tracy/Tracy.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "impl/streaming_profiler/decode.hpp"
#include "impl/streaming_profiler/ops_csv.hpp"
#include "impl/streaming_profiler/receiver.hpp"
#include "impl/streaming_profiler/tracy_consumer.hpp"
#include "impl/streaming_profiler/zone_csv.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

thread_local experimental::streaming_profiler::detail::CallbackId t_consumer_id{};

}  // namespace

void set_thread_name(const std::string& name) {
    tracy::SetThreadName(name.c_str());
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%s", name.c_str());
    pthread_setname_np(pthread_self(), buf);
}

struct Service::AttachedStream {
    std::optional<StreamDecoder> decoder;
    uint64_t cursor = 0;
    uint64_t dropped = 0;
    uint64_t order_regressions = 0;
    uint32_t dev = 0;
    uint32_t index = 0;
    std::deque<Parked*> pending;
    int64_t final_through = std::numeric_limits<int64_t>::min();
};
struct Service::Attached {
    Receiver* receiver = nullptr;
    std::vector<std::unique_ptr<AttachedStream>> streams;
    ClockMap::Reader reader;
};
struct Service::Parked {
    Attached* attached;
    uint32_t dev;
    bool delivered;
    std::unique_ptr<uint8_t[]> buffer;
    StreamDecoder::Out out;
    StreamDecoder::Produced produced;
};

constexpr size_t kBatchMaxWords = size_t{kBatchFrames} * profiler::kSpscMaxFrameWords;

namespace {

// The most output a single batch can produce.
constexpr StreamDecoder::Capacity kFullBatch = StreamDecoder::out_capacity(kBatchMaxWords, kBatchFrames);

// A parked batch's decoded output: one buffer holding a full batch's regions, each starting on a cache line.
constexpr size_t cache_lines(size_t bytes) { return (bytes + 63) / 64 * 64; }
constexpr size_t kEventsAt = cache_lines(kFullBatch.zone_bytes);
constexpr size_t kDataAt = kEventsAt + cache_lines(kFullBatch.event_bytes);
constexpr size_t kValuesAt = kDataAt + cache_lines(kFullBatch.data_bytes);
constexpr size_t kOutBufferBytes = kValuesAt + kFullBatch.value_bytes;

StreamDecoder::Out out_in(uint8_t* buffer) {
    return {buffer, buffer + kEventsAt, buffer + kDataAt, buffer + kValuesAt};
}

}  // namespace

struct Service::Consumer {
    std::string name;
    experimental::streaming_profiler::RecordType types{};
    BatchCallback callback;
    experimental::streaming_profiler::detail::CallbackId id{};
    std::thread thread;
    std::atomic<bool> stop{false};
    std::atomic<bool> control_pending{false};
    std::mutex control_mu;
    std::vector<std::pair<Receiver*, bool>> control;  // (receiver, attach), in order
};

Service::Service() : steady_(new SteadyClock) { init_site_registry(); }

SteadyClock& Service::steady() { return *steady_; }

const char* Service::plot_name(const std::string& name) {
    std::lock_guard<std::mutex> lk(plot_names_mu_);
    return plot_names_.insert(name).first->c_str();
}

void Service::set_tile_clocks(ContextId context_id, uint32_t chip, TileClocks clocks) {
    std::lock_guard<std::mutex> lk(tile_clocks_mu_);
    tile_clocks_.emplace(std::pair{context_id, chip}, std::move(clocks));
}

const TileClocks* Service::tile_clocks(ContextId context_id, uint32_t chip) const {
    std::lock_guard<std::mutex> lk(tile_clocks_mu_);
    const auto it = tile_clocks_.find({context_id, chip});
    return it == tile_clocks_.end() ? nullptr : &it->second;
}

Service& service() {
    static ttsl::Indestructible<Service> instance;
    return instance.get();
}

void Service::post_control(Consumer& c, Receiver* receiver, bool attach) {
    {
        std::lock_guard<std::mutex> lk(c.control_mu);
        c.control.emplace_back(receiver, attach);
    }
    c.control_pending.store(true, std::memory_order_release);
    pending_acks_++;
    wake_walkers();
}

void Service::wait_acks(std::unique_lock<std::mutex>& lk) {
    ack_cv_.wait(lk, [&] { return pending_acks_ == 0; });
}

experimental::streaming_profiler::detail::CallbackId Service::add_consumer(
    std::string name, experimental::streaming_profiler::RecordType types, BatchCallback callback) {
    TT_FATAL(
        t_consumer_id == experimental::streaming_profiler::detail::CallbackId{},
        "streaming profiler: add_consumer must not be called from a consumer callback");
    std::lock_guard<std::mutex> topo(topology_mu_);
    auto owned = std::make_unique<Consumer>();
    owned->name = std::move(name);
    owned->types = types;
    owned->callback = std::move(callback);
    Consumer& consumer = *owned;
    std::unique_lock<std::mutex> lk(mu_);
    consumer.id = experimental::streaming_profiler::detail::CallbackId{next_id_++};
    for (Receiver* receiver : receivers_) {
        post_control(consumer, receiver, true);
    }
    consumers_.push_back(std::move(owned));
    consumer.thread = std::thread(&Service::consumer_thread, this, std::ref(consumer));
    wait_acks(lk);
    return consumer.id;
}

void Service::remove_consumer(experimental::streaming_profiler::detail::CallbackId id) {
    const bool self = id == t_consumer_id;
    TT_FATAL(
        self || t_consumer_id == experimental::streaming_profiler::detail::CallbackId{},
        "streaming profiler: a consumer callback may unregister only itself");
    std::unique_lock<std::mutex> topo(topology_mu_, std::defer_lock);
    if (!self) {
        topo.lock();
    }
    std::unique_ptr<Consumer> victim;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = std::find_if(
            consumers_.begin(), consumers_.end(), [&](const auto& consumer) { return consumer->id == id; });
        if (self) {
            (*it)->stop.store(true, std::memory_order_release);
            return;
        }
        victim = std::move(*it);
        consumers_.erase(it);
    }
    victim->stop.store(true, std::memory_order_release);
    wake_walkers();
    victim->thread.join();
}

void Service::attach_receiver(Receiver& receiver) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    std::unique_lock<std::mutex> lk(mu_);
    receivers_.push_back(&receiver);
    for (auto& c : consumers_) {
        post_control(*c, &receiver, true);
    }
    wait_acks(lk);
}

void Service::detach_receiver(Receiver& receiver) {
    std::lock_guard<std::mutex> topo(topology_mu_);
    bool last = false;
    {
        std::unique_lock<std::mutex> lk(mu_);
        for (auto& c : consumers_) {
            post_control(*c, &receiver, false);
        }
        wait_acks(lk);
        std::erase(receivers_, &receiver);
        last = receivers_.empty();
    }
    if (last) {
        for (const auto& write : file_sinks_) {
            write();
        }
    }
}

bool Service::is_active() const {
    std::lock_guard<std::mutex> lk(mu_);
    return !receivers_.empty();
}

void Service::register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions) {
    std::call_once(builtins_once_, [&] {
        auto add_sink = [&]<typename Sink>(const char* name, const std::shared_ptr<Sink>& sink) {
            builtin_callbacks_.push_back(experimental::streaming_profiler::RegisterCallback(
                [sink](const typename Sink::Batch& batch) { (*sink)(batch); }, name));
        };
        auto add_file_sink = [&]<typename Sink>(const char* name, const std::shared_ptr<Sink>& sink) {
            add_sink(name, sink);
            file_sinks_.push_back([sink] { sink->write_csv(); });
        };
#if defined(TRACY_ENABLE)
        if (rtoptions.get_streaming_profiler_tracy_enabled()) {
            add_sink("tracy", std::make_shared<TracyConsumer>());
        }
#endif
        if (const std::string& path = rtoptions.get_streaming_profiler_zone_csv_path(); !path.empty()) {
            add_file_sink("zone-csv", std::make_shared<ZoneCsvConsumer>(path));
        }
        if (const std::string& path = rtoptions.get_streaming_profiler_ops_csv_path(); !path.empty()) {
            add_file_sink("ops-csv", std::make_shared<OpsCsvConsumer>(path));
        }
    });
}

class Service::ConsumerLoop {
public:
    ConsumerLoop(Service& service, Consumer& consumer) :
        service_(service),
        consumer_(consumer),
        frames_buf_(std::make_unique_for_overwrite<std::byte[]>(kFramesBytes)) {}

    void run() {
        set_thread_name("sp-con:" + consumer_.name);
        IdleBackoff backoff(0);
        while (true) {
            const uint32_t seen = service_.wake_token();
            if (consumer_.control_pending.load(std::memory_order_acquire)) {
                consumer_.control_pending.store(false, std::memory_order_release);
                apply_control();
            }
            if (consumer_.stop.load(std::memory_order_acquire)) {
                break;
            }
            bool any = false;
            for (auto& attached : attached_) {
                any |= pass(*attached);
            }
            any |= drain();
            if (any) {
                backoff.reset();
                continue;
            }
            if (backoff.spin()) {
                continue;
            }
            service_.wait_wake(seen);
        }
        for (const auto& attached : attached_) {
            report(*attached);
        }
    }

private:
    static constexpr size_t kFramesBytes = kBatchMaxWords * sizeof(uint32_t);

    void apply_control() {
        std::vector<std::pair<Receiver*, bool>> control;
        {
            std::lock_guard<std::mutex> lk(consumer_.control_mu);
            control.swap(consumer_.control);
        }
        for (const auto& [receiver, attach_it] : control) {
            attach_it ? attach(receiver) : detach(receiver);
        }
        std::lock_guard<std::mutex> lk(service_.mu_);
        service_.pending_acks_ -= control.size();
        service_.ack_cv_.notify_all();
    }

    void attach(Receiver* receiver) {
        auto attached = std::make_unique<Attached>();
        attached->receiver = receiver;
        attached->reader = receiver->clock_map().reader();
        const CaptureContext& ctx = receiver->capture_context();
        const auto sources = receiver->streams();
        for (uint32_t index = 0; index < sources.size(); index++) {
            const ReceiverStream& source = sources[index];
            if (source.sync) {
                continue;
            }
            auto stream = std::make_unique<AttachedStream>();
            stream->index = index;
            stream->cursor = source.walked->load(std::memory_order_acquire);
            stream->dev = source.dev;
            stream->decoder.emplace(ctx.devices[source.dev], consumer_.types);
            attached->streams.push_back(std::move(stream));
        }
        attached_.push_back(std::move(attached));
    }

    // The receiver has finished its clock solver, so every placement is final.
    void detach(Receiver* receiver) {
        auto it = std::find_if(
            attached_.begin(), attached_.end(), [&](const auto& entry) { return entry->receiver == receiver; });
        Attached& attached = **it;
        while (pass(attached)) {
        }
        for (auto& stream : attached.streams) {
            for (Parked* parked : stream->pending) {
                deliver(*parked);
            }
            stream->pending.clear();
        }
        release_delivered();
        report(attached);
        attached_.erase(it);
    }

    // One batch per stream per pass, so no stream gets lapped while another is being drained.
    bool pass(Attached& attached) {
        bool any = false;
        for (const auto& stream : attached.streams) {
            any |= take_batch(attached, *stream);
        }
        return any;
    }

    bool take_batch(Attached& attached, AttachedStream& stream) {
        const ReceiverStream& source = attached.receiver->streams()[stream.index];
        const uint64_t end = source.walked->load(std::memory_order_acquire);
        if (end == stream.cursor) {
            return false;
        }
        const Walked walked = walk_frames(
            source.fifo,
            stream.cursor,
            end,
            source.marks,
            std::span<std::byte>(frames_buf_.get(), kFramesBytes),
            frame_words_,
            [&] { return attached.receiver->live_head(stream.index); });
        if (walked.cursor == stream.cursor) {
            return false;
        }
        stream.cursor = walked.cursor;
        stream.dropped += walked.dropped;
        std::unique_ptr<uint8_t[]> buffer = take_buffer();
        const StreamDecoder::Out out = out_in(buffer.get());
        const StreamDecoder::Produced produced = stream.decoder->decode_frames(
            reinterpret_cast<const uint32_t*>(frames_buf_.get()),
            std::span<const uint32_t>(frame_words_.data(), walked.frames),
            out);
        stream.order_regressions += produced.order_regressions;
        undelivered_dropped_bytes_ += walked.dropped;
        undelivered_stalls_ += produced.stalls;
        parked_.push_back(Parked{
            .attached = &attached,
            .dev = stream.dev,
            .delivered = false,
            .buffer = std::move(buffer),
            .out = out,
            .produced = produced});
        stream.pending.push_back(&parked_.back());
        return true;
    }

    void report(const Attached& attached) const {
        uint64_t dropped = 0, order_regressions = 0;
        for (const auto& stream : attached.streams) {
            order_regressions += stream->order_regressions;
            dropped += stream->dropped;
        }
        if (dropped != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] consumer \"{}\" missed {} bytes of frames",
                consumer_.name,
                dropped);
        }
        if (order_regressions != 0) {
            log_warning(
                tt::LogMetal,
                "[streaming profiler] consumer \"{}\": {} order regressions",
                consumer_.name,
                order_regressions);
        }
    }

    bool drain() {
        bool any = false;
        for (auto& attached : attached_) {
            const ClockMap& map = attached->receiver->clock_map();
            for (auto& owned : attached->streams) {
                AttachedStream& stream = *owned;
                while (!stream.pending.empty()) {
                    Parked& parked = *stream.pending.front();
                    const int64_t newest = parked.produced.newest_ticks;
                    if (newest > stream.final_through) {
                        if (!map.is_final(attached->reader, stream.dev, newest)) {
                            break;
                        }
                        stream.final_through = newest;
                    }
                    deliver(parked);
                    stream.pending.pop_front();
                    stream.decoder->commit();
                    any = true;
                }
            }
        }
        release_delivered();
        return any;
    }

    void deliver(Parked& parked) {
        place(parked);
        const StreamDecoder::Out& out = parked.out;
        const StreamDecoder::Produced& produced = parked.produced;
        const experimental::streaming_profiler::detail::BatchData batch{
            .zones = std::launder(reinterpret_cast<const experimental::streaming_profiler::Zone*>(out.zones)),
            .zone_count = produced.zones,
            .timestamped_data =
                std::launder(reinterpret_cast<const experimental::streaming_profiler::TimestampedData*>(out.data)),
            .timestamped_data_count = produced.data,
            .events = std::launder(reinterpret_cast<const experimental::streaming_profiler::Event*>(out.events)),
            .event_count = produced.events,
            .dropped_bytes = std::exchange(undelivered_dropped_bytes_, 0),
            .stall_count = std::exchange(undelivered_stalls_, 0)};
        try {
            if (!consumer_.stop.load(std::memory_order_relaxed)) {
                consumer_.callback(batch);
            }
        } catch (const std::exception& ex) {
            log_warning(tt::LogMetal, "[streaming profiler] consumer \"{}\" threw: {}", consumer_.name, ex.what());
        }
        parked.delivered = true;
    }

    // The decoder leaves each record's tile offset in its tsc_ slot, and placement overwrites it with the host time.
    void place(Parked& parked) {
        const ClockMap& map = parked.attached->receiver->clock_map();
        ClockMap::Reader& reader = parked.attached->reader;
        const uint32_t dev = parked.dev;
        const StreamDecoder::Out& out = parked.out;
        const StreamDecoder::Produced& produced = parked.produced;
        using namespace profiler;
        // A memmove onto itself compiles to nothing and implicitly creates the zones, events and data-record array
        // already in these bytes.
        std::memmove(out.zones, out.zones, size_t{produced.zones} * kSpscZoneBytes);
        std::memmove(out.events, out.events, size_t{produced.events} * kSpscEventBytes);
        std::memmove(out.data, out.data, size_t{produced.data} * kSpscDataBytes);
        const auto wall_of = [](const uint64_t* record) {
            return static_cast<int64_t>(record[kSpscQwTimestamp] + record[kSpscQwTsc]);
        };
        const auto host_tsc = [&](const uint8_t* record) {
            return map.place_host(reader, dev, wall_of(reinterpret_cast<const uint64_t*>(record)));
        };
        for (uint32_t i = 0; i < produced.zones; i++) {
            uint64_t* const zone = reinterpret_cast<uint64_t*>(out.zones + size_t{i} * kSpscZoneBytes);
            const int64_t wall = wall_of(zone);
            const int64_t start = map.place_host(reader, dev, wall);
            const int64_t end = map.place_host(reader, dev, wall + static_cast<int64_t>(zone[kSpscQwDuration]));
            zone[kSpscQwTsc] = static_cast<uint64_t>(start);
            zone[kSpscQwEndTsc] = static_cast<uint64_t>(end);
        }
        for (uint32_t i = 0; i < produced.events; i++) {
            uint8_t* const event = out.events + size_t{i} * kSpscEventBytes;
            reinterpret_cast<uint64_t*>(event)[kSpscQwTsc] = static_cast<uint64_t>(host_tsc(event));
        }
        for (uint32_t i = 0; i < produced.data; i++) {
            uint8_t* const record = out.data + size_t{i} * kSpscDataBytes;
            construct_timestamped_data(record, host_tsc(record));
        }
    }

    // The most recently freed buffer is still in cache and already faulted in.
    std::unique_ptr<uint8_t[]> take_buffer() {
        if (free_buffers_.empty()) {
            return std::make_unique_for_overwrite<uint8_t[]>(kOutBufferBytes);
        }
        std::unique_ptr<uint8_t[]> buffer = std::move(free_buffers_.back());
        free_buffers_.pop_back();
        return buffer;
    }

    void release_delivered() {
        while (!parked_.empty() && parked_.front().delivered) {
            free_buffers_.push_back(std::move(parked_.front().buffer));
            parked_.pop_front();
        }
    }

    Service& service_;
    Consumer& consumer_;
    std::vector<std::unique_ptr<Attached>> attached_;
    std::array<uint32_t, kBatchFrames> frame_words_{};
    std::unique_ptr<std::byte[]> frames_buf_;
    std::deque<Parked> parked_;
    std::vector<std::unique_ptr<uint8_t[]>> free_buffers_;
    // Handed to the next callback, whichever batch it gets, so each call reports what is new since the last one.
    uint64_t undelivered_dropped_bytes_ = 0, undelivered_stalls_ = 0;
};

void Service::consumer_thread(Consumer& consumer) {
    t_consumer_id = consumer.id;
    ConsumerLoop(*this, consumer).run();
    // A consumer that unregistered itself stays listed until here, so attach_receiver and detach_receiver may have
    // posted it controls that only this thread can acknowledge.
    std::unique_ptr<Consumer> self;
    std::lock_guard<std::mutex> lk(mu_);
    const auto it =
        std::find_if(consumers_.begin(), consumers_.end(), [&](const auto& entry) { return entry.get() == &consumer; });
    if (it == consumers_.end()) {
        return;
    }
    {
        std::lock_guard<std::mutex> control_lock(consumer.control_mu);
        pending_acks_ -= consumer.control.size();
        consumer.control.clear();
    }
    ack_cv_.notify_all();
    self = std::move(*it);
    consumers_.erase(it);
    self->thread.detach();
}

}  // namespace tt::tt_metal::streaming_profiler
