// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <tt_stl/tt_pause.hpp>

#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/capture_context.hpp"
#include "impl/streaming_profiler/sync/tile_sync.hpp"

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::tt_metal::streaming_profiler {

class Receiver;
class SteadyClock;

void set_thread_name(const std::string& name);
void init_site_registry();

// How a polling thread waits for work: `spins` empty polls, then sleeps growing to `cap_us`.
inline constexpr uint32_t kEmptyPollsBeforeSleep = 1000;
struct IdleBackoff {
    uint32_t cap_us;
    uint32_t spins;
    uint32_t empty_polls = 0;
    uint32_t sleep_us = 1;
    explicit IdleBackoff(uint32_t cap, uint32_t spins = kEmptyPollsBeforeSleep) : cap_us(cap), spins(spins) {}
    void idle() {
        if (++empty_polls < spins) {
            ttsl::pause();
        } else {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            sleep_us = std::min(sleep_us + sleep_us / 4 + 1, cap_us);
        }
    }
    // The spin phase alone: true while the caller should poll again, false once it should park.
    bool spin() {
        if (++empty_polls < spins) {
            ttsl::pause();
            return true;
        }
        return false;
    }
    void reset() {
        empty_polls = 0;
        sleep_us = 1;
    }
};

// The device's 64-bit stream position from its 32-bit bytes_sent word and a position the ingest thread has already
// observed: the device is never more than one FIFO (2 GB at most) past what it has been credited, so the difference
// fits 32 bits.
inline uint64_t widen_head(uint64_t observed, uint32_t bytes_sent) {
    return observed + static_cast<uint32_t>(bytes_sent - static_cast<uint32_t>(observed));
}

// Everything the device has landed lies below head + SPSC_NOTIFY_CAP_BYTES (the relay writes bytes_sent before it
// pushes more than that), and a frame at `offset` is overwritten only by writes from offset + fifo_bytes on.
inline bool frame_intact(uint64_t offset, uint64_t fifo_bytes, uint64_t head) {
    return head + kernel_profiler::SPSC_NOTIFY_CAP_BYTES <= offset + fifo_bytes;
}

// The most frames a consumer or a receiver's sync thread takes from a stream in one pass.
inline constexpr uint32_t kBatchFrames = 64;

// The ingest records the first frame boundary in every kMarkBytes of a stream, so a consumer the device has lapped
// resumes at the oldest boundary past the overwrite horizon instead of at the ingest's position.
inline constexpr uint64_t kMarkBytes = 64 * 1024;
static_assert(kMarkBytes >= profiler::kSpscMaxFrameWords * 4, "every block holds at least one frame boundary");

// The first recorded boundary in [from, end], or end.
inline uint64_t resume_mark(std::span<const std::atomic<uint64_t>> marks, uint64_t from, uint64_t end) {
    for (uint64_t b = from / kMarkBytes; !marks.empty() && b * kMarkBytes < end; b++) {
        const uint64_t v = marks[b % marks.size()].load(std::memory_order_acquire);
        if (v >= from && v <= end) {
            return v;
        }
    }
    return end;
}

struct Walked {
    uint32_t frames = 0;   // copied into `out`, back to back
    size_t bytes = 0;      // their total length
    uint64_t dropped = 0;  // bytes skipped because the device had reached them
    uint64_t cursor = 0;   // where the next pass starts
};

// Copies whole frames from `cursor` up to `end` (the ingest's walk position, always a frame boundary) into `out`, at
// most frame_words.size() of them, recording each one's length in words. The FIFO (a power-of-two size) is read in
// place while the device keeps writing it, so a frame the device has reached is never trusted: a cursor the device
// has passed resumes at the first recorded boundary an eighth of the FIFO past the overwrite horizon (the device keeps
// writing during the copy) and counts what it skipped; if the device reached the pass's first frame during the copy
// the lengths read from those pages are void, so the pass is discarded and the next one resumes further on. `out`
// holds frame_words.size() frames of kSpscMaxFrameWords.
template <typename LiveHead>
Walked walk_frames(
    std::span<const std::byte> fifo,
    uint64_t cursor,
    uint64_t end,
    std::span<const std::atomic<uint64_t>> marks,
    std::span<std::byte> out,
    std::span<uint32_t> frame_words,
    LiveHead live_head) {
    Walked w{.cursor = cursor};
    if (cursor >= end) {
        return w;
    }
    uint64_t start = cursor;
    if (const uint64_t head = live_head(); !frame_intact(start, fifo.size(), head)) {
        const uint64_t written = head + kernel_profiler::SPSC_NOTIFY_CAP_BYTES;
        const uint64_t horizon = written > fifo.size() ? written - fifo.size() : 0;
        start = resume_mark(marks, horizon + fifo.size() / 8, end);
        w.dropped = start - cursor;
        w.cursor = start;
        if (start >= end) {
            return w;
        }
    }
    const size_t mask = fifo.size() - 1;
    while (w.cursor < end && w.frames < frame_words.size()) {
        uint32_t w1;
        std::memcpy(&w1, fifo.data() + ((w.cursor + 4) & mask), 4);
        const uint32_t fw = kernel_profiler::spsc_span_frame_words(w1);
        const size_t bytes = size_t{fw} * 4;
        if (w1 > profiler::kSpscMaxPayloadWords || bytes > end - w.cursor) {
            break;
        }
        const size_t at = w.cursor & mask;
        const size_t first = std::min(bytes, fifo.size() - at);
        std::memcpy(out.data() + w.bytes, fifo.data() + at, first);
        std::memcpy(out.data() + w.bytes + first, fifo.data(), bytes - first);
        frame_words[w.frames++] = fw;
        w.bytes += bytes;
        w.cursor += bytes;
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    if (!frame_intact(start, fifo.size(), live_head())) {
        w = Walked{.dropped = start - cursor, .cursor = start};
    }
    return w;
}

// One stream of a capture: its FIFO and how far the ingest has walked it. Every frame below `walked` has a valid
// header, so a consumer reads frames in place from its own cursor up to there.
struct ReceiverStream {
    std::span<const std::byte> fifo;
    const std::atomic<uint64_t>* walked = nullptr;  // bytes, absolute
    uint32_t dev = 0;                               // index into capture_context().devices
    std::span<const std::atomic<uint64_t>> marks;   // frame boundaries, one per kMarkBytes of `fifo`
    // Set on the eth relay's sync socket. The sync thread reads it and the consumers skip it.
    bool sync = false;
};

using BatchCallback = std::function<void(const experimental::streaming_profiler::detail::BatchData&)>;

class Service {
public:
    Service();
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;

    // The consumer's decoders write only records of `types`, the ones its callback can read.
    experimental::streaming_profiler::detail::CallbackId add_consumer(
        std::string name, experimental::streaming_profiler::RecordType types, BatchCallback callback);
    // Returns once the callback can no longer run. A callback removing itself returns at once.
    void remove_consumer(experimental::streaming_profiler::detail::CallbackId id);

    // Returns once every consumer reads the receiver's streams, so nothing published afterwards is missed.
    void attach_receiver(Receiver& receiver);
    // Returns once every consumer has drained the receiver's streams and delivered the batches it held back, so the
    // receiver must have finished its clock map first. Detaching the last receiver writes the file sinks.
    void detach_receiver(Receiver& receiver);
    bool is_active() const;

    // The Tracy sink and the CSV writers rtoptions select; subsequent calls do nothing.
    void register_builtin_consumers(const tt::llrt::RunTimeOptions& rtoptions);
    void set_tile_clocks(ContextId context_id, uint32_t chip, TileClocks clocks);
    const TileClocks* tile_clocks(ContextId context_id, uint32_t chip) const;
    SteadyClock& steady();
    // Returns a copy of `name` that lives as long as the process.
    const char* plot_name(const std::string& name);

    // Wakes every consumer.
    void wake_walkers() {
        wake_gen_.fetch_add(1, std::memory_order_seq_cst);
        if (sleeping_walkers_.load(std::memory_order_seq_cst) != 0) {
            wake_gen_.notify_all();
        }
    }
    // Take the token before looking for work, so a wake that lands before wait_wake() isn't missed.
    uint32_t wake_token() const { return wake_gen_.load(std::memory_order_acquire); }
    void wait_wake(uint32_t seen) const {
        sleeping_walkers_.fetch_add(1, std::memory_order_seq_cst);
        wake_gen_.wait(seen, std::memory_order_seq_cst);
        sleeping_walkers_.fetch_sub(1, std::memory_order_acq_rel);
    }

private:
    struct Consumer;
    struct AttachedStream;
    struct Attached;
    struct Parked;
    class ConsumerLoop;
    void consumer_thread(Consumer& consumer);
    void post_control(Consumer& c, Receiver* receiver, bool attach);
    void wait_acks(std::unique_lock<std::mutex>& lk);

    // Serializes add/remove/attach/detach against each other; never taken by a consumer thread.
    std::mutex topology_mu_;
    // Guards the lists below and the ack handshake; taken briefly by consumer threads.
    mutable std::mutex mu_;
    std::condition_variable ack_cv_;
    uint64_t pending_acks_ = 0;
    alignas(64) std::atomic<uint32_t> wake_gen_{0};
    alignas(64) mutable std::atomic<uint32_t> sleeping_walkers_{0};
    std::vector<std::unique_ptr<Consumer>> consumers_;
    std::vector<Receiver*> receivers_;
    std::vector<std::function<void()>> file_sinks_;
    std::unique_ptr<SteadyClock> steady_;
    std::vector<experimental::streaming_profiler::Callback> builtin_callbacks_;
    uint64_t next_id_ = 1;
    std::once_flag builtins_once_;
    mutable std::mutex tile_clocks_mu_;
    std::map<std::pair<ContextId, uint32_t>, TileClocks> tile_clocks_;
    std::mutex plot_names_mu_;
    std::unordered_set<std::string> plot_names_;
};

// The process's Service. Subscriptions outlive every device and context, so it is never destroyed.
Service& service();

}  // namespace tt::tt_metal::streaming_profiler
