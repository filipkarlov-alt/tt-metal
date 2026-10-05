// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "hostdev/dev_msgs.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/dataflow_api.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

namespace link_sync {

static_assert(kernel_profiler::kEthRefclkHz == eth_ptp::kRefclkHz);

// A link's stamped frames use a TX queue and header row that nothing else uses. The firmware uses header rows 0 to 2
// (see eth_ptp.hpp). The fabric routers send on TX queue 0, and when a router runs on two ERISCs, its receiver uses
// queue 1.
constexpr uint32_t kLinkTxq = 2;
constexpr uint32_t kLinkHeaderRow = 3;
constexpr uint32_t kLinkTcamRow = 63;
constexpr uint32_t kLinkLabel = 0x15;
// The stamp frames' destination address, which no firmware frame uses (see eth_ptp.hpp).
constexpr uint64_t kStampFrameDestination = 0x02A5'A5A5'A5A5ull;
constexpr eth_ptp::RxTcamNonIpMatch kStampMatch =
    eth_ptp::rx_tcam_match_destination(kStampFrameDestination, 0x00FF'FFFF'FF00ull);
#if defined(PROFILE_STREAMING_SYNC_CHECK)
constexpr bool kSyncCheck = true;
#else
constexpr bool kSyncCheck = false;
#endif

constexpr uint32_t kTripsPerRound = 128;
constexpr uint32_t kBurstFrames = 4;
constexpr uint32_t kBurstsPerRound = kTripsPerRound / kBurstFrames;
constexpr uint32_t kFrameBytes = 32;
static_assert(kTripsPerRound % kBurstFrames == 0);

// The transmitter sends a frame with awaiting_echo set, and the receiver clears it and echoes the frame back. It comes
// last, so once it changes, everything before it has landed.
struct LinkFrame {
    eth_ptp::FrameStampSlot stamp;
    uint32_t round;
    uint32_t pad[2];
    uint32_t awaiting_echo;
};
static_assert(sizeof(LinkFrame) == kFrameBytes && kFrameBytes % 16 == 0);
static_assert(
    offsetof(kernel_profiler::LinkSyncL1, slots) == 0 &&
    kBurstFrames * kFrameBytes == sizeof(kernel_profiler::LinkSyncL1::slots));

// The sum is 64-bit because a round's 128 offsets from its first stamp can add up to more than 2^32 ns once the round
// spans about 34 ms, and on a loaded router a round can spread over tens of milliseconds.
struct StampSum {
    uint32_t count = 0;
    uint64_t first = 0;
    uint64_t sum_from_first = 0;
    FORCE_INLINE void reset() {
        count = 0;
        sum_from_first = 0;
    }
    FORCE_INLINE void add(uint64_t stamp) {
        if (count == 0) {
            first = stamp;
        }
        sum_from_first += stamp - first;
        count++;
    }
};

// Tracks the number of AICLK cycles per refclk update (four ticks), rounded to the nearest cycle, by re-reading the
// refclk every kRemeasureCycles. The reference wall clock is 64-bit because on a large mesh a link end's first frame
// can come minutes after start(), long enough for 32-bit wall and refclk differences to wrap.
struct UpdatePeriod {
    static constexpr uint32_t kRemeasureCycles = 1u << 20;
    static constexpr uint32_t kStartMeasureTicks = 1000;
    // A longer span is not measured, so its cycle count stays within 32 bits. The router can go that long without
    // stepping during a fabric pause, and the period then stays as it was until the next remeasure.
    static constexpr uint32_t kMaxMeasureCycles = 1u << 31;
    uint64_t reference_wall = 0;
    uint32_t cycles_per_update = 0, reference_refclk = 0;
    void start() {
        const eth_ptp::ClocksLo first = eth_ptp::await_refclk_update();
        while (eth_ptp::kRefclkLo.read() - first.refclk < kStartMeasureTicks) {
        }
        const eth_ptp::ClocksLo edge = eth_ptp::await_refclk_update();
        cycles_per_update = measure(edge.wall - first.wall, edge.refclk - first.refclk);
        take_reference();
    }
    FORCE_INLINE void remeasure_if_due() {
        if (eth_ptp::kWallClockLo.read() - static_cast<uint32_t>(reference_wall) >= kRemeasureCycles) {
            remeasure();
        }
    }

private:
    static FORCE_INLINE uint32_t measure(uint32_t cycles, uint32_t refclk_ticks) {
        const uint32_t updates = refclk_ticks / eth_ptp::kRefclkTicksPerUpdate;
        return (cycles + updates / 2) / updates;
    }
    FORCE_INLINE void take_reference() {
        const eth_ptp::Instant now = eth_ptp::read_instant();
        reference_wall = now.wall;
        reference_refclk = static_cast<uint32_t>(now.refclk);
    }
    __attribute__((noinline)) void remeasure() {
        const uint64_t prev_wall = reference_wall;
        const uint32_t prev_refclk = reference_refclk;
        take_reference();
        if (reference_wall - prev_wall < kMaxMeasureCycles) {
            cycles_per_update = measure(reference_wall - prev_wall, reference_refclk - prev_refclk);
        }
    }
};

// Every member is zero-initialised, so an end has no .data for the firmware to copy. start() sets the rest.
struct EndBase {
    eth_ptp::TxHeaderRow<kLinkTxq, kLinkHeaderRow> header;
    eth_ptp::RxStampRule<kLinkTcamRow, kLinkLabel> rule;
    volatile kernel_profiler::LinkSyncL1* l1 = nullptr;
    uint32_t round = 0;
    UpdatePeriod period;
    uint32_t random_state = 0;
    StampSum egress, ingress;
    uint32_t ring_tail = 0;

    // The host zeroes the slots, ctl and done before the end starts, because a stale awaiting_echo would stall the link
    // for good.
    void start(uint32_t link_l1) {
        l1 = reinterpret_cast<volatile kernel_profiler::LinkSyncL1*>(link_l1);
        eth_ptp::restart_ptp_timer();
        rule.install(kStampMatch);
        header.install(kStampFrameDestination);
        // The queue stays armed until stop(), so every frame the end sends is stamped and no slot needs clearing.
        eth_ptp::txq_arm_in_frame(kLinkTxq);
        period.start();
        random_state = eth_ptp::kWallClockLo.read() | 1u;
    }
    void stop() {
        eth_ptp::txq_disarm(kLinkTxq);
        header.restore();
        rule.remove();
    }

protected:
    // One refclk update (80 ns) takes at most 128 cycles at AICLK up to 1.6 GHz, and Blackhole's AICLK peaks at
    // 1.35 GHz.
    static constexpr uint32_t kMaxDitherCycles = 128;
    // A burst's frame i uses slot i, at the same L1 address on both ends, so an echo lands on the frame it answers.
    FORCE_INLINE volatile LinkFrame& frame(uint32_t slot) const {
        return reinterpret_cast<volatile LinkFrame*>(l1->slots)[slot];
    }
    // Adds the burst's egress stamps, which its frames carry, and its ingress stamps, which are in the RX stamp FIFO.
    // The FIFO doesn't say which frame a stamp belongs to, so the two ends take turns. The transmitter only sends a
    // burst once the previous one has been fully echoed, and the receiver only takes a burst's stamps once all its
    // frames have arrived. A frame's stamp is in the FIFO before the frame itself is visible, and only the link's rule
    // records stamps. The FIFO therefore holds exactly this burst's stamps, in order, unless a frame was resent and
    // stamped twice, in which case the burst is dropped.
    __attribute__((noinline)) void take_burst() {
        namespace rx_stamp_fifo = eth_ptp::rx_stamp_fifo;
        if (!rx_stamp_fifo::holds_exactly<kBurstFrames>()) {
            rx_stamp_fifo::flush();
            return;
        }
#pragma GCC unroll 1
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            egress.add(eth_ptp::frame_stamp_ns(frame(i).stamp));
            ingress.add(rx_stamp_fifo::pop());
        }
    }
    // Waits a uniformly random number of cycles, up to one refclk update period, then sends the slot's frame. The
    // stamps are whole ticks, so a delay that is uniform over a whole number of ticks makes each stamp's rounding error
    // independent of when the router reached the frame, and averaging a round's frames cancels it.
    __attribute__((noinline)) bool send_dithered(uint32_t slot) {
        period.remeasure_if_due();
        const uint32_t cycles = eth_clock::draw(random_state, period.cycles_per_update);
        dither_wait(cycles / 2);
        dither_wait(cycles - cycles / 2);
        // If the queue is busy, leave the frame for a later step. Waiting on a queue that a link-level resend keeps
        // busy would stop the router serving the fabric.
        if (internal_::eth_txq_is_busy(kLinkTxq)) {
            return false;
        }
        const uint32_t word_addr = reinterpret_cast<uintptr_t>(&frame(slot)) >> 4;
        internal_::eth_send_packet_unsafe(kLinkTxq, word_addr, word_addr, kFrameBytes >> 4);
        return true;
    }
    // Waits up to half the longest delay and is called twice, because a run of nops of the full length overflows the
    // active-eth kernel config buffer when the router also has eth zones.
    __attribute__((noinline)) static void dither_wait(uint32_t cycles) { eth_clock::nops<kMaxDitherCycles / 2>(cycles); }
    static FORCE_INLINE volatile uint32_t* control_vector() {
        return reinterpret_cast<volatile uint32_t*>(GET_MAILBOX_ADDRESS_DEV(profiler.control_vector));
    }
    // The host computes the average, because a 64-bit divide would pull a library routine into the ERISC's code.
    __attribute__((noinline)) void record(const StampSum& sum, kernel_profiler::SyncRole role) {
        auto& slot = const_cast<kernel_profiler::SyncLinkRecord&>(
            l1->ring[ring_tail % kernel_profiler::kLinkSyncRingRecords].link);
        slot.meta = kernel_profiler::SyncMeta{.role = role, .kind = kernel_profiler::SyncKind::Link};
        slot.round = round;
        slot.first_ns = sum.first;
        slot.sum_from_first_ns = sum.sum_from_first;
        slot.count = sum.count;
        std::atomic_thread_fence(std::memory_order_release);
        control_vector()[kernel_profiler::SPSC_LINK_SYNC_TAIL] = ++ring_tail;
    }
    // A round with no stamps is not recorded, and neither is one the ring has no room for, because an end never waits
    // for the eth relay.
    __attribute__((noinline)) void close_round(
        kernel_profiler::SyncRole egress_role, kernel_profiler::SyncRole ingress_role) {
        if (egress.count != 0 && ring_tail - control_vector()[kernel_profiler::SPSC_LINK_SYNC_HEAD] <=
                                     kernel_profiler::kLinkSyncRingRecords - 2) {
            record(egress, egress_role);
            record(ingress, ingress_role);
        }
    }
    FORCE_INLINE void open_round(
        uint32_t next, kernel_profiler::SyncRole egress_role, kernel_profiler::SyncRole ingress_role) {
        close_round(egress_role, ingress_role);
        round = next;
        egress.reset();
        ingress.reset();
    }
};

struct TransmitterLink : EndBase {
    static constexpr uint32_t kBurstTicks =
        (kSyncCheck ? kernel_profiler::kLinkSyncCheckPaceTicks : kernel_profiler::kLinkSyncPaceTicks) / kBurstsPerRound;
    // Only the low word is kept, since a burst is never scheduled more than kMaxLeadTicks ahead.
    static constexpr uint32_t kMaxLeadTicks = 1u << 20;
    static_assert(kBurstTicks < kMaxLeadTicks);
    uint32_t next_burst_refclk = 0;
    uint32_t frames_to_send = 0, round_bursts_sent = 0;

    void start(uint32_t link_l1) {
        EndBase::start(link_l1);
        resync();
    }
    // A burst that appears more than kMaxLeadTicks ahead is overdue. After a pause of any length this delays a burst
    // by at most kMaxLeadTicks, whereas a signed difference could hold it back 2^31 ticks.
    FORCE_INLINE bool due() const {
        return frames_to_send != 0 || next_burst_refclk - eth_ptp::kRefclkLo.read() > kMaxLeadTicks;
    }
    FORCE_INLINE void serve() {
        if (frames_to_send != 0) {
            send_next();
            return;
        }
        invalidate_l1_cache();
        if (l1->ctl != kernel_profiler::LinkSyncCtl::Run) {
            round_bursts_sent = 0;
            resync();
            return;
        }
        if (!echoed()) {
            return;
        }
        burst();
    }

private:
    FORCE_INLINE void resync() { next_burst_refclk = eth_ptp::kRefclkLo.read() + kBurstTicks; }
    FORCE_INLINE bool echoed() const {
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            if (frame(i).awaiting_echo) {
                return false;
            }
        }
        return true;
    }
    __attribute__((noinline)) void send_next() { frames_to_send -= send_dithered(kBurstFrames - frames_to_send); }
    __attribute__((noinline)) void burst() {
        const uint32_t now = eth_ptp::kRefclkLo.read();
        take_burst();
        if (round_bursts_sent == 0) {
            open_round(round + 1, kernel_profiler::SyncRole::ReturnEgress, kernel_profiler::SyncRole::ReturnIngress);
        }
        if (++round_bursts_sent == kBurstsPerRound) {
            round_bursts_sent = 0;
        }
#pragma GCC unroll 1
        for (uint32_t i = 0; i < kBurstFrames; i++) {
            volatile LinkFrame& sent = frame(i);
            sent.round = round;
            sent.awaiting_echo = 1;
        }
        frames_to_send = kBurstFrames;
        next_burst_refclk += kBurstTicks;
        if (static_cast<int32_t>(next_burst_refclk - now) < 0) {
            next_burst_refclk = now;
        }
    }
};

struct ReceiverLink : EndBase {
    uint32_t frames_taken = 0, frames_echoed = 0;

    FORCE_INLINE bool due() const {
        invalidate_l1_cache();
        return frames_echoed != frames_taken || frame(frames_taken).awaiting_echo != 0;
    }
    FORCE_INLINE void serve() {
        if (frames_echoed == frames_taken) {
            take();
        }
        echo();
    }

private:
    // Takes each frame as it arrives, and the burst's stamps with its last frame, before that frame is echoed. The
    // transmitter sends nothing more until every echo is in, so the earlier frames' slots still hold their stamps, and
    // an echo carries its stamp only on the wire.
    __attribute__((noinline)) void take() {
        volatile LinkFrame& taken = frame(frames_taken);
        if (frames_taken == 0 && taken.round != round) {
            open_round(
                taken.round, kernel_profiler::SyncRole::ForwardEgress, kernel_profiler::SyncRole::ForwardIngress);
        }
        if (frames_taken == kBurstFrames - 1) {
            take_burst();
        }
        taken.awaiting_echo = 0;
        frames_taken++;
    }
    __attribute__((noinline)) void echo() {
        frames_echoed += send_dithered(frames_echoed);
        if (frames_echoed == kBurstFrames) {
            frames_taken = 0;
            frames_echoed = 0;
        }
    }
};

template <bool Transmitter>
using LinkEnd = std::conditional_t<Transmitter, TransmitterLink, ReceiverLink>;

// due() and serve() stay out of line because inlining the receiver's check makes the router's main loop spill
// registers.
template <bool Transmitter>
struct RouterEnd {
    static inline LinkEnd<Transmitter> end;
    __attribute__((noipa, cold)) static void start(uint32_t l1) { end.start(l1); }
    __attribute__((noinline)) static bool due() { return end.due(); }
    __attribute__((noinline)) static void serve() { end.serve(); }
    // Stepping every 16th loop keeps the link's cost to the router small. The sync check runs rounds 10 times as often
    // and needs every loop: at 16, its bursts outlast the time each one is given, and too few rounds are left over for
    // the check.
    static constexpr uint32_t kRouterStepLoops = kSyncCheck ? 1 : 16;
    static_assert((kRouterStepLoops & (kRouterStepLoops - 1)) == 0);
    static FORCE_INLINE void step(uint32_t iter) {
        if ((iter & (kRouterStepLoops - 1)) == 0 && due()) {
            serve();
        }
    }
    __attribute__((noipa, cold)) static void stop() { end.stop(); }
};

struct NoLinkEnd {
    static void start(uint32_t) {}
    static void step(uint32_t) {}
    static void stop() {}
};

template <kernel_profiler::LinkSyncRole Role>
using RouterHook = std::conditional_t<
    Role == kernel_profiler::LinkSyncRole::None,
    NoLinkEnd,
    RouterEnd<Role == kernel_profiler::LinkSyncRole::Transmitter>>;

}  // namespace link_sync
