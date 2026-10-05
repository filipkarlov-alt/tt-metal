// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/streaming_profiler_common.h"
#include "internal/ethernet/eth_ptp.hpp"
#include "tt_metal/impl/streaming_profiler/kernels/eth_clock.hpp"

namespace eth_clock {
// The stream's block length in cycles. Pads are drawn uniformly over this range so each pass starts at a uniform phase
// relative to the blocks.
constexpr uint32_t kPadRange = 6;
constexpr uint32_t kPadSlots = 256;
// The arguments to sampler_stream. The registers are passed in as data so the compiler keeps their addresses in
// registers instead of rebuilding them between blocks.
struct StreamArgs {
    eth_ptp::Reg<> refclk;
    eth_ptp::Reg<> wall;
    Calibration calibration;
    volatile uint32_t* slots;
    uint32_t mask;
    volatile uint32_t* head;
    uint32_t limit;
    bool stop_on_reject;
    const uint32_t* pads;
    volatile uint32_t* tail_word;
    uint32_t tail;

    // Sets up one calibration pass, which stores the first update it catches into `slot` and returns. *head must be
    // kSyncHeadStop so the reload after that update ends the call, and tail_word must point at a word the model never
    // reads. The fields are copied one by one because copying the structs goes through the stack at -Os.
    FORCE_INLINE void prepare_calibration_pass(
        uint32_t period, GapPositions gap_positions, volatile tt_l1_ptr uint32_t* slot) {
        calibration.period = period;
        calibration.gap_positions.word = gap_positions.word;
        slots = slot;
        mask = 0;
        stop_on_reject = true;
        limit = tail + 1;
    }
    // Sets up sampling, which streams every update into `ring` until the model stops it, with no limit on rejected
    // passes.
    FORCE_INLINE void prepare_sampling(
        const Calibration& calibrated, volatile tt_l1_ptr kernel_profiler::SyncSampleRing* ring) {
        calibration.period = calibrated.period;
        calibration.gap_positions.word = calibrated.gap_positions.word;
        slots = &ring->samples[0].refclk;
        mask = kernel_profiler::kSyncSampleRingSamples - 1;
        stop_on_reject = false;
        tail_word = &ring->tail;
        tail = 0;
        limit = mask;
    }
};

// Samples refclk changes into args.slots, sample n in slot n & args.mask, and returns the sample number after the last
// one stored. Each pass first waits a random number of nops, so passes start at random phases of the refclk, then reads
// blocks of refclk, refclk, wall clock, refclk until the refclk changes. A change is kept only if its block and the
// next took exactly the calibrated period, so a stalled read can't skew it. Calibration (period 0) stores the block
// lengths instead.
//
// The optimize attributes keep GCC from merging the handlers' tails and inlining the calibration path. The extra taken
// branches would slow the stream enough to miss refclk updates at low AICLK.
__attribute__((noinline, aligned(64), optimize("no-crossjumping", "reorder-blocks-algorithm=stc"))) inline uint32_t
sampler_stream(const StreamArgs& args) {
    enum class Step : uint8_t { Stored, AtLimit, Rejected };
    const eth_ptp::Reg<> refclk = args.refclk, wall = args.wall;
    const uint32_t period = args.calibration.period, mask = args.mask;
    const GapPositions gap_positions{args.calibration.gap_positions.word};
    volatile uint32_t* const slots = args.slots;
    volatile uint32_t* const head = args.head;
    uint32_t limit = args.limit;
    const bool stop_on_reject = args.stop_on_reject;
    const uint32_t* const pads = args.pads;
    volatile uint32_t* const tail_word = args.tail_word;
    uint32_t tail = args.tail;
    const auto read = [&]() __attribute__((always_inline)) {
        Block block;
        block.first_refclk = refclk.read();
        block.second_refclk = refclk.read();
        block.wall = wall.read();
        block.last_refclk = refclk.read();
        return block;
    };
    // A pass jumps into a run of nops, pad bytes before its end, so each pad costs exactly its length in cycles. A
    // switch compiles to a jump table whose cases aren't one cycle apart. The run's end label is defined in asm, so
    // this function is emitted once.
    uint32_t nops_end;
    asm("lla %0, .Lsampler_stream_nops_end" : "=r"(nops_end));
    uint32_t pad = pads[tail & (kPadSlots - 1)];
    disable_branch_predictor();
    // The pass starts on a cache line so its blocks have the same layout in every build, since layout alone moved the
    // sampled times by about a tick. Passing pad through the asm keeps its load before the alignment.
    asm volatile(".p2align 6" : "+r"(pad));
    // Marked very unlikely, so the reads stay straight-line code and a change costs one taken branch into its handler.
    const auto stepped = [](const Block& before, const Block& last) __attribute__((always_inline)) {
        return __builtin_expect_with_probability(last.last_refclk != before.last_refclk, 0, 0.001);
    };
    // Stores a change that happened between the final refclk reads of blocks `before` and `last`. The change is timed
    // with `last`'s reads, and `now`, the block after `last`, measures how long that block took.
    const auto store = [&](const Block& before, const Block& last, const Block& now)
                           __attribute__((always_inline)) -> Step {
        const uint32_t step_block_ticks = last.wall - before.wall, next_block_ticks = now.wall - last.wall;
        volatile uint32_t* const slot = slots + 2 * (tail & mask);
        if (__builtin_expect(period != 0, 1)) {
            if (((step_block_ticks ^ period) | (next_block_ticks ^ period)) != 0) {
                return Step::Rejected;
            }
            // The change fell in gap 0 if the block's first read already differs from the previous block's last, in
            // gap 2 if only the block's last read differs, and in gap 1 otherwise.
            const uint32_t gap = static_cast<uint32_t>(last.first_refclk == before.last_refclk)
                                 << (last.second_refclk == last.first_refclk);
            slot[0] = last.last_refclk;
            slot[1] = (last.wall << 3) + static_cast<uint32_t>(gap_positions.at(gap));
            *tail_word = tail;
            return ++tail == limit ? Step::AtLimit : Step::Stored;
        }
        slot[0] = next_block_ticks;
        return ++tail == limit ? Step::AtLimit : Step::Stored;
    };
    // Reads blocks until the refclk steps, then stores the step. Rotating three named blocks keeps each check's reads
    // in the registers they were loaded into. A single rotating variable or an array makes the compiler copy them into
    // one handler's registers.
    const auto run = [&]() __attribute__((always_inline)) -> Step {
        Block a = read(), b = read(), c;
#pragma GCC unroll 6
        for (uint32_t i = 0; i < 6; i++) {
            c = read();
            if (stepped(a, b)) {
                return store(a, b, c);
            }
            a = read();
            if (stepped(b, c)) {
                return store(b, c, a);
            }
            b = read();
            if (stepped(c, a)) {
                return store(c, a, b);
            }
        }
        return Step::Rejected;
    };
pass:
    // norvc keeps every nop 4 bytes, so a pad of n bytes runs n / 4 nops.
    asm volatile(
        ".option push\n\t.option norvc\n\t"
        "sub t0, %[end], %[pad]\n\t"
        "jr t0\n\t"
        ".rept %[n]\n\tnop\n\t.endr\n"
        ".Lsampler_stream_nops_end:\n\t"
        ".option pop"
        :
        : [end] "r"(nops_end), [pad] "r"(pad), [n] "i"(kPadRange - 1)
        : "t0", "memory");
    pad = pads[tail & (kPadSlots - 1)];
    switch (run()) {
        case Step::Stored: goto pass;
        case Step::AtLimit: goto at_limit;
        case Step::Rejected: goto reject;
    }
reject:
    if (!stop_on_reject) {
        goto pass;
    }
    goto done;
at_limit:
    for (;;) {
        invalidate_l1_cache();
        const uint32_t model_head = *head;
        if (__builtin_expect(static_cast<int32_t>(tail - model_head) < 0, 0)) {
            goto done;
        }
        limit = model_head + mask;
        if (limit != tail) {
            break;
        }
    }
    pad = pads[tail & (kPadSlots - 1)];
    goto pass;
done:
    enable_branch_predictor();
    return tail;
}
}  // namespace eth_clock

constexpr uint32_t kCtrlAddr = get_named_compile_time_arg_val("ctrl");
constexpr uint32_t kSampleRingAddr = get_named_compile_time_arg_val("sample_ring");

void kernel_main() {
    volatile tt_l1_ptr kernel_profiler::ResidentCtrl* ctrl =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::ResidentCtrl*>(kCtrlAddr);
    volatile tt_l1_ptr kernel_profiler::SyncSampleRing* ring =
        reinterpret_cast<volatile tt_l1_ptr kernel_profiler::SyncSampleRing*>(kSampleRingAddr);
    // The stream's pads in bytes (4 per nop), uniform over kPadRange nops. They are drawn once up front so each pass's
    // pad is a single load.
    static uint32_t pads[eth_clock::kPadSlots];
    uint32_t random_state = eth_ptp::kWallClockLo.read() | 1u;
    for (uint32_t& pad : pads) {
        pad = 4u * eth_clock::draw(random_state, eth_clock::kPadRange);
    }
    uint32_t unpublished = 0;
    eth_clock::StreamArgs args{
        .refclk = eth_ptp::kRefclkLo,
        .wall = eth_ptp::kWallClockLo,
        .head = &ring->head,
        .pads = pads,
        .tail_word = &unpublished};
    ring->head = kernel_profiler::kSyncHeadStop;
    const eth_clock::Calibration calibration = eth_clock::calibrate(
        [&](uint32_t period, eth_clock::GapPositions gap_positions, volatile tt_l1_ptr uint32_t* slot) {
            const uint32_t from = args.tail;
            args.prepare_calibration_pass(period, gap_positions, slot);
            args.tail = sampler_stream(args);
            return args.tail - from;
        },
        ctrl);
    if (ctrl->stop == 0u) {
        ring->wall_eighths = eth_ptp::read_instant().wall << 3;
        ring->head = 0;
        args.prepare_sampling(calibration, ring);
        const uint32_t tail = sampler_stream(args);
        std::atomic_thread_fence(std::memory_order_release);
        ring->tail = tail;
    }
    std::atomic_thread_fence(std::memory_order_release);
    ring->done = 1;
}
