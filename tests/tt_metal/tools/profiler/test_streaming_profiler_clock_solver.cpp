// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Host-only tests of the clock solver and clock map against a synthetic truth model, with no device. Chain feeds three
// chips joined in a chain by two links, in 10 ms batches, with chips 1 and 2 on crystals tens of ppm off chip 0's, chip
// 0 switching AICLK partway, chip 1 sending no points for 350 ms, link streams with dropped and late stamps, and every
// counter a year past power-on. It checks placements on the host timeline before, across and after each of those to
// within a TSC tick, and steady_clock times to within 1 ns. Retention overflows a series' capacity and checks where a
// record lands on the newest node, between two kept nodes, and before the oldest kept one.

#include <cmath>
#include <cstdint>
#include <optional>

#include <gtest/gtest.h>

#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/sync/clock_solver.hpp"

using namespace tt::tt_metal;
using namespace tt::tt_metal::streaming_profiler;

namespace {

using kernel_profiler::SyncRole;

constexpr int64_t kRefclkHz = kernel_profiler::kEthRefclkHz;
constexpr int64_t kNsPerSecond = 1'000'000'000;
constexpr int64_t kTicksPerNs = 3;
// The test's readings are exact, so a placement is off only by its rounding to a TSC tick (0.33 ns).
constexpr double kToleranceNs = 1.0 / kTicksPerNs;
// A steady_clock time also rounds to a whole ns.
constexpr double kSteadyToleranceNs = 1.0;
// Wall ticks per refclk tick, in eighths, before and after chip 0's DVFS switch, which is one 1/8 step of the PLL
// multiple.
constexpr uint32_t kRateFast = 216, kRateSlow = 215;
constexpr int64_t kAiclkHz = kRefclkHz * kRateFast / 8;
constexpr double kSlowAiclkRatio = static_cast<double>(kRateSlow) / kRateFast;
constexpr double kSwitchS = 0.300;
constexpr double kOneWayS = 1.0e-6;
constexpr double kTurnaroundS = 350e-9;
// Every clock reads as it would a year after power-on, so the AICLK, TSC and host counts are past 2^53 of their units.
constexpr int64_t kYear = int64_t{365} * 86400;
constexpr int64_t kRefclkTicksPerYear = kYear * kRefclkHz, kWallTicksPerYear = kYear * kAiclkHz;
constexpr int64_t kRefclk0[3] = {kRefclkTicksPerYear, kRefclkTicksPerYear + 1'000'000, kRefclkTicksPerYear + 3'000'000};
constexpr int64_t kWall0[3] = {
    kWallTicksPerYear + 1'000'000'000, kWallTicksPerYear + 7'000'000'000, kWallTicksPerYear + 4'000'000'000};
constexpr int64_t kTsc0 = kYear * kNsPerSecond * kTicksPerNs;
constexpr int64_t kHost0 = kYear * kNsPerSecond;
// Each chip's refclk and AICLK come from its own crystal, so a chip's rate offset scales both. A solver that drops a
// link's slope misplaces chips 1 and 2 by microseconds over the second.
constexpr double kRate[3] = {1.0, 1.0 + 40e-6, 1.0 - 25e-6};

double refclk(int chip, double tau) { return kRefclkHz * kRate[chip] * tau; }
double wall(int chip, double tau) {
    if (chip != 0 || tau <= kSwitchS) {
        return kRate[chip] * kAiclkHz * tau;
    }
    return kAiclkHz * kSwitchS + kSlowAiclkRatio * kAiclkHz * (tau - kSwitchS);
}
double sent_at(uint32_t round) { return 0.020 + round * 1e-3; }
double tsc(double tau) { return tau * 1e9 * kTicksPerNs; }
double host_ns(double tau) { return tau * 1e9; }
int64_t wall_tick(int chip, double tau) { return kWall0[chip] + std::llround(wall(chip, tau)); }
// Returns the tau at which the chip's wall clock reads `tick`.
double tau_at_wall(int chip, int64_t tick) {
    const auto wall_ticks = static_cast<double>(tick - kWall0[chip]);
    if (chip != 0 || wall_ticks <= wall(0, kSwitchS)) {
        return wall_ticks / (kRate[chip] * kAiclkHz);
    }
    return kSwitchS + (wall_ticks - wall(0, kSwitchS)) / (kSlowAiclkRatio * kAiclkHz);
}
uint64_t hw_stamp(int chip, double tau) {
    return static_cast<uint64_t>(kRefclk0[chip] * kNsPerRefclk + std::llround(refclk(chip, tau) * kNsPerRefclk));
}

void feed_wall_clock_point(
    ClockSolver& solver, uint32_t dev, uint64_t refclk_tick, uint64_t wall_eighths, uint32_t wall_per_refclk_eighths) {
    solver.on_record(
        dev,
        0,
        {.wall_clock = {
             .meta = {.count = 1, .kind = kernel_profiler::SyncKind::WallClock},
             .wall_per_refclk_eighths = {static_cast<uint8_t>(wall_per_refclk_eighths)},
             .first_wall_eighths_hi = static_cast<uint32_t>(wall_eighths >> 32),
             .points = {
                 {.refclk_lo = static_cast<uint32_t>(refclk_tick),
                  .wall_eighths_lo = static_cast<uint32_t>(wall_eighths)}}}});
}
// The receiver keeps the forward frame's stamps and the transmitter, end a, the return frame's.
void feed_link(ClockSolver& solver, const CaptureContext::Link& link, uint32_t round, SyncRole role, uint64_t stamp) {
    const bool at_receiver = role == SyncRole::ForwardEgress || role == SyncRole::ForwardIngress;
    solver.on_record(
        at_receiver ? link.dev_b : link.dev_a,
        at_receiver ? link.core_b : link.core_a,
        {.link = {
             .meta = {.role = role, .kind = kernel_profiler::SyncKind::Link},
             .round = round,
             .first_ns = stamp,
             .count = 1}});
}

}  // namespace

TEST(StreamingProfilerClockSolver, Chain) {
    CaptureContext ctx;
    for (uint32_t chip = 0; chip < 3; chip++) {
        ctx.devices.push_back({.chip_id = chip});
    }
    ctx.links.push_back(CaptureContext::Link{.dev_a = 0, .dev_b = 1, .core_a = 0, .core_b = 0});
    ctx.links.push_back(CaptureContext::Link{.dev_a = 1, .dev_b = 2, .core_a = 1, .core_b = 0});
    ClockSolver solver(ctx, {.root_refclk = kRefclk0[0], .tsc = kTsc0});
    ClockMap& map = solver.map();
    for (double tau : {0.0, 0.6}) {
        map.append_host(
            HostNode{.at = refclk(0, tau), .value = tsc(tau), .tangent = kTicksPerNs * 1e9 / kRefclkHz},
            refclk(0, tau + 0.6));
    }
    for (double tau : {0.0, 1.0}) {
        service().steady().append(kTsc0 + std::llround(tsc(tau)), kHost0 + std::llround(host_ns(tau)));
    }

    // A point is taken as the refclk reaches a whole tick, at or after tau.
    const auto point = [&](int chip, double tau, uint32_t wall_per_refclk_eighths) {
        const double ticks = std::ceil(refclk(chip, tau));
        const double at = ticks / (kRefclkHz * kRate[chip]);
        feed_wall_clock_point(
            solver,
            static_cast<uint32_t>(chip),
            static_cast<uint64_t>(kRefclk0[chip] + static_cast<int64_t>(ticks)),
            static_cast<uint64_t>(8 * kWall0[chip] + std::llround(8.0 * wall(chip, at))),
            wall_per_refclk_eighths);
    };
    // Each link runs 970 rounds 1 ms apart across the second the chips' points cover, several of the solve's 250 ms
    // windows, with its stream damaged the way a lapped consumer or a full ring damages it.
    constexpr uint32_t kLinkRounds = 970, kLateRound = 100, kLateArrival = 105;
    const auto link_round = [&](const CaptureContext::Link& link, uint32_t round) {
        const auto feed_forward = [&](uint32_t forward_round) {
            const double sent = sent_at(forward_round);
            feed_link(solver, link, forward_round, SyncRole::ForwardEgress, hw_stamp(link.dev_a, sent));
            feed_link(solver, link, forward_round, SyncRole::ForwardIngress, hw_stamp(link.dev_b, sent + kOneWayS));
        };
        const double sent = sent_at(round);
        const double return_egress = sent + kOneWayS + kTurnaroundS,
                     return_ingress = sent + 2 * kOneWayS + kTurnaroundS;
        feed_link(solver, link, round, SyncRole::ReturnEgress, hw_stamp(link.dev_b, return_egress));
        if (round % 11 != 5) {
            feed_link(solver, link, round, SyncRole::ReturnIngress, hw_stamp(link.dev_a, return_ingress));
        }
        if (round % 7 != 3 && round != kLateRound) {
            feed_forward(round);
        }
        if (round == kLateArrival) {
            feed_forward(kLateRound);
        }
    };
    // The records arrive in time order, and the solver publishes after every 10 ms of them, as the solver thread does
    // after each batch.
    bool switched = false;
    for (uint32_t k = 0; k < 1000; k++) {
        const double tau = k * 1e-3;
        if (!switched && tau > kSwitchS) {
            point(0, kSwitchS + 50e-6, kRateSlow);
            point(0, kSwitchS + 150e-6, kRateSlow);
            switched = true;
        }
        for (int c = 0; c < 3; c++) {
            // Chip 1 sends no points from 0.40 to 0.75 s, so its records there land on the chord across the gap.
            if (c == 1 && tau > 0.40 && tau < 0.75) {
                continue;
            }
            point(c, tau, c == 0 && tau > kSwitchS ? kRateSlow : kRateFast);
        }
        if (k < kLinkRounds) {
            for (const CaptureContext::Link& link : ctx.links) {
                link_round(link, k);
            }
        }
        if (k % 10 == 9) {
            solver.on_batch_end();
        }
    }
    solver.on_capture_end();

    ClockMap::Reader reader = map.reader();
    // A record carries a whole wall tick, so its true time is when the chip's wall clock reads that tick.
    const auto true_tau = [](int chip, double tau) { return tau_at_wall(chip, wall_tick(chip, tau)); };
    const auto placed_tsc = [&](int chip, double tau) {
        return map.place_host(reader, static_cast<uint32_t>(chip), wall_tick(chip, tau));
    };
    const auto placed_error_ns = [&](int chip, double tau) {
        return (static_cast<double>(placed_tsc(chip, tau) - kTsc0) - tsc(true_tau(chip, tau))) / kTicksPerNs;
    };
    const auto steady_error_ns = [&](int chip, double tau) {
        return static_cast<double>(*service().steady().ns(placed_tsc(chip, tau)) - kHost0) -
               host_ns(true_tau(chip, tau));
    };
    for (double tau : {0.050, 0.150, 0.280}) {
        EXPECT_NEAR(placed_error_ns(0, tau), 0.0, kToleranceNs) << "chip0 root pre-switch, tau " << tau;
    }
    EXPECT_NEAR(placed_error_ns(0, kSwitchS + 100e-6), 0.0, kToleranceNs) << "chip0 root across the switch";
    for (double tau : {0.400, 0.700, 0.950}) {
        EXPECT_NEAR(placed_error_ns(0, tau), 0.0, kToleranceNs) << "chip0 root post-switch, tau " << tau;
    }
    for (double tau : {0.050, 0.500, 0.950}) {
        EXPECT_NEAR(placed_error_ns(1, tau), 0.0, kToleranceNs) << "chip1 one hop, tau " << tau;
        EXPECT_NEAR(placed_error_ns(2, tau), 0.0, kToleranceNs) << "chip2 two hops, tau " << tau;
    }
    for (int chip : {0, 1, 2}) {
        for (double tau : {0.050, 0.500, 0.950}) {
            EXPECT_NEAR(steady_error_ns(chip, tau), 0.0, kSteadyToleranceNs)
                << "steady chip" << chip << ", tau " << tau;
        }
    }
}

TEST(StreamingProfilerClockMap, Retention) {
    constexpr uint32_t kNodes = 1u << 14;
    ClockMap map(1, kNodes, {});
    ClockMap::Reader reader = map.reader();
    constexpr int64_t kStep = 1024;
    // The first segment is steeper than the rest, which alternate in slope, and no node's tangent matches a chord, so
    // the retired first node, a kept node and a mid-series chord each place differently. Every value is a multiple of
    // 2^-5 below 2^43, where a double resolves 2^-10, and the step is a power of two, so the map's arithmetic on them
    // is exact.
    const auto value = [](uint32_t node) {
        return node == 0 ? 7.5e12 - 62.5 : 7.5e12 + 31.25 * node + 3.90625 * (node % 2);
    };
    const auto tangent = [](uint32_t node) { return 0.041015625 + 0.001953125 * (node % 3); };
    constexpr uint32_t node_count = kNodes + 1;
    for (uint32_t i = 0; i < node_count; i++) {
        map.append(0, SyncNode{.at = static_cast<int64_t>(i) * kStep, .value = value(i), .tangent = tangent(i)});
    }
    const auto root = [&](int64_t tick) { return map.place_root(reader, 0, tick, 0.0); };
    const uint32_t mid = node_count / 2;
    const std::optional<double> newest = root(static_cast<int64_t>(node_count - 1) * kStep);
    const std::optional<double> mid_series = root(static_cast<int64_t>(mid) * kStep + kStep / 2);
    const std::optional<double> retired = root(0);
    ASSERT_TRUE(newest && mid_series && retired);
    EXPECT_EQ(*newest, value(node_count - 1)) << "the newest node";
    EXPECT_EQ(*mid_series, (value(mid) + value(mid + 1)) / 2) << "a node mid-series";
    EXPECT_EQ(*retired, value(1) - tangent(1) * kStep) << "the retired first node, on the oldest kept tangent";
}
