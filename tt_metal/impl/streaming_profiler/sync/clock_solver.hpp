// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "hostdev/streaming_profiler_common.h"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/sync/least_squares.hpp"
#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

static_assert(1'000'000'000 % kernel_profiler::kEthRefclkHz == 0);
inline constexpr int64_t kNsPerRefclk = 1'000'000'000 / kernel_profiler::kEthRefclkHz;
inline constexpr double kRefclkTicksPerMs = kernel_profiler::kEthRefclkHz / 1e3;

// The wall-clock core reports wall clocks, and wall ticks per refclk tick, in eighths of a wall tick.
inline constexpr int kWallEighthBits = 3;
inline constexpr int64_t kWallEighths = int64_t{1} << kWallEighthBits;
constexpr int64_t whole_ticks(int64_t eighths) { return eighths >> kWallEighthBits; }
constexpr int64_t nearest_tick(int64_t eighths) { return (eighths + kWallEighths / 2) >> kWallEighthBits; }
constexpr double tick_fraction(int64_t eighths) {
    return static_cast<double>(eighths & (kWallEighths - 1)) / kWallEighths;
}
constexpr double eighths_as_ticks(int64_t eighths) { return static_cast<double>(eighths) / kWallEighths; }

// One link round, in the transmitter's refclk ticks: the midpoint of its stamps, and the receiver's offset from the
// transmitter at that midpoint.
struct RoundPoint {
    double mid = 0.0, offset = 0.0;
};

// Maps a chip's refclk onto the root chip's refclk.
struct RootTransform {
    double scale = 1.0, shift = 0.0;
    double operator()(double refclk) const { return scale * refclk + shift; }
};

// A point of a Tracy plot, placed at a root refclk tick.
struct PlotPoint {
    double root = 0.0, value = 0.0;
};

// A histogram of weighted errors in ns, in bins 1 / BinsPerNs ns wide over [-RangeNs, RangeNs), that also tracks the
// worst error's magnitude.
template <int BinsPerNs, int RangeNs>
struct ErrorHistogram {
    static constexpr int kCentre = RangeNs * BinsPerNs;
    double weight_sum = 0.0, worst = 0.0;
    std::array<double, 2 * kCentre> bins{};
    void add(double ns, double weight) {
        const int bin = static_cast<int>(std::floor(ns * BinsPerNs)) + kCentre;
        if (bin >= 0 && bin < 2 * kCentre) {
            bins[bin] += weight;
        }
        weight_sum += weight;
        worst = std::max(worst, std::abs(ns));
    }
    // Returns the centre of the bin that holds the |error| quantile, or the worst error if the quantile is out of
    // range. Requires weight_sum > 0.
    double abs_quantile(double quantile) const {
        double covered = 0.0;
        for (int i = 0; i < kCentre; i++) {
            covered += bins[kCentre + i] + bins[kCentre - 1 - i];
            if (covered >= quantile * weight_sum) {
                return (i + 0.5) / BinsPerNs;
            }
        }
        return worst;
    }
};

// Solves each chip's refclk-to-root transform from the links' lines by least squares. A link with no line passes null,
// and a chip that no line reaches gets no transform.
std::vector<std::optional<RootTransform>> compose_on_root(
    const CaptureContext& ctx, std::span<const LineFit* const> lines);

class SyncCheck;

// One host read of the root chip's refclk: the host TSC at the read's midpoint, its round trip, and the refclk count.
struct RefclkRead {
    int64_t mid = 0, rtt = 0;
    uint64_t refclk = 0;
};
inline constexpr uint32_t kRefclkBurstReads = 1000;
using RefclkBurst = std::array<RefclkRead, kRefclkBurstReads>;
// On an 8-chip LoudBox, a line fitted over 1 s of bursts and extrapolated 10 ms ahead misses the next burst's line by
// 0.07 ns rms (0.6 ns max).
inline constexpr auto kRefclkBurstPeriod = std::chrono::milliseconds(10);

// Builds a capture's ClockMap from its clock readings: maps each chip's wall clock onto the root chip's refclk, and the
// root chip's refclk onto the host TSC. The links are solved refclk against refclk, so DVFS on either chip can't affect
// them.
class ClockSolver {
public:
    // Starts a capture on `ctx` whose map starts at `bases`. Every device in `ctx` must have a path over its links to
    // the root (device index 0).
    ClockSolver(const CaptureContext& ctx, const ClockBases& bases);
    ~ClockSolver();
    ClockSolver(const ClockSolver&) = delete;
    ClockSolver& operator=(const ClockSolver&) = delete;

    ClockMap& map() { return map_; }

    void on_record(uint32_t dev, uint32_t core, const kernel_profiler::SyncRecord& record);
    // Publishes what the batch's records added to the map, and returns whether anything was published.
    bool on_batch_end();
    // Fits a burst of host reads of the root refclk into the host series. Returns whether the series has a line yet.
    bool on_refclk_burst(const RefclkBurst& burst);
    void on_capture_end();

private:
    struct Instant {
        int64_t refclk = 0;
        int64_t wall_eighths = 0;
        uint32_t wall_per_refclk_eighths = 0;
        double wall() const { return eighths_as_ticks(wall_eighths); }
        int64_t wall_tick() const { return nearest_tick(wall_eighths); }
        double wall_per_refclk() const { return eighths_as_ticks(wall_per_refclk_eighths); }
    };
    // Values are offsets from the chip's bases, so no double holds a count since power-on, which would lose precision.
    struct Chip {
        std::optional<int64_t> refclk_base, wall_base, last_refclk;
        std::deque<Instant> instants;
        size_t published = 0;
        const char* aiclk_plot = nullptr;
        // AICLK points not yet in the mean plot, the chip's AICLK as of the last point that is, and the root time of
        // the last point published.
        std::deque<PlotPoint> aiclk_pending;
        std::optional<double> aiclk;
        double aiclk_through = -std::numeric_limits<double>::infinity();
    };
    struct Round {
        std::array<std::optional<double>, static_cast<size_t>(kernel_profiler::SyncRole::ReturnIngress) + 1> stamps;
        std::optional<double>& operator[](kernel_profiler::SyncRole role) { return stamps[static_cast<size_t>(role)]; }
        bool complete() const {
            return std::ranges::all_of(stamps, [](const auto& stamp) { return stamp.has_value(); });
        }
    };
    struct LinkSolver {
        std::map<uint32_t, Round> pending;
        std::vector<RoundPoint> rounds;
        std::optional<LineFit> line;
        double solved_at = -std::numeric_limits<double>::infinity();
        // Each full solve window's scatter of rounds about the line, and the worst window's line error at its centre,
        // in ns.
        ErrorHistogram<64, 16> scatter_ns;
        double worst_centre_ns = 0.0;
    };
    struct CoreRef {
        uint32_t dev = 0, core = 0;
        auto operator<=>(const CoreRef&) const = default;
    };
    // A solve waits for a full window of rounds, except at capture end, where it uses whatever rounds a link has.
    enum class Window { Full, Partial };

    struct BurstPoint {
        double tsc, refclk;
    };

    void on_stamp(uint32_t dev, uint32_t core, const kernel_profiler::SyncLinkRecord& record);
    void solve(LinkSolver& solver, Window window);
    // Logs the worst link's median round scatter over the capture, and the worst single window's fit error.
    void report_precision() const;
    // Logs how much parallel links between the same two chips disagree about the chips' refclk offset.
    void report_parallel_links() const;
    const std::vector<std::optional<RootTransform>>& transforms();
    bool publish_dev(uint32_t dev);
    void plot(const char* name, std::span<const PlotPoint> series);
    // Plots the mean over chips of each chip's AICLK at every change through `until` on the root timeline.
    void plot_aiclk_mean(double until);

    const ClockBases bases_;
    ClockMap map_;
    ClockMap::Reader reader_;
    const CaptureContext& ctx_;
    std::vector<Chip> chips_;
    std::vector<LinkSolver> links_;
    std::map<CoreRef, size_t> link_of_;
    std::vector<std::optional<RootTransform>> to_root_;
    bool links_moved_ = true;
    std::vector<PlotPoint> aiclk_;
    const char* aiclk_mean_plot_ = nullptr;
    double aiclk_sum_ = 0.0;
    size_t aiclk_known_ = 0;
    std::unique_ptr<SyncCheck> check_;
    std::deque<BurstPoint> burst_points_;
    std::optional<HostNode> host_node_;
};

}  // namespace tt::tt_metal::streaming_profiler
