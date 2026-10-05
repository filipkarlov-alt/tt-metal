// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "impl/streaming_profiler/sync/clock_solver.hpp"

namespace tt::tt_metal::streaming_profiler {

// Measures the synced timeline against an independent reference. The check core's refclk readings on each chip, placed
// through the clock map, are compared with the held-out link rounds solved onto the root. A link's fixed asymmetry
// affects the reference and the sync equally, so it cancels. That asymmetry and the refclk read path shared with the
// wall-clock core are the only errors the check can't see.
class SyncCheck {
public:
    struct Reading {
        double refclk;
        int64_t wall_eighths;
        float weight = 1;
    };
    struct Round {
        size_t link;
        RoundPoint point;
    };
    struct ChipBatch {
        std::vector<Reading> readings;
        std::optional<double> root_minus_refclk;
        std::vector<PlotPoint> aiclk;  // AICLK in GHz
    };
    struct Batch {
        explicit Batch(size_t devices) : chips(devices) {}
        std::vector<Round> rounds;
        std::vector<ChipBatch> chips;
    };
    struct PlotSeries {
        std::string name;
        std::vector<PlotPoint> points;
    };

    SyncCheck(const CaptureContext& ctx, const ClockMap& map);
    SyncCheck(const SyncCheck&) = delete;
    SyncCheck& operator=(const SyncCheck&) = delete;

    // The batch the caller fills until submit().
    Batch& batch() { return staged_; }
    // Hands the batch to the check's thread and leaves it empty.
    void submit();
    // Stops the check's thread and logs the report.
    void finish();
    // Returns the worst error per millisecond, pooled and per chip.
    std::vector<PlotSeries> plots() const;

private:
    // Accumulates weighted errors in ns, and tracks the worst error's magnitude and root refclk tick.
    struct ErrorStats {
        double weight_sum = 0.0, sum = 0.0, worst = 0.0;
        double worst_root = 0.0;
        void add(double ns, double root, double weight);
        // Requires weight_sum > 0.
        double mean() const { return sum / weight_sum; }
    };
    // AICLK statistics, with each value weighted by how long it held.
    struct ClockStats {
        static constexpr double kBinMhz = 50.0;
        std::optional<PlotPoint> last;
        double span = 0.0, sum = 0.0, sum_squares = 0.0;
        double lo = std::numeric_limits<double>::infinity(), hi = -std::numeric_limits<double>::infinity();
        uint64_t changes = 0;
        std::map<int, double> by_bin;
        void add(double root, double mhz);
    };
    // A step's line needs its whole span of rounds, except once the capture is finishing.
    enum class Steps { Complete, All };
    struct LinkReference {
        std::deque<RoundPoint> rounds;
        std::map<int64_t, LineFit> lines_by_step;
        std::optional<int64_t> next_step;
        void fit_steps(Steps steps);
    };
    struct Sample {
        double root;
        float error_ns;
        float weight;
    };
    struct WorstByMs {
        int64_t first_ms = 0;
        std::deque<std::optional<float>> worst_ns;
        void add(double root, double error_ns);
    };
    struct Chip {
        std::deque<Reading> waiting;
        std::optional<double> last_refclk;
        std::deque<Sample> placed;
        uint64_t popped = 0;
        uint64_t no_reference = 0, no_node = 0;
        std::optional<double> root_minus_refclk;
        ErrorStats error;
        WorstByMs worst;
        ClockStats clock;
        const Sample& at(uint64_t index) const { return placed[index - popped]; }
        uint64_t end_index() const { return popped + placed.size(); }
    };
    // next_a is chip_a's next sample to pair, and last_b is where the search for chip_b's nearest sample starts.
    struct Pair {
        uint32_t chip_a = 0, chip_b = 0;
        uint64_t next_a = 0, last_b = 0;
        ErrorStats error;
    };
    enum class ReferenceState { Ready, Wait, None };
    // Each chip's refclk-to-root transform at one step.
    using StepTransforms = std::vector<std::optional<RootTransform>>;

    int64_t transmitter_step(size_t link_index, double root) const;
    double root_step(uint32_t dev, double refclk) const;
    std::pair<ReferenceState, const StepTransforms*> transforms_at(int64_t step);
    // Returns the root refclk of a reading at `refclk` on `dev`, using the link reference.
    std::pair<ReferenceState, double> reference(uint32_t dev, double refclk);
    bool place(uint32_t dev, ClockMap::Reader& reader);
    bool pair_up(double until);
    void prune();
    void run(std::stop_token stop);
    void report() const;

    const CaptureContext& ctx_;
    const ClockMap& map_;
    std::vector<LinkReference> links_;
    std::vector<Chip> chips_;
    std::vector<Pair> pairs_;
    std::map<int64_t, StepTransforms> transforms_by_step_;
    ErrorHistogram<16, 256> pooled_;
    WorstByMs worst_;
    bool finishing_seen_ = false;
    Batch staged_;
    std::mutex mu_;
    std::vector<Batch> submitted_;
    std::jthread worker_;
};

}  // namespace tt::tt_metal::streaming_profiler
