// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/check.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <limits>
#include <map>
#include <mutex>
#include <ranges>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

#include "impl/streaming_profiler/service.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

constexpr double kStepTicks = kRefclkTicksPerMs;
// A line through 50 ms of held-out rounds (about 45 of them) is good to about 0.1 ns, and two crystals stay on a line
// over that span to about 0.02 ns.
constexpr double kHalfSpanTicks = 25 * kRefclkTicksPerMs;
// A span only gets a line if it has rounds within 5 ms of both edges and at least 20 of its ~45 rounds, so no line
// extrapolates across a gap in its link's rounds.
constexpr double kEdgeTicks = 5 * kRefclkTicksPerMs;
constexpr size_t kMinLineRounds = 20;

constexpr double kMaxPairGapTicks = 0.05 * kRefclkTicksPerMs;

// AICLK bins that held for less than this share of the capture are left out of the report.
constexpr double kMinReportedBinShare = 1e-3;

int64_t step_of(double refclk) { return round_nearest(refclk / kStepTicks); }

}  // namespace

void SyncCheck::ErrorStats::add(double ns, double root, double weight) {
    weight_sum += weight;
    sum += weight * ns;
    if (std::abs(ns) > worst) {
        worst = std::abs(ns);
        worst_root = root;
    }
}

void SyncCheck::ClockStats::add(double root, double mhz) {
    if (last) {
        const double held = root - last->root;
        const double last_mhz = last->value;
        if (held > 0.0) {
            span += held;
            sum += held * last_mhz;
            sum_squares += held * last_mhz * last_mhz;
            lo = std::min(lo, last_mhz);
            hi = std::max(hi, last_mhz);
            by_bin[static_cast<int>(std::floor(last_mhz / kBinMhz))] += held;
        }
        changes += mhz != last_mhz ? 1 : 0;
    }
    last = PlotPoint{.root = root, .value = mhz};
}

void SyncCheck::WorstByMs::add(double root, double error_ns) {
    const auto ms = static_cast<int64_t>(std::floor(root / kRefclkTicksPerMs));
    if (worst_ns.empty()) {
        first_ms = ms;
    }
    for (; ms < first_ms; first_ms--) {
        worst_ns.push_front(std::nullopt);
    }
    while (ms >= first_ms + static_cast<int64_t>(worst_ns.size())) {
        worst_ns.push_back(std::nullopt);
    }
    std::optional<float>& worst = worst_ns[ms - first_ms];
    worst = std::max(worst.value_or(0.0f), static_cast<float>(std::abs(error_ns)));
}

SyncCheck::SyncCheck(const CaptureContext& ctx, const ClockMap& map) :
    ctx_(ctx), map_(map), links_(ctx.links.size()), chips_(ctx.devices.size()), staged_(ctx.devices.size()) {
    for (uint32_t a = 0; a < chips_.size(); a++) {
        for (uint32_t b = a + 1; b < chips_.size(); b++) {
            pairs_.push_back(Pair{.chip_a = a, .chip_b = b});
        }
    }
    chips_[CaptureContext::kRootDevice].root_minus_refclk = 0.0;
    worker_ = std::jthread([this](std::stop_token stop) { run(stop); });
}

void SyncCheck::submit() {
    std::lock_guard lock(mu_);
    submitted_.push_back(std::exchange(staged_, Batch(chips_.size())));
}

void SyncCheck::finish() {
    worker_.request_stop();
    worker_.join();
    report();
}

std::vector<SyncCheck::PlotSeries> SyncCheck::plots() const {
    const auto series = [&](std::string name, const WorstByMs& worst) {
        PlotSeries out{.name = std::move(name)};
        for (size_t i = 0; i < worst.worst_ns.size(); i++) {
            if (worst.worst_ns[i]) {
                out.points.push_back(
                    {.root = (static_cast<double>(worst.first_ms + static_cast<int64_t>(i)) + 0.5) * kRefclkTicksPerMs,
                     .value = *worst.worst_ns[i]});
            }
        }
        return out;
    };
    std::vector<PlotSeries> out{series("sync error bound (ns)", worst_)};
    for (uint32_t d = 0; d < chips_.size(); d++) {
        out.push_back(series(fmt::format("sync error bound chip{} (ns)", ctx_.devices[d].chip_id), chips_[d].worst));
    }
    return out;
}

int64_t SyncCheck::transmitter_step(size_t link_index, double root) const {
    return step_of(root - *chips_[ctx_.links[link_index].dev_a].root_minus_refclk);
}

double SyncCheck::root_step(uint32_t dev, double refclk) const {
    return (refclk + *chips_[dev].root_minus_refclk) / kStepTicks;
}

void SyncCheck::LinkReference::fit_steps(Steps steps) {
    if (rounds.empty()) {
        return;
    }
    if (!next_step) {
        next_step = step_of(rounds.front().mid);
    }
    const double newest = rounds.back().mid;
    size_t lo = 0;
    for (int64_t& step = *next_step;; step++) {
        const double centre = static_cast<double>(step) * kStepTicks;
        if (steps == Steps::All ? centre - kHalfSpanTicks > newest : newest < centre + kHalfSpanTicks) {
            break;
        }
        while (lo < rounds.size() && rounds[lo].mid < centre - kHalfSpanTicks) {
            lo++;
        }
        size_t hi = lo;
        while (hi < rounds.size() && rounds[hi].mid < centre + kHalfSpanTicks) {
            hi++;
        }
        const size_t count = hi - lo;
        if (count < kMinLineRounds || rounds[lo].mid > centre - kHalfSpanTicks + kEdgeTicks ||
            rounds[hi - 1].mid < centre + kHalfSpanTicks - kEdgeTicks) {
            continue;
        }
        lines_by_step[step] =
            fit_line(rounds | std::views::drop(lo) | std::views::take(count), &RoundPoint::mid, &RoundPoint::offset);
    }
    rounds.erase(rounds.begin(), rounds.begin() + static_cast<std::ptrdiff_t>(lo));
}

std::pair<SyncCheck::ReferenceState, const SyncCheck::StepTransforms*> SyncCheck::transforms_at(int64_t step) {
    auto it = transforms_by_step_.find(step);
    if (it == transforms_by_step_.end()) {
        const double centre = static_cast<double>(step) * kStepTicks;
        std::vector<const LineFit*> lines(ctx_.links.size(), nullptr);
        bool complete = true;
        for (size_t link_index = 0; link_index < lines.size() && complete; link_index++) {
            const std::optional<double>& transmitter_offset = chips_[ctx_.links[link_index].dev_a].root_minus_refclk;
            const LinkReference& link_reference = links_[link_index];
            const int64_t transmitter = transmitter_offset ? transmitter_step(link_index, centre) : 0;
            if (!finishing_seen_ &&
                (!transmitter_offset || !link_reference.next_step || transmitter >= *link_reference.next_step)) {
                return {ReferenceState::Wait, nullptr};
            }
            const auto line = transmitter_offset ? link_reference.lines_by_step.find(transmitter)
                                                 : link_reference.lines_by_step.end();
            complete = line != link_reference.lines_by_step.end();
            lines[link_index] = complete ? &line->second : nullptr;
        }
        it = transforms_by_step_.emplace(step, complete ? compose_on_root(ctx_, lines) : StepTransforms{}).first;
    }
    const StepTransforms& to_root = it->second;
    return {to_root.empty() ? ReferenceState::None : ReferenceState::Ready, &to_root};
}

std::pair<SyncCheck::ReferenceState, double> SyncCheck::reference(uint32_t dev, double refclk) {
    if (dev == CaptureContext::kRootDevice) {
        return {ReferenceState::Ready, refclk};
    }
    if (!chips_[dev].root_minus_refclk) {
        return {finishing_seen_ ? ReferenceState::None : ReferenceState::Wait, 0.0};
    }
    const double step = root_step(dev, refclk);
    const auto before = static_cast<int64_t>(std::floor(step));
    std::array<const StepTransforms*, 2> around{};
    for (int64_t i = 0; i < 2; i++) {
        const auto [state, transforms] = transforms_at(before + i);
        if (state != ReferenceState::Ready) {
            return {state, 0.0};
        }
        around[i] = transforms;
    }
    const double root_before = (*(*around[0])[dev])(refclk);
    const double root_after = (*(*around[1])[dev])(refclk);
    return {ReferenceState::Ready, root_before + (step - static_cast<double>(before)) * (root_after - root_before)};
}

bool SyncCheck::place(uint32_t dev, ClockMap::Reader& reader) {
    Chip& chip = chips_[dev];
    bool moved = false;
    for (; !chip.waiting.empty(); chip.waiting.pop_front()) {
        const Reading& reading = chip.waiting.front();
        const int64_t wall = whole_ticks(reading.wall_eighths);
        // The reading lies between its wall tick and the next one.
        if (!map_.is_final(reader, dev, wall + 1)) {
            break;
        }
        const auto [state, reference_root] = reference(dev, reading.refclk);
        if (state == ReferenceState::Wait) {
            break;
        }
        moved = true;
        chip.last_refclk = reading.refclk;
        if (state == ReferenceState::None) {
            chip.no_reference++;
            continue;
        }
        const std::optional<double> placed = map_.place_root(reader, dev, wall, tick_fraction(reading.wall_eighths));
        if (!placed) {
            chip.no_node++;
            continue;
        }
        const Sample sample{
            .root = reference_root,
            .error_ns = static_cast<float>((*placed - reference_root) * kNsPerRefclk),
            .weight = reading.weight};
        chip.placed.push_back(sample);
        chip.error.add(sample.error_ns, sample.root, sample.weight);
        worst_.add(sample.root, sample.error_ns);
        chip.worst.add(sample.root, sample.error_ns);
    }
    return moved;
}

bool SyncCheck::pair_up(double until) {
    bool moved = false;
    for (Pair& pair : pairs_) {
        Chip &chip_a = chips_[pair.chip_a], &chip_b = chips_[pair.chip_b];
        if (chip_b.placed.empty()) {
            continue;
        }
        const uint64_t b_end = chip_b.end_index();
        for (; pair.next_a < chip_a.end_index() && chip_a.at(pair.next_a).root < until; pair.next_a++) {
            const Sample& sample = chip_a.at(pair.next_a);
            while (pair.last_b + 1 < b_end && chip_b.at(pair.last_b + 1).root <= sample.root) {
                pair.last_b++;
            }
            const Sample* nearest = &chip_b.at(pair.last_b);
            if (pair.last_b + 1 < b_end &&
                std::abs(chip_b.at(pair.last_b + 1).root - sample.root) < std::abs(nearest->root - sample.root)) {
                nearest = &chip_b.at(pair.last_b + 1);
            }
            const double gap = std::abs(nearest->root - sample.root);
            if (gap > kMaxPairGapTicks) {
                continue;
            }
            const double error_ns = static_cast<double>(nearest->error_ns) - static_cast<double>(sample.error_ns);
            pair.error.add(error_ns, sample.root, sample.weight);
            pooled_.add(error_ns, sample.weight);
            worst_.add(sample.root, error_ns);
            chip_a.worst.add(sample.root, error_ns);
            chip_b.worst.add(sample.root, error_ns);
            moved = true;
        }
    }
    for (uint32_t d = 0; d < chips_.size(); d++) {
        Chip& chip = chips_[d];
        uint64_t keep = chip.end_index();
        for (const Pair& pair : pairs_) {
            keep = std::min(keep, pair.chip_a == d ? pair.next_a : pair.chip_b == d ? pair.last_b : keep);
        }
        for (; chip.popped < keep; chip.popped++) {
            chip.placed.pop_front();
        }
    }
    return moved;
}

void SyncCheck::prune() {
    int64_t oldest_step = std::numeric_limits<int64_t>::max();
    for (uint32_t d = 0; d < chips_.size(); d++) {
        const Chip& chip = chips_[d];
        if (!chip.root_minus_refclk || (chip.waiting.empty() && !chip.last_refclk)) {
            return;
        }
        const double refclk = chip.waiting.empty() ? *chip.last_refclk : chip.waiting.front().refclk;
        oldest_step = std::min(oldest_step, static_cast<int64_t>(std::floor(root_step(d, refclk))) - 1);
    }
    transforms_by_step_.erase(transforms_by_step_.begin(), transforms_by_step_.lower_bound(oldest_step));
    for (size_t link_index = 0; link_index < links_.size(); link_index++) {
        auto& lines = links_[link_index].lines_by_step;
        lines.erase(
            lines.begin(),
            lines.lower_bound(transmitter_step(link_index, static_cast<double>(oldest_step) * kStepTicks) - 1));
    }
}

void SyncCheck::run(std::stop_token stop) {
    set_thread_name("sp-check");
    ClockMap::Reader reader = map_.reader();
    std::vector<Batch> taken;
    while (true) {
        {
            std::lock_guard lock(mu_);
            taken.swap(submitted_);
            // Read under the lock, because submit()'s unlock happens before request_stop(), so the last batch is
            // visible.
            finishing_seen_ = stop.stop_requested();
        }
        bool moved = false;
        for (const Batch& batch : taken) {
            for (const Round& round : batch.rounds) {
                links_[round.link].rounds.push_back(round.point);
                moved = true;
            }
            for (uint32_t d = 0; d < chips_.size(); d++) {
                Chip& chip = chips_[d];
                const ChipBatch& from = batch.chips[d];
                chip.waiting.insert(chip.waiting.end(), from.readings.begin(), from.readings.end());
                for (const PlotPoint& aiclk : from.aiclk) {
                    chip.clock.add(aiclk.root, 1e3 * aiclk.value);
                }
                if (from.root_minus_refclk) {
                    chip.root_minus_refclk = from.root_minus_refclk;
                }
            }
        }
        taken.clear();
        for (LinkReference& link : links_) {
            link.fit_steps(finishing_seen_ ? Steps::All : Steps::Complete);
        }
        double placed_on_all_until = std::numeric_limits<double>::infinity();
        for (uint32_t d = 0; d < chips_.size(); d++) {
            Chip& chip = chips_[d];
            moved = place(d, reader) || moved;
            placed_on_all_until = std::min(
                placed_on_all_until,
                chip.placed.empty() ? -std::numeric_limits<double>::infinity() : chip.placed.back().root);
        }
        const double pair_until =
            finishing_seen_ ? std::numeric_limits<double>::infinity() : placed_on_all_until - kMaxPairGapTicks;
        moved = pair_up(pair_until) || moved;
        if (finishing_seen_) {
            return;
        }
        prune();
        if (!moved) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

void SyncCheck::report() const {
    double worst = 0.0, worst_root = 0.0, worst_mean = 0.0;
    std::string worst_of;
    size_t measured = 0;
    for (const Pair& pair : pairs_) {
        if (pair.error.weight_sum <= 0.0) {
            continue;
        }
        measured++;
        worst_mean = std::max(worst_mean, std::abs(pair.error.mean()));
        if (pair.error.worst > worst) {
            worst = pair.error.worst;
            worst_root = pair.error.worst_root;
            worst_of =
                fmt::format("chip {} - chip {}", ctx_.devices[pair.chip_a].chip_id, ctx_.devices[pair.chip_b].chip_id);
        }
    }
    uint64_t no_node = 0;
    for (uint32_t d = 0; d < chips_.size(); d++) {
        const Chip& chip = chips_[d];
        no_node += chip.no_node;
        if (chip.error.weight_sum > 0.0 && chip.error.worst > worst) {
            worst = chip.error.worst;
            worst_root = chip.error.worst_root;
            worst_of = fmt::format("chip {} against the reference", ctx_.devices[d].chip_id);
        }
    }
    if (no_node != 0) {
        log_warning(
            tt::LogMetal, "[streaming profiler] sync check: {} check readings the clock map could not place", no_node);
    }
    if (pooled_.weight_sum <= 0.0) {
        log_warning(tt::LogMetal, "[streaming profiler] sync check: no chip pair measured");
        return;
    }
    log_info(
        tt::LogMetal,
        "[streaming profiler] sync check: chip-to-chip error of the global timeline, a bound, over {} of {} chip "
        "pairs and {:.0f} samples: |err| p50 {:.2f}, p99 {:.2f}, p99.9 {:.2f}, max {:.2f} ns ({}, {:.3f} s in); "
        "the largest pair's mean {:.2f} ns",
        measured,
        pairs_.size(),
        pooled_.weight_sum,
        pooled_.abs_quantile(0.5),
        pooled_.abs_quantile(0.99),
        pooled_.abs_quantile(0.999),
        worst,
        worst_of,
        worst_root / kernel_profiler::kEthRefclkHz,
        worst_mean);
    for (uint32_t d = 0; d < chips_.size(); d++) {
        const Chip& chip = chips_[d];
        if (chip.error.weight_sum > 0.0) {
            log_info(
                tt::LogMetal,
                "[streaming profiler] sync check chip {}: {:.0f} readings against the reference, mean {:+.2f} ns, "
                "max {:.2f} ns ({:.3f} s in); unplaced {} with no reference, {} with no node",
                ctx_.devices[d].chip_id,
                chip.error.weight_sum,
                chip.error.mean(),
                chip.error.worst,
                chip.error.worst_root / kernel_profiler::kEthRefclkHz,
                chip.no_reference,
                chip.no_node);
        }
        const ClockStats& clock = chip.clock;
        if (clock.span > 0.0) {
            std::string bins;
            for (const auto& [bin, held] : clock.by_bin) {
                if (held >= kMinReportedBinShare * clock.span) {
                    bins += fmt::format(
                        "{}{:.0f} {:.1f}%",
                        bins.empty() ? "" : ", ",
                        bin * ClockStats::kBinMhz,
                        100.0 * held / clock.span);
                }
            }
            const double mean = clock.sum / clock.span;
            log_info(
                tt::LogMetal,
                "[streaming profiler] sync check chip {} AICLK: mean {:.0f} MHz, sd {:.0f}, {:.0f}-{:.0f} MHz, "
                "{:.1f} changes/s; time by {:.0f} MHz bin: {}",
                ctx_.devices[d].chip_id,
                mean,
                std::sqrt(std::max(0.0, clock.sum_squares / clock.span - mean * mean)),
                clock.lo,
                clock.hi,
                static_cast<double>(clock.changes) / (clock.span / kernel_profiler::kEthRefclkHz),
                ClockStats::kBinMhz,
                bins);
        }
    }
}

}  // namespace tt::tt_metal::streaming_profiler
