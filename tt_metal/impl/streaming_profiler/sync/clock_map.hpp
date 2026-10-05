// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

#include "tt_metal/common/broadcast_ring.hpp"

namespace tt::tt_metal::streaming_profiler {

// Reads the host TSC, the counter every host time in the profiler is in.
int64_t tsc_now() noexcept;
// Returns the host TSC's ns per tick. The first call blocks while it measures the rate.
double ns_per_tsc_tick();

// Rounds x to the nearest integer, ties to even.
inline int64_t round_nearest(double x) noexcept { return static_cast<int64_t>(std::nearbyint(x)); }

// A node of a piecewise-linear map from Key to a double. At key `at` the map's value is `value` and its slope is
// `tangent`.
template <typename Key>
struct ClockNode {
    Key at{};
    double value = 0.0;
    double tangent = 0.0;
};
using SyncNode = ClockNode<int64_t>;
using HostNode = ClockNode<double>;

struct ClockBases {
    int64_t root_refclk = 0, tsc = 0;
};

// A piecewise-linear map through its nodes, written by one thread and read lock-free by any thread. Keys up to `cover`
// are final: no later node changes what they map to.
template <typename Key>
struct ClockSeries {
    using Node = ClockNode<Key>;
    explicit ClockSeries(uint32_t series_nodes) : nodes(series_nodes) {}
    BroadcastRing<Node, SlotBacking::OnFirstWrite> nodes;
    Node last{.at = std::numeric_limits<Key>::lowest()};  // the writer's own copy
    std::atomic<Key> cover{std::numeric_limits<Key>::lowest()};

    void append(const Node& node);
    // `until` must not be below the cover.
    void extend(Key until) { cover.store(until, std::memory_order_release); }
};
extern template struct ClockSeries<int64_t>;
extern template struct ClockSeries<double>;

// One linear piece of a series, over keys [from, to]. A reader keeps the last one it found, so most lookups skip the
// search.
template <typename Key>
struct Segment {
    Key from = std::numeric_limits<Key>::max();
    Key to = std::numeric_limits<Key>::lowest();
    Key origin{};
    double value = 0.0;
    double slope = 0.0;
    uint64_t hint = 0;  // the index of the first node after `from`, where the next search starts
    bool holds(Key key) const noexcept { return key >= from && key <= to; }
    double at(Key key) const noexcept { return value + slope * static_cast<double>(key - origin); }
};

// Each chip's series maps its wall ticks onto the root chip's refclk, and the host series maps root refclk ticks onto
// the host TSC. One thread writes, and readers never lock or block it.
class ClockMap {
public:
    // At 24 B per node, this is at most 1.5 GB per chip. A chip adds a node at least every millisecond (kPointTicks in
    // kernels/eth_clock_model.cpp), so a series holds about 18.6 hours of a steady clock.
    static constexpr uint32_t kSeriesNodes = 1u << 26;

    // Placement state for one ClockMap, owned and used by a single thread.
    class Reader {
    public:
        Reader() = default;

    private:
        friend class ClockMap;
        Reader(size_t devices, int64_t tsc_base) : chips_(devices), tsc_base_(tsc_base) {}
        std::vector<Segment<int64_t>> chips_;
        Segment<double> host_;
        int64_t tsc_base_ = 0;
    };

    ClockMap(size_t devices, uint32_t series_nodes, ClockBases bases);
    Reader reader() const;

    // A node at or before its series' last node is dropped.
    void append(uint32_t dev, SyncNode node);
    // Appends `node` to the host series and makes keys up to `until` final along its tangent.
    void append_host(HostNode node, double until);
    // Makes every wall tick of the chip's series final.
    void finish(uint32_t dev);
    int64_t root_base() const noexcept { return root_base_; }

    // Returns the root refclk of the chip's wall tick plus `tick_fraction`, or nullopt before the chip's series has a
    // node.
    std::optional<double> place_root(Reader& reader, uint32_t dev, int64_t wall, double tick_fraction) const noexcept;
    int64_t place_host(Reader& reader, uint32_t dev, int64_t wall) const {
        if (const std::optional<int64_t> tsc = on_segments(reader.chips_[dev], reader.host_, reader.tsc_base_, wall)) {
            return *tsc;
        }
        return place_slow(reader, dev, wall);
    }
    // Returns the host TSC of a root refclk tick, or nullopt before the host series has a node.
    std::optional<int64_t> place_tsc(Reader& reader, double root) const noexcept;
    bool has_host_nodes() const noexcept;
    // Returns whether the host time of the chip's wall tick is final: no later node of either series can move it.
    bool is_final(Reader& reader, uint32_t dev, int64_t wall) const noexcept;

private:
    static std::optional<int64_t> on_segments(
        const Segment<int64_t>& chip, const Segment<double>& host, int64_t tsc_base, int64_t wall) {
        if (chip.holds(wall)) {
            const double root = chip.at(wall);
            if (host.holds(root)) {
                return tsc_base + round_nearest(host.at(root));
            }
        }
        return std::nullopt;
    }
    int64_t place_slow(Reader& reader, uint32_t dev, int64_t wall) const;

    std::deque<ClockSeries<int64_t>> chips_;
    ClockSeries<double> host_;
    const int64_t root_base_, tsc_base_;
};

// Maps host TSC to steady_clock ns. Each capture's sync thread appends under a lock, and ns() reads lock-free from any
// thread and caches the segment it used in the calling thread's detail::tsc_to_steady_line, which is why a process has
// only one.
class SteadyClock {
public:
    // Appends a pair of the host TSC and CLOCK_MONOTONIC read together.
    void sample();
    void append(int64_t tsc, int64_t mono_ns);
    std::optional<int64_t> ns(int64_t tsc) const noexcept;

private:
    friend class Service;
    SteadyClock() = default;

    std::mutex append_mu_;
    ClockSeries<int64_t> steady_{ClockMap::kSeriesNodes};
    std::optional<int64_t> base_ns_;
};

}  // namespace tt::tt_metal::streaming_profiler
