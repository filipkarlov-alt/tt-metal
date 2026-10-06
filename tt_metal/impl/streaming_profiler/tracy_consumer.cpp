// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/tracy_consumer.hpp"

#if defined(TRACY_ENABLE)

#include <algorithm>

#include <fmt/format.h>
#include <common/TracyTTDeviceData.hpp>
#include <tracy/Tracy.hpp>
#include <client/TracyProfiler.hpp>

namespace tt::tt_metal::streaming_profiler {

namespace {

constexpr size_t kSrclocTableInitial = 1024;

uint64_t srcloc_key(std::string_view name, uint32_t processor) {
    return (static_cast<uint64_t>(name.size()) << 32) | processor;
}

size_t srcloc_hash(const char* name, uint64_t key) {
    return (((reinterpret_cast<uintptr_t>(name) >> 3) ^ key) * 0x9E3779B97F4A7C15ull) >> 32;
}

constexpr uint32_t kProcessorColors[kProcessorCount] = {
    tracy::Color::Orange2,
    tracy::Color::SeaGreen3,
    tracy::Color::SkyBlue3,
    tracy::Color::Turquoise2,
    tracy::Color::CadetBlue1,
    tracy::Color::Yellow3,
    tracy::Color::Yellow2};

// Each processor's row on the timeline, in the lane id space Tracy keeps apart from host threads. It uses the physical
// coordinate, because an eth core's logical coordinate can equal a Tensix core's, and the low 3 bits are the processor.
uint32_t lane_thread(const experimental::streaming_profiler::Core& core) {
    return tracy::TTDeviceMarker::LANE_ID_FLAG | (static_cast<uint32_t>(core.chip_id) << 16) |
           (static_cast<uint32_t>(core.physical.y) << 11) | (static_cast<uint32_t>(core.physical.x) << 3) |
           static_cast<uint32_t>(core.processor);
}

}  // namespace

TracyConsumer::TracyConsumer() : anchor_tracy_(tracy::Profiler::GetTime()), srcloc_table_(kSrclocTableInitial) {}

void TracyConsumer::operator()(const Batch& batch) {
    using experimental::streaming_profiler::Event;
    using experimental::streaming_profiler::TimestampedData;
    using experimental::streaming_profiler::Zone;
    for (const Zone& zone : batch.records<Zone>()) {
        push_zone(zone.core(), zone.site().name, zone.start_tsc(), zone.end_tsc());
    }
    for (const TimestampedData& data : batch.records<TimestampedData>()) {
        push_marker(data.core(), data.site().name, data.tsc(), data.runtime_id(), data.payload());
    }
    for (const Event& event : batch.records<Event>()) {
        push_marker(event.core(), event.site().name, event.tsc(), event.runtime_id(), {});
    }
}

// Record TSCs and Tracy's GetTime() count the same host counter, so a record's offset from a GetTime() anchor, scaled
// by Tracy's multiplier, is its place on Tracy's timeline.
int64_t TracyConsumer::timeline_ns(int64_t tsc) const {
    return static_cast<int64_t>(static_cast<double>(tsc - anchor_tracy_) * TracyGetTimerMul());
}

TracyConsumer::Lane TracyConsumer::lane(const Core& core) {
    const uint64_t key = lane_thread(core);
    if (key == lane_key_) {
        return lane_hit_;
    }
    Lane found;
    found.thread = static_cast<uint32_t>(key);
    found.processor = static_cast<uint32_t>(core.processor);
    auto [it, fresh] = cores_.try_emplace(key & ~uint64_t{0x7});
    CoreEntry& entry = it->second;
    if (fresh) {
        entry.ctx = TracyTTContext();
        // Everything this consumer emits goes through this thread's lock-free queue, and its FIFO order keeps the
        // context ahead of the zones that refer to it.
        TracyTTContextPopulateCalibratedLockfree(entry.ctx, anchor_tracy_, 0.0, 1.0);
        const std::string name = fmt::format(
            "Device: {}, {}Logical ({},{}) Physical ({},{})",
            core.chip_id,
            core.processor >= experimental::streaming_profiler::Processor::ERISC0 ? "Ethernet " : "",
            core.logical.x,
            core.logical.y,
            core.physical.x,
            core.physical.y);
        TracyTTContextNameLockfree(entry.ctx, name.c_str(), name.size());
    }
    if ((entry.named & (1u << found.processor)) == 0) {
        tracy::SetThreadName(found.thread, kProcessorNames[found.processor]);
        entry.named |= static_cast<uint8_t>(1u << found.processor);
    }
    found.ctx = entry.ctx;
    lane_key_ = key;
    lane_hit_ = found;
    return found;
}

const tracy::SourceLocationData* TracyConsumer::srcloc(std::string_view name, uint32_t processor) {
    const uint64_t key = srcloc_key(name, processor);
    const size_t mask = srcloc_table_.size() - 1;
    const char* const name_ptr = name.data();  // NOLINT(bugprone-suspicious-stringview-data-usage)
    for (size_t i = srcloc_hash(name_ptr, key) & mask;; i = (i + 1) & mask) {
        const SrclocEntry& e = srcloc_table_[i];
        if (e.name == name_ptr && e.key == key) {
            return e.srcloc;
        }
        if (e.name == nullptr) {
            return srcloc_slow(name, processor);
        }
    }
}

const tracy::SourceLocationData* TracyConsumer::srcloc_slow(std::string_view name, uint32_t processor) {
    const char* const name_ptr = name.data();  // NOLINT(bugprone-suspicious-stringview-data-usage)
    const uint64_t key = srcloc_key(name, processor);
    const uint32_t color = name == experimental::streaming_profiler::STALL_ZONE_NAME ? tracy::Color::Tomato3
                                                                                     : kProcessorColors[processor];
    auto [it, fresh] = srclocs_.try_emplace({std::string(name), color});
    if (fresh) {
        it->second = {it->first.first.c_str(), "kernel_profiler", "kernel_profiler", 0, color};
    }
    auto insert = [](std::vector<SrclocEntry>& table, const SrclocEntry& e) {
        const size_t mask = table.size() - 1;
        size_t i = srcloc_hash(e.name, e.key) & mask;
        while (table[i].name != nullptr) {
            i = (i + 1) & mask;
        }
        table[i] = e;
    };
    if ((srcloc_count_ + 1) * 2 > srcloc_table_.size()) {
        std::vector<SrclocEntry> old = std::move(srcloc_table_);
        srcloc_table_.assign(old.size() * 2, {});
        for (const SrclocEntry& e : old) {
            if (e.name != nullptr) {
                insert(srcloc_table_, e);
            }
        }
    }
    insert(srcloc_table_, {name_ptr, key, &it->second});
    srcloc_count_++;
    return &it->second;
}

void TracyConsumer::push_zone(const Core& core, std::string_view name, int64_t start_tsc, int64_t end_tsc) {
    const int64_t start_ns = timeline_ns(start_tsc);
    const int64_t end_ns = std::max(timeline_ns(end_tsc), start_ns);
    const Lane zone_lane = lane(core);
    TracyTTPushZone(
        zone_lane.ctx,
        srcloc(name, zone_lane.processor),
        zone_lane.thread,
        static_cast<uint64_t>(start_ns),
        static_cast<uint64_t>(end_ns));
}

void TracyConsumer::push_marker(
    const Core& core, std::string_view name, int64_t tsc, uint32_t runtime_id, std::span<const uint64_t> values) {
    const int64_t timestamp_ns = timeline_ns(tsc);
    const Lane marker_lane = lane(core);
    tracy::TTDeviceMarker marker;
    marker.color = kProcessorColors[marker_lane.processor];
    marker.timestamp = static_cast<uint64_t>(timestamp_ns);
    marker.runtime_host_id = runtime_id;
    marker.marker_type = values.empty() ? tracy::TTDeviceMarkerType::EVENT : tracy::TTDeviceMarkerType::DATA;
    marker.marker_name = std::string(name);
    marker.file = "kernel_profiler";
    marker.line = 0;
    if (!values.empty()) {
        marker.data = values[0];
    }
    if (values.size() > 1) {
        marker.data_high = values[1];
    }
#ifdef TRACY_TT_HAS_FULL_DEPS
    for (size_t i = 2; i < values.size(); i++) {
        marker.meta_data[fmt::format("value{}", i)] = values[i];
    }
#endif
    TracyTTPushMarkerLockfree(marker_lane.ctx, marker, marker_lane.thread);
}

}  // namespace tt::tt_metal::streaming_profiler

#endif
