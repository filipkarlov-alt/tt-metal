// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <new>
#include <span>
#include <type_traits>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "impl/streaming_profiler/spsc_marker_decode.hpp"
#include "impl/streaming_profiler/capture_context.hpp"

namespace tt::tt_metal::streaming_profiler {

static_assert(sizeof(experimental::streaming_profiler::Record) == profiler::kSpscEventBytes);
static_assert(sizeof(experimental::streaming_profiler::PointRecord) == profiler::kSpscEventBytes);
static_assert(sizeof(experimental::streaming_profiler::Zone) == profiler::kSpscZoneBytes);
static_assert(sizeof(experimental::streaming_profiler::Event) == profiler::kSpscEventBytes);
static_assert(sizeof(experimental::streaming_profiler::TimestampedData) == profiler::kSpscDataBytes);
static_assert(std::is_trivially_copyable_v<experimental::streaming_profiler::Zone>);
static_assert(std::is_trivially_copyable_v<experimental::streaming_profiler::Event>);
static_assert(profiler::kSpscNRiscDecode == static_cast<size_t>(experimental::streaming_profiler::Processor::ERISC0));

struct SpscLane {
    uint64_t cursor = 0;   // the last absolute zone end, which a delta16 zone's end delta counts from
    uint64_t last_ts = 0;  // lanes emit in end order, so a step back is a torn read
    // A regression repairs that record in place, so this is only valid while the decode_frames output numbered
    // last_rec_seq hasn't been delivered.
    uint8_t* last_rec = nullptr;
    uint64_t last_rec_seq = 0;
    bool last_rec_zone = false;  // a repair may fix a zone's duration in qword 4, but data keeps its value count there
    uint32_t timer_hi = 0;       // the sticky wall-clock high word
    uint32_t prog = 0;           // the sticky runtime id
    bool seeded = false;
    bool need_state = true;
    // The producer writes the first zone after a launch or rewind as an absolute ZONE_ATOMIC, and a resync recovers at
    // the next one.
    bool need_anchor = true;
    profiler::SpscRecConsts rec{};
    profiler::SpscLaneConsts consts{};
};

class StreamDecoder {
public:
    struct Out {
        uint8_t* zones;
        uint8_t* events;
        uint8_t* data;
        uint8_t* values;
    };
    struct Produced {
        uint32_t zones, events, data, values;
        // The newest last record among the lanes the call advanced, in the wall ticks of the chip's wall-clock core, or
        // INT64_MIN if none of them has a record yet.
        int64_t newest_ticks;
        uint64_t stalls;
        uint64_t order_regressions;
    };
    struct Capacity {
        size_t zone_bytes, event_bytes, data_bytes, value_bytes;
    };
    static_assert(std::ranges::all_of(profiler::kFormats, [](const profiler::PacketFormat& format) {
        return format.kind == profiler::Kind::Sticky || format.words >= 2;
    }));
    // A packet that becomes a record is at least two words, so each kind takes at most words / 2 records, plus the
    // whole quads the block kernels write past their last record. Values take two words each, plus the 32 bytes per
    // frame that a point kernel stores past its last value.
    static constexpr Capacity out_capacity(size_t words, uint32_t frames) {
        const size_t records = words / 2 + profiler::kSpscSinkSlackRecs;
        return Capacity{
            .zone_bytes = records * profiler::kSpscZoneBytes,
            .event_bytes = records * profiler::kSpscEventBytes,
            .data_bytes = records * profiler::kSpscDataBytes,
            .value_bytes = words * 4 + size_t{32} * frames};
    }

    // `dev` must outlive the decoder. It produces only records of `types`, and steps over the other packets.
    StreamDecoder(const CaptureContext::Device& dev, experimental::streaming_profiler::RecordType types);
    // Size `out` with out_capacity(). A later call may repair each lane's last record in place, so keep the output
    // writable until its commit().
    Produced decode_frames(const uint32_t* frames, std::span<const uint32_t> frame_words, Out out);

    // Marks the oldest undelivered decode_frames output as delivered. Outputs are delivered in decode order.
    void commit() { delivered_seq_++; }

private:
    std::vector<SpscLane> lanes_;
    // decode_frames reads a core's heads with one 256-bit load.
    static constexpr size_t kHeadsPerLoad = 8;
    static_assert(profiler::kSpscNRiscDecode <= kHeadsPerLoad);
    std::vector<std::array<uint32_t, kHeadsPerLoad>> heads_;
    const CoreTable& core_of_xy_;
    // One bit per wire type the decoder steps over instead of producing.
    uint32_t skipped_types_ = 0;
    uint64_t decoded_seq_ = 0, delivered_seq_ = 0;
};

}  // namespace tt::tt_metal::streaming_profiler

namespace tt::tt_metal::experimental::streaming_profiler {
inline void detail::construct_timestamped_data(
    void* at, const PointRecord& point, int64_t tsc, uint64_t value_count, const uint64_t* values) {
    new (at) TimestampedData(point, tsc, value_count, values);
}
}  // namespace tt::tt_metal::experimental::streaming_profiler

namespace tt::tt_metal::streaming_profiler {

// Constructs the TimestampedData held in the decoded bytes at `record`, placed at host TSC `tsc`. The record borrows
// its values from the consumer's arena, which reclaims both without running ~TimestampedData.
inline void construct_timestamped_data(uint8_t* record, int64_t tsc) {
    experimental::streaming_profiler::PointRecord point;
    uint64_t value_count = 0;
    uint64_t* values = nullptr;
    std::memcpy(&point, record, sizeof(point));
    std::memcpy(&value_count, record + profiler::kSpscSharedBytes, sizeof(value_count));
    std::memcpy(&values, record + profiler::kSpscSharedBytes + sizeof(value_count), sizeof(values));
    // A memmove onto itself implicitly creates the uint64_t objects that the decoder's vector stores wrote as bytes.
    experimental::streaming_profiler::detail::construct_timestamped_data(
        record,
        point,
        tsc,
        value_count,
        static_cast<const uint64_t*>(std::memmove(values, values, value_count * sizeof(uint64_t))));
}

}  // namespace tt::tt_metal::streaming_profiler
