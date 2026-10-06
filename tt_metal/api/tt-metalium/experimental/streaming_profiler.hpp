// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>

// The streaming profiler's host API. A callback registered here receives the records that device kernels emit.
//
// This API is experimental and may change or be removed without notice.
//
//     Callback callback = RegisterCallback(
//         [](const Batch<Zone>& batch) {
//             for (const Zone& zone : batch.records<Zone>()) {
//                 use(zone.site().name, zone.core().logical, zone.duration());
//             }
//         },
//         "my-tool");
//
namespace tt::tt_metal::experimental::streaming_profiler {

enum class Processor : uint8_t { BRISC = 0, NCRISC = 1, TRISC0 = 2, TRISC1 = 3, TRISC2 = 4, ERISC0 = 5, ERISC1 = 6 };

struct SourceLocation {
    std::string_view file;  // Valid for the lifetime of the process.
    uint32_t line = 0;
};

/** @brief A marker's name and where it is written in kernel source code. */
struct MarkerSite {
    std::string_view name;  // Valid for the lifetime of the process.
    SourceLocation location;
};

/**
 * @brief The core a record came from.
 *
 * `logical` is the coordinate a program addresses it with; `physical` is its NoC 0 position on the die.
 * Tensix and Ethernet cores have separate logical grids, so two cores on a chip can share `logical` but never
 * `physical`.
 */
struct Core {
    CoreCoord logical;
    CoreCoord physical;
    ChipId chip_id = 0;
    Processor processor = Processor::BRISC;
};

/** @brief Site name of a stall zone. */
inline constexpr std::string_view STALL_ZONE_NAME = "PROFILER-STALL";

class Zone;
class TimestampedData;
class Event;
class Callback;

// Implementation detail.
namespace detail {
template <typename... Ts>
struct RecordList {
    static constexpr uint32_t count = sizeof...(Ts);
    template <typename T>
    static consteval uint32_t index() {
        constexpr bool matches[] = {std::is_same_v<T, Ts>...};
        for (uint32_t i = 0; i < count; i++) {
            if (matches[i]) {
                return i;
            }
        }
        return count;
    }
};
using RecordTypes = RecordList<Zone, TimestampedData, Event>;
template <typename T>
inline constexpr uint32_t record_index = RecordTypes::index<T>();
template <typename T>
inline constexpr uint32_t record_bit = record_index<T> < RecordTypes::count ? 1u << record_index<T> : 0;
template <typename... Ts>
consteval uint32_t record_mask() {
    static_assert(sizeof...(Ts) > 0 && ((record_bit<Ts> != 0) && ...), "Batch takes record types only");
    return (record_bit<Ts> | ...);
}

inline constexpr uint32_t ZONE_ID_BITS = 27;
inline constexpr uint32_t ZONE_LOCAL_BITS = 14;
inline constexpr uint32_t ZONE_TU_COUNT = 1u << (ZONE_ID_BITS - ZONE_LOCAL_BITS);

struct SiteTu {
    std::span<const MarkerSite* const> sites;
};
struct SiteRegistry {
    static std::atomic<const SiteTu*> tus[ZONE_TU_COUNT];
};
inline constexpr MarkerSite UNNAMED_SITE{};

inline const MarkerSite& site_of(uint32_t zone_id) {
    const SiteTu* tu = SiteRegistry::tus[zone_id >> ZONE_LOCAL_BITS].load(std::memory_order_acquire);
    const uint32_t local = zone_id & ((1u << ZONE_LOCAL_BITS) - 1u);
    const MarkerSite* s = tu != nullptr && local < tu->sites.size() ? tu->sites[local] : nullptr;
    return s != nullptr ? *s : UNNAMED_SITE;
}

struct Region {
    const void* data = nullptr;
    size_t count = 0;
};
struct BatchData {
    std::array<Region, RecordTypes::count> regions{};
    uint64_t dropped_bytes = 0;
    uint64_t stall_count = 0;
};
template <uint32_t RecordMask>
class RecordBatch {
public:
    template <typename T>
        requires((RecordMask & record_bit<T>) != 0)
    std::span<const T> records() const {
        const Region& region = data_.regions[record_index<T>];
        return {std::launder(static_cast<const T*>(region.data)), region.count};
    }
    uint64_t dropped_bytes() const { return data_.dropped_bytes; }
    uint64_t stall_count() const { return data_.stall_count; }

private:
    template <uint32_t M>
    friend RecordBatch<M> make_batch(const BatchData& data);
    explicit RecordBatch(const BatchData& data) : data_(data) {}

    BatchData data_;
};
enum class CallbackId : uint64_t {};
template <uint32_t M>
RecordBatch<M> make_batch(const BatchData& data) {
    return RecordBatch<M>(data);
}
inline void construct_timestamped_data(uint8_t* record, int64_t tsc);
Callback register_callback(std::string name, uint32_t types, std::function<void(const BatchData&)> callback);

template <typename Signature>
inline constexpr uint32_t batch_arg = 0;
template <typename R, uint32_t M>
inline constexpr uint32_t batch_arg<std::function<R(const RecordBatch<M>&)>> = M;
template <typename R, uint32_t M>
inline constexpr uint32_t batch_arg<std::function<R(RecordBatch<M>)>> = M;

template <typename F>
constexpr uint32_t accepted_batch() {
    using G = std::remove_reference_t<std::unwrap_ref_decay_t<F>>;
    if constexpr (requires { std::function{std::declval<G>()}; }) {
        return batch_arg<decltype(std::function{std::declval<G>()})>;
    } else {
        return 0;
    }
}

extern double ns_per_tsc_tick;

struct TscToSteadyLine {
    int64_t from = 0, to = -1;
    int64_t origin = 0, base_ns = 0;
    double value = 0.0, slope = 0.0;
};
inline constinit thread_local TscToSteadyLine tsc_to_steady_line{};
std::chrono::steady_clock::time_point refill_tsc_to_steady_line(int64_t tsc);

inline std::chrono::steady_clock::time_point tsc_to_steady(int64_t tsc) {
    const TscToSteadyLine& line = tsc_to_steady_line;
    if (tsc < line.from || tsc > line.to) {
        return refill_tsc_to_steady_line(tsc);
    }
    return std::chrono::steady_clock::time_point(std::chrono::nanoseconds(
        line.base_ns +
        static_cast<int64_t>(std::nearbyint(line.value + line.slope * static_cast<double>(tsc - line.origin)))));
}
}  // namespace detail

/** @brief Nanoseconds per host TSC tick. */
double NsPerTscTick() noexcept;

/** @brief Base class of every record, holding its site, core and program id. */
class Record {
public:
    /** @brief The marker this record came from. */
    const MarkerSite& site() const { return detail::site_of(zone_id_); }
    /** @brief The core that emitted the record. */
    Core core() const {
        return Core{
            .logical = CoreCoord(logical_x_, logical_y_),
            .physical = CoreCoord(physical_x_, physical_y_),
            .chip_id = chip_id_,
            .processor = processor_};
    }
    /** @brief Host runtime ID of the program. */
    uint32_t runtime_id() const { return runtime_id_; }

protected:
    uint64_t device_cycles_ = 0;
    uint32_t zone_id_ = 0;
    uint32_t runtime_id_ = 0;
    uint8_t logical_x_ = 0, logical_y_ = 0, physical_x_ = 0, physical_y_ = 0;
    uint16_t chip_id_ = 0;
    Processor processor_ = Processor::BRISC;
    int64_t tsc_ = 0;
};

/**
 * @brief One closed DeviceZoneScopedN scope.
 *
 * A stall, the time a core spent waiting for the profiler to drain its records, is delivered as a Zone with site name
 * STALL_ZONE_NAME.
 */
class Zone : public Record {
public:
    /** @brief When the device opened the zone, on steady_clock. */
    std::chrono::steady_clock::time_point start_time() const { return detail::tsc_to_steady(tsc_); }
    /** @brief When the device closed the zone, on steady_clock. */
    std::chrono::steady_clock::time_point end_time() const { return detail::tsc_to_steady(end_tsc_); }
    /** @brief When the device opened the zone, in host TSC ticks. */
    int64_t start_tsc() const { return tsc_; }
    /** @brief When the device closed the zone, in host TSC ticks. */
    int64_t end_tsc() const { return end_tsc_; }
    /** @brief When the zone opened, in device clock cycles. */
    uint64_t start_device_cycles() const { return device_cycles_; }
    /** @brief When the zone closed, in device clock cycles. */
    uint64_t end_device_cycles() const { return device_cycles_ + duration_cycles_; }
    /** @brief Length of the zone. */
    std::chrono::nanoseconds duration() const {
        return std::chrono::nanoseconds(
            static_cast<int64_t>(std::nearbyint(static_cast<double>(end_tsc_ - tsc_) * detail::ns_per_tsc_tick)));
    }
    /** @brief The device's average clock frequency over the zone. */
    double frequency_ghz() const {
        const int64_t tsc = end_tsc_ - tsc_;
        return tsc > 0 ? static_cast<double>(duration_cycles_) / (static_cast<double>(tsc) * detail::ns_per_tsc_tick)
                       : 0.0;
    }

private:
    uint64_t duration_cycles_ = 0;
    int64_t end_tsc_ = 0;
};

/** @brief Base class of records that mark one instant. */
class PointRecord : public Record {
public:
    /** @brief When the device recorded the marker, on steady_clock. */
    std::chrono::steady_clock::time_point time() const { return detail::tsc_to_steady(tsc_); }
    /** @brief When the device recorded the marker, in host TSC ticks. */
    int64_t tsc() const { return tsc_; }
    /** @brief When the device recorded the marker, in device clock cycles. */
    uint64_t device_cycles() const { return device_cycles_; }
};

/** @brief One DeviceTimestampedData marker and the values it recorded. */
class TimestampedData : public PointRecord {
public:
    TimestampedData() = default;
    TimestampedData(const TimestampedData& other);
    TimestampedData(TimestampedData&& other) noexcept;
    TimestampedData& operator=(TimestampedData other) noexcept;
    ~TimestampedData();

    /** @brief The marker's values. */
    std::span<const uint64_t> payload() const { return {values_, value_count_}; }

private:
    friend void detail::construct_timestamped_data(uint8_t* record, int64_t tsc);

    uint64_t value_count_ = 0;
    const uint64_t* values_ = nullptr;
};

/** @brief One DeviceRecordEvent marker. */
class Event : public PointRecord {};

/**
 * @brief The records of each type in Ts that a callback receives.
 *
 * - `records<T>()`: the batch's records of type T, one of Ts.
 * - `dropped_bytes()`: bytes of device output lost since this callback last ran; nonzero if the callback could not keep
 *   up with incoming data. Raising TT_METAL_STREAMING_PROFILER_FIFO_MB lets a callback fall further behind before it
 *   loses data.
 * - `stall_count()`: how many times any core waited for the profiler to drain its records since the previous
 *   batch.
 *
 * Records from one processor are in the order it emitted them, so its zones are ordered by end time. The records are
 * valid only inside the callback; to keep one, copy it.
 */
template <typename... Ts>
using Batch = detail::RecordBatch<detail::record_mask<Ts...>()>;

/** @brief A copy-constructible callable taking one `const Batch<Ts...>&`. */
template <typename F>
concept BatchCallable = detail::accepted_batch<F>() != 0 && std::is_copy_constructible_v<F>;

/**
 * @brief A registered callback, which stays registered until this object is destroyed, reset or assigned over.
 *
 * If the callback is running, unregistering waits for it to return. A callback may unregister only itself, which
 * returns at once and delivers it no further batches.
 */
class [[nodiscard]] Callback {
public:
    Callback() = default;
    Callback(Callback&& other) noexcept;
    Callback& operator=(Callback&& other) noexcept;
    ~Callback();

    /** @brief Unregisters the callback. */
    void reset() noexcept;

private:
    friend Callback detail::register_callback(
        std::string name, uint32_t types, std::function<void(const detail::BatchData&)> callback);
    explicit Callback(detail::CallbackId id) : id_(id) {}
    detail::CallbackId id_{};
};

/**
 * @brief Registers a callback to be invoked when streaming profiler records arrive from a device.
 *
 * Multiple callbacks can be registered; each runs on its own thread, one invocation at a time. If a callback shares a
 * resource with other callbacks, access it in a thread-safe way (e.g. with a lock). Callbacks that are too slow to keep
 * up with incoming data miss records; this is reported by Batch::dropped_bytes. TT_METAL_STREAMING_PROFILER_FIFO_MB
 * sets how far a callback can fall behind before it misses records. May be called before, during or between captures.
 *
 * @param callback Takes one `const Batch<Ts...>&`; Ts selects the record types it receives.
 * @param name Optional name for the callback in the profiler's logs and thread names.
 * @return The registration; the callback runs until it is destroyed.
 */
template <BatchCallable F>
Callback RegisterCallback(F callback, std::string name = {}) {
    constexpr uint32_t kAccepted = detail::accepted_batch<F>();
    return detail::register_callback(
        std::move(name), kAccepted, [callback = std::move(callback)](const detail::BatchData& data) mutable {
            callback(detail::make_batch<kAccepted>(data));
        });
}

/** @brief Returns true if the streaming profiler is currently running on at least one chip. */
bool IsActive();

}  // namespace tt::tt_metal::experimental::streaming_profiler
