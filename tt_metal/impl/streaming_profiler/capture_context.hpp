// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

#include <tt-metalium/experimental/streaming_profiler.hpp>

#include "hostdev/streaming_profiler_common.h"

namespace tt::tt_metal::streaming_profiler {

inline constexpr size_t kProcessorCount = static_cast<size_t>(experimental::streaming_profiler::Processor::ERISC1) + 1;
inline constexpr std::array<const char*, kProcessorCount> kProcessorNames = {
    "BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2", "ERISC_0", "ERISC_1"};

// 64x64 covers every supported grid, and a coordinate outside it is treated as unknown.
struct CoreTable {
    static constexpr uint16_t kNone = 0xFFFF;
    static constexpr uint32_t kCoordBits = 6;
    static constexpr uint32_t kCoordMask = (1u << kCoordBits) - 1;
    static constexpr size_t kSlots = size_t{1} << (2 * kCoordBits);
    std::vector<uint16_t> slot = std::vector<uint16_t>(kSlots, kNone);
    static uint32_t index(kernel_profiler::NocXy c) { return ((c.y & kCoordMask) << kCoordBits) | (c.x & kCoordMask); }
    uint16_t& operator[](uint32_t xy) { return slot[index(kernel_profiler::word_as<kernel_profiler::NocXy>(xy))]; }
    uint32_t find(uint32_t xy) const {
        const auto c = kernel_profiler::word_as<kernel_profiler::NocXy>(xy);
        return (c.x | c.y) > kCoordMask ? kNone : slot[index(c)];
    }
};

struct FileClose {
    void operator()(FILE* file) const { std::fclose(file); }
};

// Immutable once the receiver starts.
struct CaptureContext {
    // The device whose refclk the host reads over PCIe. Every chip's clock is mapped onto this device's refclk.
    static constexpr uint32_t kRootDevice = 0;
    struct Device {
        struct Tile {
            uint32_t xy = 0;  // kernel_profiler::NocXy
            // The wall-clock core's wall tick minus this core's. Every tile keeps its own wall clock on the same AICLK,
            // so this is a single integer for the whole capture.
            int64_t clock_offset = 0;
        };
        // kernel_profiler::PROFILER_SPSC_TENSIX_RISC lanes per tile, in tile order and then RISC order. An eth tile
        // fills its unused lanes with ERISC1.
        std::vector<experimental::streaming_profiler::Core> lanes;
        std::vector<Tile> tiles;  // by core index
        CoreTable core_of_xy;     // a tile's xy to its core index
        uint32_t chip_id = 0;
        // The wall-clock core's wall tick minus the check core's, or 0 without a check core.
        int64_t check_offset = 0;
    };
    std::vector<Device> devices;
    struct Link {
        uint32_t dev_a = 0, dev_b = 0;
        uint32_t core_a = 0, core_b = 0;
        CoreCoord eth_a, eth_b;
    };
    std::vector<Link> links;
    bool sync_check = false;
};

template <std::predicate<size_t> Usable>
std::vector<bool> reached_from_root(std::span<const CaptureContext::Link> links, size_t devices, Usable usable) {
    std::vector<bool> reached(devices, false);
    reached[CaptureContext::kRootDevice] = true;
    bool grew = false;
    do {
        grew = false;
        for (size_t link_index = 0; link_index < links.size(); link_index++) {
            const CaptureContext::Link& link = links[link_index];
            if (reached[link.dev_a] != reached[link.dev_b] && usable(link_index)) {
                reached[link.dev_a] = reached[link.dev_b] = true;
                grew = true;
            }
        }
    } while (grew);
    return reached;
}

}  // namespace tt::tt_metal::streaming_profiler
