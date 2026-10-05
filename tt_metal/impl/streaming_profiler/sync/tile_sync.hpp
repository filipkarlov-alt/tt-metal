// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <span>
#include <utility>

#include <tt-metalium/core_coord.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include "impl/context/context_types.hpp"

namespace tt::tt_metal {
class Device;

namespace streaming_profiler {

// Each tile's wall tick minus the chip's first tile's at the same instant, by core type and logical coordinate.
using TileClocks = std::map<std::pair<CoreType, CoreCoord>, int64_t>;

// Measures each chip's tile wall clocks against its first Tensix tile and hands them to the Service. Every tile's wall
// clock runs on the same AICLK, so each stays a fixed whole number of ticks from every other while the chip is up. The
// difference can reach hours, because the Tensix clocks halt while the chip idles between sessions and the eth and DRAM
// clocks don't. It must run before the fabric and dispatch firmware start, since it launches kernels on every Tensix,
// eth and DRAM core and lets only one tile per chip read at a time. A chip whose clocks this process has already
// measured keeps them.
void measure_tile_clocks(std::span<Device* const> devices, ContextId context_id);

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
