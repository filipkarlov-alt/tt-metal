// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device_types.hpp>
#include "hostdev/streaming_profiler_common.h"
#include "impl/context/context_types.hpp"

namespace tt {
class Cluster;
}
namespace tt::llrt {
class RunTimeOptions;
}
namespace tt::tt_fabric {
class ControlPlane;
class FabricContext;
class FabricNodeId;
}

namespace tt::tt_metal {

class Device;
class Hal;
class MetalContext;

namespace streaming_profiler {

// Without fabric, the profiler runs the link ends as resident kernels. With fabric, the routers on the chosen links run
// them, under LINK_SYNC_ROLE in fabric_erisc_router.cpp.
namespace link_sync {

struct Link {
    uint32_t chip_a = 0, chip_b = 0;  // chip_a < chip_b, and chip_a's end is the transmitter
    CoreCoord eth_a, eth_b;
};
// Returns every connected eth pair between the two chips, or with fabric on (`fabric` is its control plane, null when
// it is off), only those on the fabric's active channels.
std::vector<Link> links_between(
    const tt::Cluster& cluster, const tt_fabric::ControlPlane* fabric, uint32_t chip_x, uint32_t chip_y);

// Returns the role of this eth core's end on the link the profiler syncs through it, or None if it syncs no link.
kernel_profiler::LinkSyncRole role_of(
    const tt::Cluster& cluster, const tt_fabric::ControlPlane* fabric, ChipId chip, const CoreCoord& eth_logical);
// Returns the address of the link end's kernel_profiler::LinkSyncL1 on every active eth core. It is at the top of
// ACTIVE_ETH UNRESERVED, whether a resident kernel or a router runs the end.
uint32_t l1_addr(const Hal& hal);
// Returns the top of the L1 a fabric router may load into. That is the start of the link end's LinkSyncL1 whenever the
// profiler captures, and `router_limit` otherwise.
uint32_t router_l1_limit(const Hal& hal, const llrt::RunTimeOptions& rtoptions, uint32_t router_limit);
// Adds the named compile-time args that fabric_erisc_router.cpp reads whenever the JIT defines PROFILE_STREAMING on
// Blackhole. LINK_SYNC_ROLE is the role of the link end the router runs (only ERISC0's router runs one, and only where
// the profiler captures), and LINK_SYNC_L1_ADDR is the address of its LinkSyncL1.
void add_router_compile_args(
    const tt_fabric::FabricContext& fabric,
    uint32_t risc_id,
    const tt_fabric::FabricNodeId& node,
    const CoreCoord& eth_logical,
    std::unordered_map<std::string, uint32_t>& named_args);
// Zeroes every active eth core's link-end slots, control word and done word where the profiler captures. Call it before
// the fabric routers start, so a router end never reads what an earlier process left there.
void zero_link_end_l1(std::span<Device* const> devices, ContextId context_id);

}  // namespace link_sync

}  // namespace streaming_profiler
}  // namespace tt::tt_metal
