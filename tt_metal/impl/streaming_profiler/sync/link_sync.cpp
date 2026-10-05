// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/link_sync.hpp"

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <vector>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <umd/device/types/core_coordinates.hpp>

#include "context/metal_context.hpp"
#include "fabric/fabric_context.hpp"
#include "impl/device/device_impl.hpp"
#include "impl/streaming_profiler/device_programs.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler::link_sync {

namespace {

bool on_synced_channel(
    const tt::Cluster& cluster, const tt_fabric::ControlPlane* fabric, uint32_t chip, const CoreCoord& eth_logical) {
    if (fabric == nullptr) {
        return true;
    }
    const auto& soc = cluster.get_soc_desc(static_cast<ChipId>(chip));
    const auto channels = fabric->get_active_fabric_eth_channels(
        fabric->get_fabric_node_id_from_physical_chip_id(static_cast<ChipId>(chip)));
    return std::ranges::any_of(channels | std::views::keys, [&](const auto channel) {
        return soc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL) == eth_logical;
    });
}

}  // namespace

std::vector<Link> links_between(
    const tt::Cluster& cluster, const tt_fabric::ControlPlane* fabric, uint32_t chip_x, uint32_t chip_y) {
    std::vector<Link> out;
    const uint32_t lo = std::min(chip_x, chip_y), hi = std::max(chip_x, chip_y);
    const auto by_peer = cluster.get_ethernet_cores_grouped_by_connected_chips(static_cast<ChipId>(lo));
    const auto it = by_peer.find(static_cast<ChipId>(hi));
    if (it == by_peer.end()) {
        return out;
    }
    for (const CoreCoord& eth_a : it->second) {
        const CoreCoord eth_b =
            std::get<1>(cluster.get_connected_ethernet_core(std::make_tuple(static_cast<ChipId>(lo), eth_a)));
        if (on_synced_channel(cluster, fabric, lo, eth_a) && on_synced_channel(cluster, fabric, hi, eth_b)) {
            out.push_back(Link{.chip_a = lo, .chip_b = hi, .eth_a = eth_a, .eth_b = eth_b});
        }
    }
    return out;
}

kernel_profiler::LinkSyncRole role_of(
    const tt::Cluster& cluster, const tt_fabric::ControlPlane* fabric, ChipId chip, const CoreCoord& eth_logical) {
    const auto& connections = cluster.get_ethernet_connections();
    const auto on_chip = connections.find(chip);
    if (on_chip == connections.end() ||
        !on_chip->second.contains(cluster.get_soc_desc(chip).logical_eth_core_to_chan_map.at(eth_logical))) {
        return kernel_profiler::LinkSyncRole::None;
    }
    const auto self = static_cast<uint32_t>(chip);
    const auto peer =
        static_cast<uint32_t>(std::get<0>(cluster.get_connected_ethernet_core(std::make_tuple(chip, eth_logical))));
    for (const Link& link : links_between(cluster, fabric, self, peer)) {
        if (link.chip_a == self && link.eth_a == eth_logical) {
            return kernel_profiler::LinkSyncRole::Transmitter;
        }
        if (link.chip_b == self && link.eth_b == eth_logical) {
            return kernel_profiler::LinkSyncRole::Receiver;
        }
    }
    return kernel_profiler::LinkSyncRole::None;
}

uint32_t l1_addr(const Hal& hal) {
    return hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) +
           hal.get_dev_size(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED) -
           sizeof(kernel_profiler::LinkSyncL1);
}

uint32_t router_l1_limit(const Hal& hal, const llrt::RunTimeOptions& rtoptions, uint32_t router_limit) {
    return can_capture(hal, rtoptions) ? l1_addr(hal) : router_limit;
}

void add_router_compile_args(
    const tt_fabric::FabricContext& fabric,
    uint32_t risc_id,
    const tt_fabric::FabricNodeId& node,
    const CoreCoord& eth_logical,
    std::unordered_map<std::string, uint32_t>& named_args) {
    const Hal& hal = fabric.get_hal();
    if (hal.get_arch() != tt::ARCH::BLACKHOLE) {
        return;
    }
    auto role = kernel_profiler::LinkSyncRole::None;
    if (risc_id == 0 && can_capture(hal, fabric.get_rtoptions())) {
        const tt_fabric::ControlPlane& control_plane = fabric.get_control_plane();
        role = role_of(
            fabric.get_cluster(),
            &control_plane,
            control_plane.get_physical_chip_id_from_fabric_node_id(node),
            eth_logical);
    }
    named_args["LINK_SYNC_ROLE"] = static_cast<uint32_t>(role);
    named_args["LINK_SYNC_L1_ADDR"] = l1_addr(hal);
}

void zero_link_end_l1(std::span<Device* const> devices, ContextId context_id) {
    auto& mc = MetalContext::instance(context_id);
    if (!can_capture(mc)) {
        return;
    }
    const uint32_t addr = l1_addr(mc.hal());
    for (Device* device : devices) {
        for (const CoreCoord& logical : device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false)) {
            zero_l1(
                mc.get_cluster(),
                device->id(),
                device->ethernet_core_from_logical_core(logical),
                addr,
                offsetof(kernel_profiler::LinkSyncL1, ring));
        }
    }
}

}  // namespace tt::tt_metal::streaming_profiler::link_sync
