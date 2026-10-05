// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tests the Blackhole Ethernet 1588 layer on every active eth link between two chips: kRounds round trips with one-step
// egress stamps and an RX stamp rule, then kTwoStepFrames two-step stamped frames. It needs slow dispatch
// (TT_METAL_SLOW_DISPATCH_MODE=1) on a multi-chip Blackhole system.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/distributed.hpp>

#include "device_fixture.hpp"
#include "eth_test_common.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ptp_stamps.hpp"

namespace tt::tt_metal {
namespace {

using eth_ptp_stamps::kStampTickNs;
using eth_ptp_stamps::Result;

constexpr const char* kKernel = "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/eth_ptp_stamps.cpp";

// One round trip's four stamps. Each end's stamps are on its own PTP time.
struct RoundTrip {
    double transmitter_egress, receiver_ingress, receiver_egress, transmitter_ingress;
};

std::vector<RoundTrip> round_trips(const Result& transmitter, const Result& receiver) {
    std::vector<RoundTrip> trips;
    for (uint32_t i = 0; i < eth_ptp_stamps::kRounds; i++) {
        trips.push_back(
            {.transmitter_egress = static_cast<double>(receiver.stamps[i].peer_egress),
             .receiver_ingress = static_cast<double>(receiver.stamps[i].ingress),
             .receiver_egress = static_cast<double>(transmitter.stamps[i].peer_egress),
             .transmitter_ingress = static_cast<double>(transmitter.stamps[i].ingress)});
    }
    return trips;
}

// Compares only stamps from the same end.
void expect_causal(std::span<const RoundTrip> trips, const std::string& link) {
    for (size_t i = 0; i < trips.size(); i++) {
        const RoundTrip& trip = trips[i];
        EXPECT_LT(trip.transmitter_egress, trip.transmitter_ingress) << link << " round " << i;
        EXPECT_LT(trip.receiver_ingress, trip.receiver_egress) << link << " round " << i;
        if (i > 0) {
            EXPECT_GT(trip.transmitter_egress, trips[i - 1].transmitter_ingress) << link << " round " << i;
            EXPECT_GT(trip.receiver_ingress, trips[i - 1].receiver_egress) << link << " round " << i;
        }
    }
}

// Each stamp is its true time rounded down to a multiple of kStampTickNs, so a correctly paired round trip is within a
// tick of the truth and never two ticks from the median one-way time or from the line fitted to the offsets. A missing
// or mispaired stamp is off by far more.
void expect_steady(std::span<const RoundTrip> trips, const std::string& link) {
    const auto n = static_cast<double>(trips.size());
    std::vector<double> one_way, midpoint, offset;
    double midpoint_mean = 0, offset_mean = 0;
    for (const RoundTrip& trip : trips) {
        one_way.push_back(
            0.5 *
            ((trip.transmitter_ingress - trip.transmitter_egress) - (trip.receiver_egress - trip.receiver_ingress)));
        midpoint.push_back(0.5 * (trip.transmitter_egress + trip.transmitter_ingress));
        offset.push_back(
            0.5 *
            ((trip.receiver_ingress + trip.receiver_egress) - (trip.transmitter_egress + trip.transmitter_ingress)));
        midpoint_mean += midpoint.back() / n;
        offset_mean += offset.back() / n;
    }
    std::vector<double> ranked = one_way;
    const auto middle = ranked.begin() + ranked.size() / 2;
    std::ranges::nth_element(ranked, middle);
    const double one_way_median = *middle;
    fmt::print(
        "{}: one-way {:.2f} ns, the mean of {} round trips\n",
        link,
        std::reduce(one_way.begin(), one_way.end()) / n,
        trips.size());
    double midpoint_variance = 0, midpoint_offset_covariance = 0;
    for (size_t i = 0; i < trips.size(); i++) {
        midpoint_variance += (midpoint[i] - midpoint_mean) * (midpoint[i] - midpoint_mean);
        midpoint_offset_covariance += (midpoint[i] - midpoint_mean) * (offset[i] - offset_mean);
    }
    const double slope = midpoint_offset_covariance / midpoint_variance;
    for (size_t i = 0; i < trips.size(); i++) {
        EXPECT_LT(std::abs(one_way[i] - one_way_median), 2 * kStampTickNs) << link << " round " << i;
        EXPECT_LT(std::abs(offset[i] - (offset_mean + slope * (midpoint[i] - midpoint_mean))), 2 * kStampTickNs)
            << link << " round " << i;
    }
}

void run_link(
    MeshDispatchFixture* fixture,
    const std::shared_ptr<distributed::MeshDevice>& transmitter_mesh,
    const std::shared_ptr<distributed::MeshDevice>& receiver_mesh,
    const CoreCoord& transmitter_core,
    const CoreCoord& receiver_core) {
    IDevice* transmitter_device = transmitter_mesh->get_devices()[0];
    IDevice* receiver_device = receiver_mesh->get_devices()[0];
    const uint32_t base =
        MetalContext::instance().hal().get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED);
    const uint32_t result_addr = base + eth_ptp_stamps::kResultOffset;
    std::vector<uint32_t> zeros(sizeof(Result) / sizeof(uint32_t), 0);
    detail::WriteToDeviceL1(transmitter_device, transmitter_core, result_addr, zeros, CoreType::ETH);
    detail::WriteToDeviceL1(receiver_device, receiver_core, result_addr, zeros, CoreType::ETH);

    const auto make_workload = [&](const CoreCoord& core, bool transmitter) {
        Program program;
        EthernetConfig config{.compile_args = {transmitter ? 1u : 0u}};
        eth_test_common::set_arch_specific_eth_config(config);
        const KernelHandle kernel = CreateKernel(program, kKernel, core, config);
        SetRuntimeArgs(program, kernel, core, {base});
        distributed::MeshWorkload workload;
        const distributed::MeshCoordinate zero_coord(0, 0);
        workload.add_program(distributed::MeshCoordinateRange(zero_coord, zero_coord), std::move(program));
        return workload;
    };
    distributed::MeshWorkload transmitter_workload = make_workload(transmitter_core, true);
    distributed::MeshWorkload receiver_workload = make_workload(receiver_core, false);
    std::thread transmitter_thread([&] { fixture->RunProgram(transmitter_mesh, transmitter_workload); });
    std::thread receiver_thread([&] { fixture->RunProgram(receiver_mesh, receiver_workload); });
    transmitter_thread.join();
    receiver_thread.join();

    const std::string link = fmt::format(
        "chip {} eth {} -> chip {} eth {}",
        transmitter_device->id(),
        transmitter_core.str(),
        receiver_device->id(),
        receiver_core.str());
    const auto read = [&](IDevice* dev, const CoreCoord& core) {
        Result result{};
        detail::ReadFromDeviceL1(
            dev, core, result_addr, std::span(reinterpret_cast<uint8_t*>(&result), sizeof(result)), CoreType::ETH);
        return result;
    };
    const Result transmitter = read(transmitter_device, transmitter_core);
    const Result receiver = read(receiver_device, receiver_core);

    for (const auto& [result, end] : {std::pair{&transmitter, "transmitter"}, std::pair{&receiver, "receiver"}}) {
        ASSERT_EQ(result->done, eth_ptp_stamps::kDone) << link << ": the " << end << " kernel did not finish";
        EXPECT_EQ(result->header_select_after, result->header_select_before)
            << link << ": the " << end << "'s TX header row selection was not restored";
        EXPECT_EQ(result->no_match_after, result->no_match_before)
            << link << ": the " << end << "'s RX no-match actions were not restored";
        // Both clocks are read just after a refclk update, so they agree to the ns when the offset is right.
        EXPECT_EQ(result->restart_error_ns, 0)
            << link << ": the " << end << "'s PTP time is " << result->restart_error_ns << " ns off refclk * "
            << kStampTickNs << " ns";
        ASSERT_EQ(result->rounds, eth_ptp_stamps::kRounds)
            << link << ": the " << end << " stopped waiting for its peer";
        EXPECT_EQ(result->unstamped, 0u) << link << ": frames to the " << end << " without an egress stamp";
        EXPECT_EQ(result->ingress_mismatched, 0u)
            << link << ": " << end
            << " frames whose ingress stamp wasn't the RX FIFO's only entry with the stamp rule's label";
    }
    EXPECT_EQ(transmitter.two_step_matched, eth_ptp_stamps::kTwoStepFrames)
        << link << ": two-step frames without a matching, in-window TX FIFO entry";

    const std::vector<RoundTrip> trips = round_trips(transmitter, receiver);
    expect_causal(trips, link);
    expect_steady(trips, link);
}

}  // namespace

TEST_F(MeshDeviceFixture, ActiveEthPtpStamps) {
    if (arch_ != tt::ARCH::BLACKHOLE) {
        GTEST_SKIP() << "Blackhole only";
    }
    auto& cluster = MetalContext::instance().get_cluster();
    std::map<ChipId, std::shared_ptr<distributed::MeshDevice>> mesh_of;
    for (const auto& mesh : devices_) {
        mesh_of.emplace(mesh->get_device_ids()[0], mesh);
    }
    size_t links = 0;
    for (const auto& [transmitter_chip, transmitter_mesh] : mesh_of) {
        for (const auto& [receiver_chip, cores] :
             cluster.get_ethernet_cores_grouped_by_connected_chips(transmitter_chip)) {
            const auto peer = mesh_of.find(receiver_chip);
            if (receiver_chip <= transmitter_chip || peer == mesh_of.end()) {
                continue;
            }
            for (const CoreCoord& transmitter_core : cores) {
                const CoreCoord receiver_core = std::get<1>(
                    cluster.get_connected_ethernet_core(std::make_tuple(transmitter_chip, transmitter_core)));
                run_link(this, transmitter_mesh, peer->second, transmitter_core, receiver_core);
                links++;
            }
        }
    }
    if (links == 0) {
        GTEST_SKIP() << "no ethernet link between local chips";
    }
}

}  // namespace tt::tt_metal
