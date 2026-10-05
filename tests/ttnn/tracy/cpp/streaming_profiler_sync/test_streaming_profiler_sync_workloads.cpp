// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The sync check's device workloads, one per subcommand: idle, host_sync and didt. Each opens the system mesh with the
// streaming profiler on. host_sync and didt check the timeline it records. Needs a Blackhole system.

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <ranges>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt_stl/assert.hpp>

#include "impl/context/metal_context.hpp"
#include "kernels/workload_layout.hpp"
#include "llrt/tt_cluster.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/matmul/matmul.hpp"
#include "ttnn/operations/transformer/sdpa/sdpa.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

using namespace tt;
using namespace tt::tt_metal;
namespace streaming_profiler = tt::tt_metal::experimental::streaming_profiler;

namespace {

constexpr std::string_view kKernelDir = "tests/ttnn/tracy/cpp/streaming_profiler_sync/kernels/";

// One region of every worker's L1, taken from the allocator, that holds the kernels' flag (FlagWord) and the
// multicast's round values.
struct WorkerL1 {
    static constexpr uint32_t kFlagBytes = 64;
    static constexpr uint32_t kValuesBytes = 64 * 1024;
    std::shared_ptr<distributed::MeshBuffer> buffer;
    uint32_t flag() const { return static_cast<uint32_t>(buffer->address()); }
    uint32_t ack() const { return flag() + kAckWord * sizeof(uint32_t); }
    uint32_t values() const { return flag() + kFlagBytes; }
};

// One page per L1 bank, and every worker is one bank, so each worker holds the region at the same address.
WorkerL1 reserve_worker_l1(distributed::MeshDevice& mesh) {
    constexpr uint32_t kPerCore = WorkerL1::kFlagBytes + WorkerL1::kValuesBytes;
    const uint32_t banks = mesh.allocator()->get_num_banks(BufferType::L1);
    return {distributed::MeshBuffer::create(
        distributed::ReplicatedBufferConfig{.size = uint64_t{kPerCore} * banks},
        distributed::DeviceLocalBufferConfig{.page_size = kPerCore, .buffer_type = BufferType::L1},
        &mesh)};
}

using Clock = std::chrono::steady_clock;
double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

double median(std::vector<double> values) {
    std::ranges::nth_element(values, values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2));
    return values[values.size() / 2];
}

void run_once(distributed::MeshCommandQueue& cq, distributed::MeshWorkload& workload) {
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
    distributed::Finish(cq);
}

std::shared_ptr<distributed::MeshDevice> open_system_mesh() {
    return distributed::MeshDevice::create(
        distributed::MeshDeviceConfig(distributed::SystemMesh::instance().shape()),
        DEFAULT_L1_SMALL_SIZE,
        DEFAULT_TRACE_REGION_SIZE,
        /*num_command_queues=*/1);
}

CoreRange worker_grid(distributed::MeshDevice& mesh) {
    const CoreCoord grid = mesh.compute_with_storage_grid_size();
    return CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
}

struct FlagCore {
    IDevice* device;
    CoreCoord core;
};

void clear_flags(const WorkerL1& worker_l1, const std::vector<FlagCore>& cores) {
    std::vector<uint32_t> zero(WorkerL1::kFlagBytes / sizeof(uint32_t), 0);
    for (const FlagCore& flag_core : cores) {
        detail::WriteToDeviceL1(flag_core.device, flag_core.core, worker_l1.flag(), zero);
    }
}

}  // namespace

// Opens the mesh and sleeps for `seconds`, so the sync check measures clocks no workload disturbs.
namespace idle {

int run(double seconds) {
    auto mesh_device = open_system_mesh();
    std::printf("[idle] %zu chips, %.1f s\n", mesh_device->num_devices(), seconds);
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    mesh_device->close();
    return 0;
}

}  // namespace idle

// Checks the device-to-host part of the timeline. For each round the host writes a round number into one worker's L1
// on every chip and polls for that worker's ack, and the worker records a HOST_RX zone in between. Placed on the host
// timeline, every zone must start after the host began its write and before it read the ack. The window is the host's
// MMIO round trip, a few microseconds wide, so the check is loose and catches only a host mapping off by more than
// that. It also fails unless a callback that unregisters itself on its first call runs exactly once.
namespace host_sync {

namespace {
constexpr uint32_t kRounds = 2000;
constexpr auto kAckTimeout = std::chrono::seconds(1);
// Spaced so the capture runs long enough for the sync check to measure, and spans seconds of clock drift.
constexpr auto kRoundGap = std::chrono::milliseconds(1);

struct Window {
    Clock::time_point before, after;
};

double to_us(Clock::duration duration) { return std::chrono::duration<double, std::micro>(duration).count(); }
}  // namespace

int run() {
    std::map<uint32_t, std::vector<Clock::time_point>> starts;
    uint64_t dropped_bytes = 0;
    auto registration = streaming_profiler::RegisterCallback(
        [&](const streaming_profiler::Batch<streaming_profiler::RecordType::Zones>& batch) {
            dropped_bytes += batch.dropped_bytes();
            for (const streaming_profiler::Zone& zone : batch.zones()) {
                if (std::string_view(zone.site().name) == "HOST_RX") {
                    starts[zone.core().chip_id].push_back(zone.start_time());
                }
            }
        },
        "host_sync");
    std::atomic<uint32_t> once_calls{0};
    streaming_profiler::Callback once;
    once = streaming_profiler::RegisterCallback(
        [&](const streaming_profiler::Batch<streaming_profiler::RecordType::Zones>&) {
            if (once_calls++ == 0) {
                once.reset();
            }
        },
        "host_sync-once");

    auto mesh_device = open_system_mesh();
    const WorkerL1 worker_l1 = reserve_worker_l1(*mesh_device);
    const CoreCoord core(0, 0);
    std::vector<FlagCore> cores;
    for (IDevice* device : mesh_device->get_devices()) {
        cores.push_back({device, core});
    }
    clear_flags(worker_l1, cores);

    Program program = CreateProgram();
    const auto kernel = CreateKernel(
        program,
        std::string(kKernelDir) + "host_poke_dm.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    SetRuntimeArgs(program, kernel, core, {worker_l1.flag(), kRounds});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device->shape()), std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);

    // ReadFromDeviceL1 barriers every core on the chip first, which takes about 0.5 ms and would widen each window by
    // that much, so the ack poll reads the word directly.
    const auto& cluster = MetalContext::instance().get_cluster();
    std::map<uint32_t, std::vector<Window>> windows;
    bool timed_out = false;
    std::vector<uint32_t> round_word(1, 0);
    for (uint32_t round = 1; round <= kRounds && !timed_out; round++) {
        std::this_thread::sleep_for(kRoundGap);
        round_word[0] = round;
        for (const FlagCore& flag_core : cores) {
            const tt_cxy_pair ack_core(
                flag_core.device->id(),
                flag_core.device->virtual_core_from_logical_core(flag_core.core, CoreType::WORKER));
            const Clock::time_point before = Clock::now();
            detail::WriteToDeviceL1(flag_core.device, flag_core.core, worker_l1.flag(), round_word);
            while (true) {
                uint32_t ack = 0;
                cluster.read_core(&ack, sizeof(ack), ack_core, worker_l1.ack());
                if (ack == round) {
                    break;
                }
                if (Clock::now() - before > kAckTimeout) {
                    std::printf("[host_sync] chip %d gave no ack for round %u\n", flag_core.device->id(), round);
                    timed_out = true;
                    break;
                }
            }
            windows[flag_core.device->id()].push_back({before, Clock::now()});
        }
    }
    distributed::Finish(cq);
    mesh_device->close();
    registration.reset();

    size_t failed = 0;
    for (const auto& [chip, chip_windows] : windows) {
        const std::vector<Clock::time_point>& zone_starts = starts[chip];
        if (zone_starts.size() != chip_windows.size()) {
            std::printf(
                "[host_sync] chip %u: FAIL, %zu zones for %zu rounds\n", chip, zone_starts.size(), chip_windows.size());
            failed++;
            continue;
        }
        size_t outside = 0;
        for (size_t k = 0; k < chip_windows.size(); k++) {
            const double write_to_zone_us = to_us(zone_starts[k] - chip_windows[k].before);
            const double zone_to_ack_us = to_us(chip_windows[k].after - zone_starts[k]);
            if (write_to_zone_us < 0.0 || zone_to_ack_us < 0.0) {
                if (outside++ == 0) {
                    std::printf(
                        "[host_sync] chip %u round %zu: the zone is %.2f us after the write and "
                        "%.2f us before the ack\n",
                        chip,
                        k + 1,
                        write_to_zone_us,
                        zone_to_ack_us);
                }
            }
        }
        const bool pass = outside == 0;
        failed += pass ? 0 : 1;
        std::printf(
            "[host_sync] chip %u: %s, %zu of %zu zones outside their window\n",
            chip,
            pass ? "PASS" : "FAIL",
            outside,
            chip_windows.size());
    }
    const bool pass = failed == 0 && !timed_out && dropped_bytes == 0 && once_calls == 1;
    std::printf(
        "[host_sync] %s: %zu of %zu chips failed, %llu bytes dropped, a callback that unregisters itself ran %u "
        "times%s\n",
        pass ? "PASS" : "FAIL",
        failed,
        windows.size(),
        static_cast<unsigned long long>(dropped_bytes),
        once_calls.load(),
        timed_out ? ", an ack timed out" : "");
    return pass ? 0 : 1;
}

}  // namespace host_sync

// Checks each chip's timeline core to core. One Tensix core per chip broadcasts rounds over each NoC, and every worker
// records the arrival as an MC_RX zone. A multicast takes a fixed 9 cycles per router hop, so once each arrival is on
// the host timeline and its hops are subtracted, every core should report the same instant. A run fails if any core's
// mean differs from the others' by a cycle or more, a round goes missing or records are dropped.
namespace multicast {

namespace {
constexpr uint32_t kRounds = 4000;
static_assert(kRoundValueStrideBytes * (kRounds + 1) <= WorkerL1::kValuesBytes);
static_assert(kRounds % 2 == 0);
// The Blackhole NoC documentation gives 9 cycles per router hop.
constexpr double kCyclesPerHop = 9.0;

struct Arrival {
    int64_t tsc;
    int64_t cycles;
    uint16_t chip;
    uint8_t logical_x, logical_y;
    uint8_t physical_x, physical_y;
};

distributed::MeshWorkload make_multicast(
    distributed::MeshDevice& mesh, const WorkerL1& worker_l1, uint32_t runtime_id) {
    const CoreRange cores = worker_grid(mesh);
    const CoreCoord &low = cores.start_coord, &high = cores.end_coord;
    const CoreCoord virtual_low = mesh.worker_core_from_logical_core(low);
    const CoreCoord virtual_high = mesh.worker_core_from_logical_core(high);
    const uint32_t num_dests = static_cast<uint32_t>(cores.size() - 1);
    Program program = CreateProgram();
    program.set_runtime_id(runtime_id);
    const auto kernel = CreateKernel(
        program,
        std::string(kKernelDir) + "multicast_dm.cpp",
        CoreRangeSet(cores),
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
    for (const CoreCoord& core : cores) {
        const MulticastRole role = core == low ? MulticastRole::Noc0Source
                                               : (core == high ? MulticastRole::Noc1Source : MulticastRole::Receiver);
        // A NoC 1 rectangle runs from its start corner at the high coordinates down to the low ones.
        const CoreCoord& rect_start = role == MulticastRole::Noc1Source ? virtual_high : virtual_low;
        const CoreCoord& rect_end = role == MulticastRole::Noc1Source ? virtual_low : virtual_high;
        SetRuntimeArgs(
            program,
            kernel,
            core,
            {static_cast<uint32_t>(role),
             static_cast<uint32_t>(rect_start.x),
             static_cast<uint32_t>(rect_start.y),
             static_cast<uint32_t>(rect_end.x),
             static_cast<uint32_t>(rect_end.y),
             num_dests,
             worker_l1.flag(),
             worker_l1.values(),
             kRounds});
    }
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh.shape()), std::move(program));
    return workload;
}

// A chip's arrivals by receiving core, each core's in arrival order.
using CoreArrivals = std::map<CoreCoord, std::vector<const Arrival*>>;

// Groups the arrivals by chip and core, leaving out a chip with a core missing or with the wrong number of arrivals.
// A source receives only the other source's rounds.
std::map<uint32_t, CoreArrivals> group_arrivals(const std::vector<Arrival>& arrivals, const CoreRange& cores) {
    std::map<uint32_t, CoreArrivals> by_chip;
    for (const Arrival& arrival : arrivals) {
        by_chip[arrival.chip][CoreCoord{arrival.logical_x, arrival.logical_y}].push_back(&arrival);
    }
    std::erase_if(by_chip, [&](auto& chip_arrivals) {
        auto& [chip, by_core] = chip_arrivals;
        bool counts_ok = by_core.size() == cores.size();
        for (auto& [core, core_arrivals] : by_core) {
            std::ranges::sort(core_arrivals, {}, &Arrival::cycles);
            const size_t expected = core == cores.start_coord || core == cores.end_coord ? kRounds / 2 : kRounds;
            if (core_arrivals.size() != expected) {
                std::printf(
                    "[multicast] chip %u core (%zu,%zu): %zu arrivals, expected %zu\n",
                    chip,
                    core.x,
                    core.y,
                    core_arrivals.size(),
                    expected);
                counts_ok = false;
            }
        }
        return !counts_ok;
    });
    return by_chip;
}

// Returns, in cycles, how far apart the cores' mean arrival times of `noc`'s multicasts are once each arrival's hops
// are subtracted. Each arrival is taken relative to the reference core's in the same round, because absolute TSC times
// summed over the rounds would lose nanoseconds in a double. Subtracting a per-round time moves every core's mean
// alike, so the span stays as it is.
double lane_span_cycles(const CoreArrivals& by_core, const CoreRange& cores, uint32_t noc) {
    const double ns_per_tsc = streaming_profiler::NsPerTscTick();
    const CoreCoord& source = noc == 0 ? cores.start_coord : cores.end_coord;
    const CoreCoord& other_source = noc == 0 ? cores.end_coord : cores.start_coord;
    // A receiver's arrivals alternate the NoC 0 source's odd rounds and the NoC 1 source's even ones.
    const auto arrival = [&](const CoreCoord& core, size_t round) {
        return by_core.at(core)[core == other_source ? round : 2 * round + noc];
    };
    const CoreCoord src(by_core.at(source).front()->physical_x, by_core.at(source).front()->physical_y);
    const CoreCoord& reference =
        by_core.begin()->first == source ? std::next(by_core.begin())->first : by_core.begin()->first;
    constexpr size_t rounds = kRounds / 2;
    std::vector<double> ns_per_cycle(rounds);
    for (size_t k = 0; k < rounds; k++) {
        const Arrival* next = arrival(reference, k + 1 < rounds ? k + 1 : k - 1);
        ns_per_cycle[k] = static_cast<double>(next->tsc - arrival(reference, k)->tsc) * ns_per_tsc /
                          static_cast<double>(next->cycles - arrival(reference, k)->cycles);
    }
    double min_mean_ns = std::numeric_limits<double>::max(), max_mean_ns = std::numeric_limits<double>::lowest();
    for (const auto& [core, arrivals] : by_core) {
        if (core == source) {
            continue;
        }
        const CoreCoord physical(arrivals.front()->physical_x, arrivals.front()->physical_y);
        const int hops_x = noc == 0 ? static_cast<int>(physical.x) - static_cast<int>(src.x)
                                    : static_cast<int>(src.x) - static_cast<int>(physical.x);
        const int hops_y = noc == 0 ? static_cast<int>(physical.y) - static_cast<int>(src.y)
                                    : static_cast<int>(src.y) - static_cast<int>(physical.y);
        double sum_ns = 0.0;
        for (size_t k = 0; k < rounds; k++) {
            sum_ns += static_cast<double>(arrival(core, k)->tsc - arrival(reference, k)->tsc) * ns_per_tsc -
                      kCyclesPerHop * (hops_x + hops_y) * ns_per_cycle[k];
        }
        const double mean_ns = sum_ns / static_cast<double>(rounds);
        min_mean_ns = std::min(min_mean_ns, mean_ns);
        max_mean_ns = std::max(max_mean_ns, mean_ns);
    }
    return (max_mean_ns - min_mean_ns) /
           (std::reduce(ns_per_cycle.begin(), ns_per_cycle.end()) / static_cast<double>(rounds));
}

// Collects every run's arrivals by the run's runtime id. Records reach the host by the capture's end, so verify() runs
// after the mesh has closed.
class Check {
public:
    Check() :
        registration_(streaming_profiler::RegisterCallback(
            [this](const streaming_profiler::Batch<streaming_profiler::RecordType::Zones>& batch) {
                dropped_bytes_ += batch.dropped_bytes();
                dropped_bytes_ += batch.dropped_bytes();
                for (const streaming_profiler::Zone& zone : batch.zones()) {
                    if (std::string_view(zone.site().name) != "MC_RX") {
                        continue;
                    }
                    const streaming_profiler::Core core = zone.core();
                    by_run_[zone.runtime_id()].push_back(Arrival{
                        .tsc = zone.start_tsc(),
                        .cycles = static_cast<int64_t>(zone.start_device_cycles()),
                        .chip = static_cast<uint16_t>(core.chip_id),
                        .logical_x = static_cast<uint8_t>(core.logical.x),
                        .logical_y = static_cast<uint8_t>(core.logical.y),
                        .physical_x = static_cast<uint8_t>(core.physical.x),
                        .physical_y = static_cast<uint8_t>(core.physical.y)});
                }
            },
            "multicast")) {}
    Check(const Check&) = delete;
    Check& operator=(const Check&) = delete;

    void run(distributed::MeshDevice& mesh, std::string_view name) {
        cores_ = worker_grid(mesh);
        num_chips_ = mesh.get_devices().size();
        const WorkerL1 worker_l1 = reserve_worker_l1(mesh);
        std::vector<FlagCore> cores;
        for (IDevice* device : mesh.get_devices()) {
            for (const CoreCoord& core : cores_) {
                cores.push_back({device, core});
            }
        }
    }
    clear_flags(worker_l1, flag_cores);
    distributed::MeshWorkload workload = make_multicast(mesh, worker_l1, runtime_id);
    run_once(mesh.mesh_command_queue(), workload);
    std::printf(
        "[multicast] %zux%zu Tensix cores x %u rounds (%u per NoC) on %zu chips\n",
        cores.grid_size().x,
        cores.grid_size().y,
        kRounds,
        kRounds / 2,
        mesh.get_devices().size());
}

// Returns whether a run's arrivals reached every core of `num_chips` chips and put each chip's cores within a cycle of
// each other.
bool verify(const std::vector<Arrival>& arrivals, const CoreRange& cores, size_t num_chips) {
    const std::map<uint32_t, CoreArrivals> by_chip = group_arrivals(arrivals, cores);
    double worst_span_cycles = 0.0;
    for (const auto& [chip, by_core] : by_chip) {
        for (const uint32_t noc : {0u, 1u}) {
            const double span_cycles = lane_span_cycles(by_core, cores, noc);
            worst_span_cycles = std::max(worst_span_cycles, span_cycles);
            std::printf("chip %u NoC %u: span %.2f cycles\n", chip, noc, span_cycles);
        }
    }
    const bool complete = by_chip.size() == num_chips;
    std::printf(
        "[multicast] worst core-to-core span of the device-local timeline %.2f cycles%s\n",
        worst_span_cycles,
        complete ? "" : "; some chips missing or with wrong arrival counts");
    return worst_span_cycles < 1.0 && complete;
}
}  // namespace

}  // namespace multicast

// Runs fabric traffic under the profiler. Over 2D fabric, one worker on each side of every linked chip pair ping-pongs
// atomic increments for kSeconds, recording a PP_TX zone per send and a PP_RX zone per arrival. A run fails if a kernel
// gives up waiting for its peer or any round's zones don't all reach the host. As a sanity check of the synced
// timeline, it also fails if a pair's first rounds break causality or its two directions' replies take different times.
namespace pingpong {

namespace {
constexpr double kSeconds = 10.0;
// A pass takes at least 36 ms (1.81 us per round trip), so the host work between passes is a small part of the time.
constexpr uint32_t kRounds = 20000;

// One end of a linked chip pair: its worker and the link it sends over.
struct End {
    uint32_t chip;
    CoreCoord core;
    uint32_t link;
};
// Each end's PingpongRole is its index.
using Pair = std::array<End, 2>;

constexpr uint32_t kTimedRounds = 1000;
struct ZoneStart {
    uint64_t cycles;
    Clock::time_point start;
    bool tx;
};
// Every zone is counted. The first kTimedRounds rounds' zones are also kept for the timeline check.
struct CoreZones {
    size_t tx = 0, rx = 0;
    std::vector<ZoneStart> timed;
};
using ZonesByCore = std::map<std::pair<uint32_t, CoreCoord>, CoreZones>;
struct Chip {
    distributed::MeshCoordinate coord;
    IDevice* device;
    tt::tt_fabric::FabricNodeId node;
};
using Chips = std::map<uint32_t, Chip>;

std::vector<Pair> find_pairs(distributed::MeshDevice& mesh_device, const Chips& chips) {
    const CoreCoord grid = mesh_device.compute_with_storage_grid_size();
    std::map<uint32_t, uint32_t> next_worker;
    const auto take_worker = [&](uint32_t chip) {
        const uint32_t worker = next_worker[chip]++;
        return CoreCoord(worker % grid.x, worker / grid.x);
    };
    std::vector<Pair> pairs;
    for (const auto& [id_a, chip_a] : chips) {
        for (const auto& [id_b, chip_b] : chips) {
            if (id_a >= id_b || tt::tt_fabric::get_neighbor_eth_directions(chip_a.node, chip_b.node).empty()) {
                continue;
            }
            const auto links_ab = tt::tt_fabric::get_forwarding_link_indices(chip_a.node, chip_b.node);
            const auto links_ba = tt::tt_fabric::get_forwarding_link_indices(chip_b.node, chip_a.node);
            TT_FATAL(!links_ab.empty() && !links_ba.empty(), "no fabric link between chips {} and {}", id_a, id_b);
            pairs.push_back(
                {End{.chip = id_a, .core = take_worker(id_a), .link = links_ab.front()},
                 End{.chip = id_b, .core = take_worker(id_b), .link = links_ba.front()}});
        }
    }
    return pairs;
}

distributed::MeshWorkload build_workload(
    distributed::MeshDevice& mesh_device,
    const WorkerL1& worker_l1,
    const std::vector<Pair>& pairs,
    const Chips& chips,
    uint32_t runtime_id) {
    std::map<uint32_t, Program> programs;
    for (const uint32_t chip : std::views::keys(chips)) {
        programs.emplace(chip, CreateProgram()).first->second.set_runtime_id(runtime_id);
    }
    for (const Pair& pair : pairs) {
        for (uint32_t role = 0; role < pair.size(); role++) {
            const End &self = pair[role], &peer = pair[1 - role];
            const tt::tt_fabric::FabricNodeId &src = chips.at(self.chip).node, &dst = chips.at(peer.chip).node;
            Program& program = programs.at(self.chip);
            const auto kernel = CreateKernel(
                program,
                std::string(kKernelDir) + "pingpong_fabric_dm.cpp",
                self.core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
            const CoreCoord virtual_peer = mesh_device.worker_core_from_logical_core(peer.core);
            std::vector<uint32_t> args = {
                role,
                static_cast<uint32_t>(virtual_peer.x),
                static_cast<uint32_t>(virtual_peer.y),
                worker_l1.flag(),
                kRounds,
                static_cast<uint32_t>(dst.chip_id),
                static_cast<uint32_t>(dst.mesh_id.get())};
            tt::tt_fabric::append_fabric_connection_rt_args(src, dst, self.link, program, self.core, args);
            SetRuntimeArgs(program, kernel, self.core, args);
        }
    }
    distributed::MeshWorkload workload;
    for (auto& [chip, program] : programs) {
        const auto& coord = chips.at(chip).coord;
        workload.add_program(distributed::MeshCoordinateRange(coord, coord), std::move(program));
    }
    return workload;
}

uint32_t count_missing(const std::vector<Pair>& pairs, ZonesByCore& by_core, size_t expected) {
    uint32_t failures = 0;
    for (const Pair& pair : pairs) {
        for (const End& end : pair) {
            const CoreZones& zones = by_core[{end.chip, end.core}];
            if (zones.tx != expected || zones.rx != expected) {
                std::printf(
                    "chip %u core (%zu,%zu): FAIL, %zu PP_TX and %zu PP_RX zones, of %zu each\n",
                    end.chip,
                    end.core.x,
                    end.core.y,
                    zones.tx,
                    zones.rx,
                    expected);
                failures++;
            }
        }
    }
    return failures;
}
// Half the reply difference is the pair's clock error plus half the link's asymmetry. On a LoudBox it stays within
// +-5 ns, and this bound is twice that.
constexpr double kReplySkewBoundNs = 10.0;

uint32_t check_timeline(const std::vector<Pair>& pairs, ZonesByCore& by_core) {
    uint32_t failures = 0;
    for (const Pair& pair : pairs) {
        const auto& [a, b] = pair;
        std::vector<ZoneStart>& timed_a = by_core[{a.chip, a.core}].timed;
        std::vector<ZoneStart>& timed_b = by_core[{b.chip, b.core}].timed;
        for (std::vector<ZoneStart>* timed : {&timed_a, &timed_b}) {
            std::ranges::sort(*timed, {}, &ZoneStart::cycles);
        }
        enum Direction : size_t { kAToB, kBToA };
        std::array<std::vector<double>, 2> replies;
        uint32_t acausal = 0, misread = 0;
        for (size_t k = 0; k < std::min(timed_a.size(), timed_b.size()) / 2; k++) {
            // Round k + 1: b sends first on odd rounds, a on even ones.
            const bool b_first = (k & 1u) == 0;
            const ZoneStart* sender = &(b_first ? timed_b : timed_a)[2 * k];
            const ZoneStart* replier = &(b_first ? timed_a : timed_b)[2 * k];
            if (!sender[0].tx || sender[1].tx || replier[0].tx || !replier[1].tx) {
                misread++;
                continue;
            }
            const double ping_ns = std::chrono::duration<double, std::nano>(replier[0].start - sender[0].start).count();
            const double reply_ns =
                std::chrono::duration<double, std::nano>(sender[1].start - replier[1].start).count();
            acausal += (ping_ns <= 0.0 ? 1u : 0u) + (reply_ns <= 0.0 ? 1u : 0u);
            replies[b_first ? kAToB : kBToA].push_back(reply_ns);
        }
        if (replies[kAToB].empty() || replies[kBToA].empty()) {
            std::printf("chip %u - chip %u: FAIL, no timed rounds in some direction\n", a.chip, b.chip);
            failures++;
            continue;
        }
        const double reply_skew_ns = (median(replies[kAToB]) - median(replies[kBToA])) / 2;
        const bool pair_ok = acausal == 0 && misread == 0 && std::abs(reply_skew_ns) <= kReplySkewBoundNs;
        std::printf(
            "chip %u - chip %u: %s, half the reply difference %+.1f ns; %u acausal, %u misread of %zu rounds\n",
            a.chip,
            b.chip,
            pair_ok ? "ok" : "FAIL",
            reply_skew_ns,
            acausal,
            misread,
            replies[kAToB].size() + replies[kBToA].size() + misread);
        failures += pair_ok ? 0 : 1;
    }
    return failures;
}

// Like multicast::Check, for the PP_TX and PP_RX zones.
class Check {
public:
    Check() :
        registration_(streaming_profiler::RegisterCallback(
            [this](const streaming_profiler::Batch<streaming_profiler::RecordType::Zones>& batch) {
                dropped_bytes_ += batch.dropped_bytes();
                dropped_bytes_ += batch.dropped_bytes();
                for (const streaming_profiler::Zone& zone : batch.zones()) {
                    const std::string_view name = zone.site().name;
                    if (name == "PP_TX" || name == "PP_RX") {
                        CoreZones& zones = by_run_[zone.runtime_id()][{zone.core().chip_id, zone.core().logical}];
                        (name == "PP_TX" ? zones.tx : zones.rx)++;
                        if (zones.timed.size() < 2 * kTimedRounds) {
                            zones.timed.push_back({zone.start_device_cycles(), zone.start_time(), name == "PP_TX"});
                        }
                    }
                }
            },
            "pingpong")) {}
    Check(const Check&) = delete;
    Check& operator=(const Check&) = delete;

    // Needs the mesh opened with 2D fabric.
    void run(distributed::MeshDevice& mesh_device, std::string_view name) {
        Chips chips;
        for (const auto& coord : distributed::MeshCoordinateRange(mesh_device.shape())) {
            IDevice* device = mesh_device.get_device(coord);
            chips.emplace(
                device->id(),
                Chip{coord, device, tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device->id())});
        }
        const auto runtime_id = static_cast<uint32_t>(runs_.size() + 1);
        Run& result = runs_.emplace_back(
            Run{.name = std::string(name), .runtime_id = runtime_id, .pairs = find_pairs(mesh_device, chips)});
        TT_FATAL(!result.pairs.empty(), "no fabric-linked chip pairs");
        std::vector<FlagCore> flag_cores;
        for (const Pair& pair : result.pairs) {
            flag_cores.push_back({chips.at(pair.chip_a).device, pair.core_a});
            flag_cores.push_back({chips.at(pair.chip_b).device, pair.core_b});
        }
    }
    const WorkerL1 worker_l1 = reserve_worker_l1(mesh_device);
    distributed::MeshWorkload workload = build_workload(mesh_device, worker_l1, result.pairs, chips, runtime_id);
    distributed::MeshCommandQueue& cq = mesh_device.mesh_command_queue();
    const auto start = Clock::now();
    do {
        clear_flags(worker_l1, flag_cores);
        run_once(cq, workload);
        result.passes++;
    } while (seconds_since(start) < kSeconds);
    std::printf(
        "[pingpong] %zu linked pairs on %zu chips, %u passes of %u rounds\n",
        result.pairs.size(),
        chips.size(),
        result.passes,
        kRounds);
    return result;
}

// Counts a PP_TX or PP_RX zone, and keeps it for the timeline check if it is in the first kTimedRounds rounds.
void add_zone(CoreZones& zones, const streaming_profiler::Zone& zone, bool tx) {
    (tx ? zones.tx : zones.rx)++;
    if (zones.timed.size() < 2 * kTimedRounds) {
        zones.timed.push_back({zone.start_device_cycles(), zone.start_time(), tx});
    }
}

bool verify(const Run& result, ZonesByCore& by_core) {
    const uint32_t failures =
        count_missing(result.pairs, by_core, size_t{result.passes} * kRounds) + check_timeline(result.pairs, by_core);
    return failures == 0;
}
}  // namespace

}  // namespace pingpong

// Runs the FF1 matmul and SDPA di/dt ops (tests/didt/test_ff1_matmul.py's "all and without_gelu" and test_sdpa_op.py's
// "all and bf16_HiFi2"), whose throttling moves AICLK, on one mesh with 2D fabric. The multicast and ping-pong checks
// run before, between and after them, and every run of each must pass.
namespace didt {

namespace {
constexpr uint32_t kFf1Iterations = 3000;
constexpr uint32_t kSdpaIterations = 500;

ttnn::Tensor random_normal(
    distributed::MeshDevice& mesh, const ttnn::Shape& shape, DataType dtype, std::mt19937& generator) {
    std::normal_distribution<float> normal;
    std::vector<float> values(shape.volume());
    std::ranges::generate(values, [&] { return normal(generator); });
    const TensorSpec spec(shape, TensorLayout(dtype, PageConfig(Layout::TILE), ttnn::DRAM_MEMORY_CONFIG));
    return ttnn::Tensor::from_vector(std::move(values), spec, &mesh);
}

template <typename Op>
void loop(distributed::MeshDevice& mesh, uint32_t iterations, const Op& op) {
    for (uint32_t i = 0; i < iterations; i++) {
        ttnn::Tensor out = op();
        distributed::Synchronize(mesh, std::nullopt);
        out.deallocate(/*force=*/true);
    }
}

void ff1_matmul(distributed::MeshDevice& mesh) {
    constexpr uint32_t kPerCoreM = 4, kPerCoreN = 72, kShardWidth = 576;
    const CoreCoord grid = mesh.compute_with_storage_grid_size();
    std::mt19937 generator(1234);
    const ttnn::MemoryConfig in0_config(
        TensorMemoryLayout::BLOCK_SHARDED,
        BufferType::L1,
        ShardSpec(
            CoreRangeSet(worker_grid(mesh)),
            {constants::TILE_HEIGHT * kPerCoreM, kShardWidth},
            ShardOrientation::ROW_MAJOR));
    const ttnn::Tensor in0 = ttnn::to_memory_config(
        random_normal(
            mesh,
            ttnn::Shape({1, 1, constants::TILE_HEIGHT * kPerCoreM * grid.y, kShardWidth * grid.x}),
            DataType::BFLOAT16,
            generator),
        in0_config);
    const ttnn::Tensor in1 = random_normal(
        mesh,
        ttnn::Shape({1, 1, kShardWidth * grid.x, constants::TILE_WIDTH * kPerCoreN * grid.x}),
        DataType::BFLOAT8_B,
        generator);
    const ttnn::operations::matmul::MatmulProgramConfig program_config =
        ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig{
            .compute_with_storage_grid_size = grid,
            .in0_block_w = 3,
            .out_subblock_h = 1,
            .out_subblock_w = 8,
            .out_block_h = kPerCoreM,
            .out_block_w = kPerCoreN,
            .per_core_M = kPerCoreM,
            .per_core_N = kPerCoreN,
            .transpose_mcast = false};
    const ttnn::BlackholeComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::LoFi,
        .math_approx_mode = false,
        .fp32_dest_acc_en = false,
        .packer_l1_acc = true};
    loop(mesh, kFf1Iterations, [&] {
        return ttnn::matmul(
            in0,
            in1,
            /*transpose_a=*/false,
            /*transpose_b=*/false,
            ttnn::L1_BLOCK_SHARDED_MEMORY_CONFIG,
            DataType::BFLOAT16,
            program_config,
            /*activation=*/std::nullopt,
            compute_config);
    });
}

// Runs the multicast and ping-pong checks once per phase. Records reach the host by the capture's end, so verify() runs
// after the mesh has closed.
class Checks {
public:
    Checks() :
        callback_(streaming_profiler::RegisterCallback(
            [this](const streaming_profiler::Batch<streaming_profiler::Zone>& batch) {
                dropped_bytes_ += batch.dropped_bytes();
                for (const streaming_profiler::Zone& zone : batch.records<streaming_profiler::Zone>()) {
                    const std::string_view name = zone.site().name;
                    const streaming_profiler::Core core = zone.core();
                    if (name == "MC_RX") {
                        arrivals_[zone.runtime_id()].push_back(multicast::Arrival{
                            .tsc = zone.start_tsc(),
                            .cycles = static_cast<int64_t>(zone.start_device_cycles()),
                            .chip = static_cast<uint16_t>(core.chip_id),
                            .logical_x = static_cast<uint8_t>(core.logical.x),
                            .logical_y = static_cast<uint8_t>(core.logical.y),
                            .physical_x = static_cast<uint8_t>(core.physical.x),
                            .physical_y = static_cast<uint8_t>(core.physical.y)});
                    } else if (name == "PP_TX" || name == "PP_RX") {
                        // A batch's zones come a core at a time, so the last core's counters usually serve.
                        const auto key = std::tuple(zone.runtime_id(), core.chip_id, core.logical);
                        if (last_zones_ == nullptr || key != last_key_) {
                            last_key_ = key;
                            last_zones_ = &zones_[zone.runtime_id()][{core.chip_id, core.logical}];
                        }
                        pingpong::add_zone(*last_zones_, zone, name == "PP_TX");
                    }
                }
            },
            "didt")) {}
    Checks(const Checks&) = delete;
    Checks& operator=(const Checks&) = delete;

    // Phase k's programs carry runtime id k + 1.
    void run(distributed::MeshDevice& mesh, std::string_view name) {
        const auto runtime_id = static_cast<uint32_t>(phases_.size() + 1);
        cores_ = worker_grid(mesh);
        num_chips_ = mesh.get_devices().size();
        multicast::run(mesh, runtime_id);
        phases_.push_back({std::string(name), pingpong::run(mesh, runtime_id)});
        std::fflush(stdout);
    }

    bool verify() {
        callback_.reset();
        size_t failed = 0;
        for (size_t k = 0; k < phases_.size(); k++) {
            const Phase& phase = phases_[k];
            std::printf("[didt] checks %s\n", phase.name.c_str());
            const bool multicast_pass = multicast::verify(arrivals_[k + 1], cores_, num_chips_);
            const bool pingpong_pass = pingpong::verify(phase.pingpong, zones_[k + 1]);
            std::printf(
                "[didt] checks %s: multicast %s, ping-pong %s\n",
                phase.name.c_str(),
                multicast_pass ? "PASS" : "FAIL",
                pingpong_pass ? "PASS" : "FAIL");
            failed += multicast_pass && pingpong_pass ? 0 : 1;
        }
        const bool pass = failed == 0 && dropped_bytes_ == 0;
        std::printf(
            "[didt] %s: %zu of %zu phases failed, %llu bytes dropped\n",
            pass ? "PASS" : "FAIL",
            failed,
            phases_.size(),
            static_cast<unsigned long long>(dropped_bytes_));
        return pass;
    }

private:
    struct Phase {
        std::string name;
        pingpong::Run pingpong;
    };
    std::vector<Phase> phases_;
    CoreRange cores_{CoreCoord{0, 0}};
    size_t num_chips_ = 0;
    std::map<uint32_t, std::vector<multicast::Arrival>> arrivals_;
    std::map<uint32_t, pingpong::ZonesByCore> zones_;
    std::tuple<uint32_t, ChipId, CoreCoord> last_key_;
    pingpong::CoreZones* last_zones_ = nullptr;
    uint64_t dropped_bytes_ = 0;
    streaming_profiler::Callback callback_;
};

void sdpa(distributed::MeshDevice& mesh) {
    const ttnn::Shape shape({1, 10, 9472, 128});
    std::mt19937 generator(1234);
    const ttnn::Tensor q = random_normal(mesh, shape, DataType::BFLOAT16, generator);
    const ttnn::Tensor k = random_normal(mesh, shape, DataType::BFLOAT16, generator);
    const ttnn::Tensor v = random_normal(mesh, shape, DataType::BFLOAT16, generator);
    const ttnn::operations::transformer::SDPAProgramConfig program_config{
        .compute_with_storage_grid_size = mesh.compute_with_storage_grid_size(),
        .q_chunk_size = 256,
        .k_chunk_size = 256,
        .exp_approx_mode = false};
    const ttnn::DeviceComputeKernelConfig compute_config = ttnn::init_device_compute_kernel_config(
        mesh.arch(),
        std::nullopt,
        MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false);
    loop(mesh, kSdpaIterations, [&] {
        return ttnn::transformer::scaled_dot_product_attention(
            q,
            k,
            v,
            /*attn_mask=*/std::nullopt,
            /*is_causal=*/false,
            /*scale=*/std::nullopt,
            /*sliding_window_size=*/std::nullopt,
            /*memory_config=*/std::nullopt,
            program_config,
            compute_config);
    });
}
}  // namespace

int run() {
    Checks checks;
    tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_2D);
    auto mesh_device = open_system_mesh();
    checks.run(*mesh_device, "before the ops");
    ff1_matmul(*mesh_device);
    std::printf("[didt] FF1 matmul: %u iterations\n", kFf1Iterations);
    checks.run(*mesh_device, "after the FF1 matmul");
    sdpa(*mesh_device);
    std::printf("[didt] SDPA: %u iterations\n", kSdpaIterations);
    checks.run(*mesh_device, "after SDPA");
    mesh_device->close();
    return checks.verify() ? 0 : 1;
}

}  // namespace didt

int main(int argc, char** argv) {
    const std::string_view workload = argc > 1 ? argv[1] : "";
    if (workload == "idle" && argc == 4 && std::string_view(argv[2]) == "--seconds") {
        return idle::run(std::strtod(argv[3], nullptr));
    }
    if (workload == "host_sync" && argc == 2) {
        return host_sync::run();
    }
    if (workload == "didt" && argc == 2) {
        return didt::run();
    }
    std::fprintf(stderr, "usage: %s idle --seconds S | host_sync | didt\n", argv[0]);
    return 2;
}
