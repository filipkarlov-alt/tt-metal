// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/sync/tile_sync.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <map>
#include <optional>
#include <set>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "context/metal_context.hpp"
#include "hostdev/streaming_profiler_common.h"
#include "impl/device/device_impl.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/slow_dispatch.hpp"
#include "impl/streaming_profiler/device_programs.hpp"
#include "impl/streaming_profiler/sync/clock_map.hpp"
#include "impl/streaming_profiler/service.hpp"
#include "impl/streaming_profiler/sync/least_squares.hpp"
#include "llrt/hal.hpp"
#include "llrt/metal_soc_descriptor.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal::streaming_profiler {

namespace {

// A reader's readings of one partner over one NoC, as its kernel posts them in a TileSyncPartner. doubled_median holds
// only the clocks' low words, and doubled_offset() widens it to the full offset using whole_difference.
struct Reading {
    uint32_t partner, noc;
    // Set only on the both-NoC reader's extra reads: the index of its read of the same partner over the other NoC.
    std::optional<uint32_t> other_noc_read;
    int32_t doubled_median = 0;
    int64_t whole_difference = 0;
    int64_t doubled_offset() const {
        return 2 * whole_difference +
               static_cast<int32_t>(
                   static_cast<uint32_t>(doubled_median) - static_cast<uint32_t>(2 * whole_difference));
    }
};

// A mirrored pair's doubled offsets differ by, and a both-NoC pair's add up to, four times the tiles' offset.
constexpr int64_t kDoubledPairPerTick = 4;

constexpr double ticks_past(int64_t doubled, int64_t base) {
    return static_cast<double>(doubled - kDoubledPairPerTick * base) / kDoubledPairPerTick;
}

// A tile's offset from the ground tile, as a whole-tick base plus the fitted ticks from it. An eth or DRAM tile's clock
// keeps counting while the Tensix clocks halt between sessions, so its offset can reach about 1e14 ticks. A
// least-squares fit in doubles over values that large loses tenths of a tick to cancellation, so only the small
// remainder goes through the fit.
struct Placement {
    int64_t base = 0;
    double from_base = 0.0;
    int64_t offset() const { return base + round_nearest(from_base); }
};

struct Tile : CoreCoords {
    CoreType type;
    HalProgrammableCoreType core_type;
    uint32_t scratch = 0;
    uint64_t host_scratch = 0;
    std::vector<Reading> reads;
};

bool same_row_or_column(const CoreCoord& reader, const CoreCoord& partner) {
    return reader != partner && (reader.x == partner.x || reader.y == partner.y);
}
// NoC 0 runs towards higher raw coordinates, and NoC 1 towards lower ones.
bool upward(const CoreCoord& reader, const CoreCoord& partner) {
    return reader.x == partner.x ? partner.y > reader.y : partner.x > reader.x;
}

std::vector<Tile> plan_tiles(IDevice* device, ContextId context_id) {
    auto& mc = MetalContext::instance(context_id);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    const uint32_t chip = static_cast<uint32_t>(device->id());
    const auto& soc = cluster.get_soc_desc(chip);
    std::vector<Tile> tiles;
    const auto add = [&](CoreType type, HalProgrammableCoreType core_type, const CoreCoord& logical, uint32_t scratch) {
        tiles.push_back(Tile{
            {locate(cluster, chip, logical, type)},
            type,
            core_type,
            scratch,
            scratch + hal.get_l1_noc_offset(core_type)});
    };
    const auto add_unreserved = [&](CoreType type, HalProgrammableCoreType core_type, const CoreCoord& logical) {
        add(type, core_type, logical, hal.get_dev_addr(core_type, HalL1MemAddrType::UNRESERVED));
    };
    // A compute tile's profiler ring can't be used as its scratch, even if it is zeroed afterwards, because the first
    // frames depend on its contents.
    const uint32_t user_l1 = static_cast<uint32_t>(device->allocator()->get_base_allocator_addr(HalMemType::L1));
    TT_FATAL(
        hal.get_dev_size(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER) >=
            kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE + sizeof(kernel_profiler::TileSyncScratch),
        "streaming profiler: a profiler L1 region cannot hold the tile clock scratch");
    const uint32_t dispatch_scratch =
        static_cast<uint32_t>(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::PROFILER)) +
        kernel_profiler::PROFILER_L1_CONTROL_BUFFER_SIZE;
    const CoreCoord compute = device->compute_with_storage_grid_size();
    const CoreCoord grid = soc.get_grid_size(CoreType::TENSIX);
    for (uint32_t y = 0; y < grid.y; y++) {
        for (uint32_t x = 0; x < grid.x; x++) {
            const bool is_compute = x < compute.x && y < compute.y;
            add(CoreType::WORKER, HalProgrammableCoreType::TENSIX, {x, y}, is_compute ? user_l1 : dispatch_scratch);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::IDLE_ETH)) {
        for (const CoreCoord& logical : sorted_yx(device->get_inactive_ethernet_cores())) {
            add_unreserved(CoreType::ETH, HalProgrammableCoreType::IDLE_ETH, logical);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::ACTIVE_ETH)) {
        for (const CoreCoord& logical :
             sorted_yx(device->get_active_ethernet_cores(/*skip_reserved_tunnel_cores=*/false))) {
            add_unreserved(CoreType::ETH, HalProgrammableCoreType::ACTIVE_ETH, logical);
        }
    }
    if (hal.has_programmable_core_type(HalProgrammableCoreType::DRAM)) {
        // Firmware holds a DRAM view endpoint's NIU in NOC2AXI mode, where a read arriving on it goes to GDDR.
        for (const CoreCoord& logical : soc.get_metal_dram_cores(CoordSystem::LOGICAL)) {
            if (soc.get_dram_endpoint_noc_mask(soc.get_physical_dram_core_from_logical(logical)) == 0) {
                add_unreserved(CoreType::DRAM, HalProgrammableCoreType::DRAM, logical);
            }
        }
    }
    // One idle eth tile also reads every eth tile in its row over the other NoC. Around the ring, the two NoCs' hop
    // delays cancel, and the reader's own latency is the same for every target.
    const auto both_noc_reader = static_cast<uint32_t>(
        std::ranges::find(tiles, HalProgrammableCoreType::IDLE_ETH, &Tile::core_type) - tiles.begin());
    for (uint32_t r = 0; r < tiles.size(); r++) {
        Tile& reader = tiles[r];
        for (uint32_t partner = 0; partner < tiles.size(); partner++) {
            if (same_row_or_column(reader.phys, tiles[partner].phys)) {
                reader.reads.push_back(
                    Reading{.partner = partner, .noc = upward(reader.phys, tiles[partner].phys) ? 0u : 1u});
            }
        }
        if (r == both_noc_reader) {
            for (uint32_t i = 0, direct_reads = static_cast<uint32_t>(reader.reads.size()); i < direct_reads; i++) {
                const Reading& direct = reader.reads[i];
                if (tiles[direct.partner].type == CoreType::ETH) {
                    reader.reads.push_back(
                        Reading{.partner = direct.partner, .noc = direct.noc ^ 1u, .other_noc_read = i});
                }
            }
        }
        TT_FATAL(
            reader.reads.size() <= kernel_profiler::kTileSyncMaxPartners,
            "streaming profiler: tile ({},{}) has {} row and column partners, the table holds {}",
            reader.logical.x,
            reader.logical.y,
            reader.reads.size(),
            kernel_profiler::kTileSyncMaxPartners);
    }
    return tiles;
}

KernelHandle create_tile_kernel(Program& program, HalProgrammableCoreType core_type, const CoreRangeSet& cores) {
    const char* src = "tt_metal/impl/streaming_profiler/kernels/tile_sync.cpp";
    switch (core_type) {
        case HalProgrammableCoreType::TENSIX:
            return CreateKernel(
                program,
                src,
                cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::IDLE_ETH:
            return CreateKernel(
                program, src, cores, EthernetConfig{.eth_mode = Eth::IDLE, .noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::ACTIVE_ETH:
            return CreateKernel(program, src, cores, EthernetConfig{.noc = NOC::RISCV_0_default});
        case HalProgrammableCoreType::DRAM: return CreateKernel(program, src, cores, DramConfig{.noc = NOC::NOC_0});
        case HalProgrammableCoreType::DISPATCH:
        case HalProgrammableCoreType::COUNT: break;
    }
    TT_THROW("Unreachable");
}

// One chip's tiles, each running its tile kernel. If a later step throws, the kernels stay running, waiting on their go
// words.
struct Network {
    IDevice* device = nullptr;
    uint32_t chip = 0;
    std::vector<Tile> tiles;
    struct Kind {
        std::set<CoreRange> cores;
        std::optional<Program> program;
        KernelHandle kernel = 0;
    };
    std::map<HalProgrammableCoreType, Kind> kinds;
};

uint64_t table_of(const Tile& tile) { return tile.host_scratch + offsetof(kernel_profiler::TileSyncScratch, table); }

Network launch_network(IDevice* device, ContextId context_id) {
    auto& mc = MetalContext::instance(context_id);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    Network net{.device = device, .chip = static_cast<uint32_t>(device->id()), .tiles = plan_tiles(device, context_id)};
    for (const Tile& tile : net.tiles) {
        net.kinds[tile.core_type].cores.insert(CoreRange(tile.logical, tile.logical));
    }
    for (auto& [core_type, kind] : net.kinds) {
        kind.program = CreateProgram();
        kind.kernel = create_tile_kernel(*kind.program, core_type, CoreRangeSet(kind.cores));
    }
    for (const Tile& tile : net.tiles) {
        std::vector<uint32_t> args{tile.scratch, static_cast<uint32_t>(tile.reads.size())};
        for (const Reading& read : tile.reads) {
            const CoreCoord& partner = net.tiles[read.partner].virt;
            args.push_back(kernel_profiler::word_of(kernel_profiler::TileSyncRead{
                .x = static_cast<uint32_t>(partner.x), .y = static_cast<uint32_t>(partner.y), .noc = read.noc}));
        }
        const Network::Kind& kind = net.kinds[tile.core_type];
        SetRuntimeArgs(*kind.program, kind.kernel, tile.logical, args);
        zero_l1(cluster, net.chip, tile.virt, table_of(tile), offsetof(kernel_profiler::TileSyncTable, partner));
        // The firmware takes its ring position from the control vector at the session's first launch, which is this
        // one, so a stale tail would put it thousands of words ahead.
        zero_profiler_control(
            cluster, net.chip, tile.virt, hal.get_dev_noc_addr(tile.core_type, HalL1MemAddrType::PROFILER));
    }
    for (auto& [core_type, kind] : net.kinds) {
        launch_resident(device, *kind.program);
    }
    return net;
}

constexpr auto kTablePostTimeout = std::chrono::seconds(10);

kernel_profiler::TileSyncTable await_table(
    tt::Cluster& cluster, const Network& net, const Tile& tile, kernel_profiler::TileSyncReady ready) {
    kernel_profiler::TileSyncTable table{};
    const auto table_bytes = static_cast<uint32_t>(
        offsetof(kernel_profiler::TileSyncTable, partner) +
        tile.reads.size() * sizeof(kernel_profiler::TileSyncPartner));
    const auto deadline = std::chrono::steady_clock::now() + kTablePostTimeout;
    do {
        cluster.read_core(&table, table_bytes, tt_cxy_pair(net.chip, tile.virt), table_of(tile));
        TT_FATAL(
            table.ready == ready || std::chrono::steady_clock::now() < deadline,
            "streaming profiler: device {} tile {} did not post its tile clock table",
            net.chip,
            tile.virt.str());
    } while (table.ready != ready);
    return table;
}

void start_tile_reads(tt::Cluster& cluster, const Network& net, const Tile& tile) {
    await_table(cluster, net, tile, kernel_profiler::TileSyncReady::Up);
    const kernel_profiler::TileSyncGo go_measure = kernel_profiler::TileSyncGo::Measure;
    cluster.write_core(&go_measure, sizeof(go_measure), tt_cxy_pair(net.chip, tile.virt), table_of(tile));
}

void collect_tile_reads(tt::Cluster& cluster, const Network& net, Tile& tile) {
    const kernel_profiler::TileSyncTable table = await_table(cluster, net, tile, kernel_profiler::TileSyncReady::Done);
    for (size_t i = 0; i < tile.reads.size(); i++) {
        tile.reads[i].doubled_median = table.partner[i].doubled_median;
        tile.reads[i].whole_difference = table.partner[i].whole_difference;
    }
}

void retire_network(ContextId context_id, Network& net) {
    auto& mc = MetalContext::instance(context_id);
    auto& cluster = mc.get_cluster();
    const auto& hal = mc.hal();
    for (const Tile& tile : net.tiles) {
        const kernel_profiler::TileSyncGo go_exit = kernel_profiler::TileSyncGo::Exit;
        cluster.write_core(&go_exit, sizeof(go_exit), tt_cxy_pair(net.chip, tile.virt), table_of(tile));
    }
    for (auto& [core_type, kind] : net.kinds) {
        slow_dispatch::WaitProgramDone(*net.device, *kind.program);
    }
    // Fast dispatch's go signal reaches every core, and a host-launched kernel leaves its launch slot valid, so the
    // firmware's initial launch message is restored.
    for (const Tile& tile : net.tiles) {
        auto msg = hal.get_dev_msgs_factory(tile.core_type).create<dev_msgs::launch_msg_t>();
        cluster.write_core(
            msg.data(),
            static_cast<uint32_t>(msg.size()),
            tt_cxy_pair(net.chip, tile.virt),
            hal.get_dev_noc_addr(tile.core_type, HalL1MemAddrType::LAUNCH));
    }
}

bool is_active_eth(const Tile& tile) { return tile.core_type == HalProgrammableCoreType::ACTIVE_ETH; }

// plan_tiles lists the Tensix grid first, so tile 0 is the first Tensix tile. Every other tile is placed relative to
// it.
constexpr uint32_t kGroundTile = 0;

// Places every tile except the active eth ones from the mirrored pairs, relative to the first Tensix tile. Active eth
// tiles are left out because their reads leave the NIU about 6 cycles later than an idle eth tile's, which would bias a
// mirrored pair by a tick and a half.
std::vector<Placement> place_mirrored(const std::vector<Tile>& tiles, uint32_t chip) {
    const auto tile_count = static_cast<uint32_t>(tiles.size());
    std::vector<std::optional<size_t>> unknown_of(tile_count);
    std::vector<uint32_t> tile_of_unknown;
    for (uint32_t i = 0; i < tile_count; i++) {
        if (i != kGroundTile && !is_active_eth(tiles[i])) {
            unknown_of[i] = tile_of_unknown.size();
            tile_of_unknown.push_back(i);
        }
    }
    struct MirroredPair {
        uint32_t reader, partner;
        int64_t doubled_difference;
        std::optional<size_t> partner_unknown, reader_unknown;
    };
    std::vector<MirroredPair> pairs;
    for (uint32_t reader = 0; reader < tile_count; reader++) {
        for (const Reading& there : tiles[reader].reads) {
            if (there.other_noc_read || there.noc != 0 || is_active_eth(tiles[reader]) ||
                is_active_eth(tiles[there.partner])) {
                continue;
            }
            const Reading& back = *std::ranges::find(tiles[there.partner].reads, reader, &Reading::partner);
            pairs.push_back(
                {.reader = reader,
                 .partner = there.partner,
                 .doubled_difference = there.doubled_offset() - back.doubled_offset(),
                 .partner_unknown = unknown_of[there.partner],
                 .reader_unknown = unknown_of[reader]});
        }
    }
    std::vector<std::optional<int64_t>> base(tile_count);
    base[kGroundTile] = 0;
    for (bool grew = true; grew;) {
        grew = false;
        for (const MirroredPair& pair : pairs) {
            const int64_t whole = pair.doubled_difference / kDoubledPairPerTick;
            if (base[pair.reader] && !base[pair.partner]) {
                base[pair.partner] = *base[pair.reader] + whole;
                grew = true;
            } else if (base[pair.partner] && !base[pair.reader]) {
                base[pair.reader] = *base[pair.partner] - whole;
                grew = true;
            }
        }
    }
    for (const uint32_t t : tile_of_unknown) {
        TT_FATAL(
            base[t],
            "streaming profiler: device {} {} tile ({},{}) has no chain of mirrored pairs to the first Tensix tile",
            chip,
            tt::to_str(tiles[t].type),
            tiles[t].logical.x,
            tiles[t].logical.y);
    }
    const auto offset = [&](const MirroredPair& pair) {
        return ticks_past(pair.doubled_difference, *base[pair.partner] - *base[pair.reader]);
    };
    const std::vector<double> from_base = solve_potential(
        pairs, &MirroredPair::partner_unknown, &MirroredPair::reader_unknown, offset, tile_of_unknown.size());
    std::vector<Placement> placed(tile_count);
    for (size_t u = 0; u < tile_of_unknown.size(); u++) {
        placed[tile_of_unknown[u]] = Placement{.base = *base[tile_of_unknown[u]], .from_base = from_base[u]};
    }
    return placed;
}

// Places each active eth tile from its both-NoC reading, shifted onto the mirrored placement by the idle eth tiles that
// also have both-NoC readings.
void place_active_eth(const std::vector<Tile>& tiles, uint32_t chip, std::vector<Placement>& placed) {
    std::vector<std::optional<int64_t>> doubled_sum(tiles.size());
    double shift_sum = 0.0;
    size_t reference_count = 0;
    for (const Tile& reader : tiles) {
        for (const Reading& read : reader.reads) {
            if (!read.other_noc_read) {
                continue;
            }
            const int64_t sum = reader.reads[*read.other_noc_read].doubled_offset() + read.doubled_offset();
            doubled_sum[read.partner] = sum;
            if (!is_active_eth(tiles[read.partner])) {
                const Placement& reference = placed[read.partner];
                shift_sum += reference.from_base - ticks_past(sum, reference.base);
                reference_count++;
            }
        }
    }
    for (uint32_t t = 0; t < tiles.size(); t++) {
        if (!is_active_eth(tiles[t])) {
            continue;
        }
        TT_FATAL(
            reference_count != 0 && doubled_sum[t],
            "streaming profiler: device {} active eth tile ({},{}) has no both-NoC reading to place it from",
            chip,
            tiles[t].logical.x,
            tiles[t].logical.y);
        const int64_t base = *doubled_sum[t] / kDoubledPairPerTick;
        placed[t] = Placement{
            .base = base,
            .from_base = ticks_past(*doubled_sum[t], base) + shift_sum / static_cast<double>(reference_count)};
    }
}

}  // namespace

void measure_tile_clocks(std::span<Device* const> devices, ContextId context_id) {
    auto& mc = MetalContext::instance(context_id);
    if (!can_capture(mc)) {
        return;
    }
    auto& cluster = mc.get_cluster();
    std::vector<std::future<Network>> launches;
    for (Device* device : devices) {
        if (service().tile_clocks(context_id, static_cast<uint32_t>(device->id())) == nullptr) {
            launches.push_back(std::async(std::launch::async, launch_network, device, context_id));
        }
    }
    std::vector<Network> nets;
    size_t most_tiles = 0;
    for (auto& launch : launches) {
        nets.push_back(launch.get());
        most_tiles = std::max(most_tiles, nets.back().tiles.size());
    }
    // On each chip only one tile reads at a time. Chips have their own NoCs, so all chips run in parallel.
    for (size_t t = 0; t < most_tiles; t++) {
        for (const Network& net : nets) {
            if (t < net.tiles.size()) {
                start_tile_reads(cluster, net, net.tiles[t]);
            }
        }
        for (Network& net : nets) {
            if (t < net.tiles.size()) {
                collect_tile_reads(cluster, net, net.tiles[t]);
            }
        }
    }
    for (Network& net : nets) {
        retire_network(context_id, net);
        std::vector<Placement> placed = place_mirrored(net.tiles, net.chip);
        place_active_eth(net.tiles, net.chip, placed);
        TileClocks clocks;
        for (size_t i = 0; i < net.tiles.size(); i++) {
            clocks.emplace(std::pair(net.tiles[i].type, net.tiles[i].logical), placed[i].offset());
        }
        service().set_tile_clocks(context_id, net.chip, std::move(clocks));
    }
}

}  // namespace tt::tt_metal::streaming_profiler
