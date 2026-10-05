// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "impl/streaming_profiler/ops_csv.hpp"

#include <algorithm>
#include <cstdio>
#include <string>

#include <tt_stl/assert.hpp>

namespace tt::tt_metal::streaming_profiler {

using experimental::streaming_profiler::Processor;

OpsCsvConsumer::OpsCsvConsumer(const std::string& path) : file_(std::fopen(path.c_str(), "w")) {
    TT_FATAL(file_ != nullptr, "streaming profiler: cannot open {} for the ops CSV", path);
}

void OpsCsvConsumer::operator()(const Batch& batch) {
    for (const auto& zone : batch.zones()) {
        if (zone.runtime_id() == 0 || !zone.site().name.ends_with("-KERNEL")) {
            continue;
        }
        const experimental::streaming_profiler::Core core = zone.core();
        const bool eth = core.processor >= Processor::ERISC0;
        const uint32_t processor = static_cast<uint32_t>(core.processor);
        const uint32_t core_key =
            (static_cast<uint32_t>(core.physical.y) << 16) | static_cast<uint32_t>(core.physical.x);
        // The wrapper zone never nests, so the k-th one on a lane for a program is its k-th execution.
        uint32_t& completed = executions_seen_[{core.chip_id, core_key, processor, zone.runtime_id()}];
        OpAgg& op = ops_[{core.chip_id, zone.runtime_id(), completed}];
        completed++;
        if (!eth) {
            op.start_cycles = std::min(op.start_cycles, zone.start_device_cycles());
            op.end_cycles = std::max(op.end_cycles, zone.end_device_cycles());
        }
        const Time start = zone.start_time();
        const Time end = zone.end_time();
        op.last_kernel_start = std::max(op.last_kernel_start, start);
        for (Span* span : {&op.processors[processor], &op.cores[core_key]}) {
            span->start = std::min(span->start, start);
            span->end = std::max(span->end, end);
        }
    }
}

void OpsCsvConsumer::write_csv() {
    FILE* const out = file_.get();
    if (!header_written_) {
        header_written_ = true;
        std::fputs(
            "DEVICE ID,GLOBAL CALL COUNT,EXECUTION,CORE COUNT,DEVICE KERNEL START CYCLE,DEVICE KERNEL END CYCLE,"
            "DEVICE KERNEL DURATION [ns],DEVICE KERNEL DURATION DM START [ns],"
            "DEVICE KERNEL DURATION PER CORE MIN [ns],DEVICE KERNEL DURATION PER CORE MAX [ns],"
            "DEVICE KERNEL DURATION PER CORE AVG [ns],DEVICE KERNEL FIRST TO LAST START [ns],"
            "DEVICE BRISC KERNEL DURATION [ns],DEVICE NCRISC KERNEL DURATION [ns],"
            "DEVICE TRISC0 KERNEL DURATION [ns],DEVICE TRISC1 KERNEL DURATION [ns],"
            "DEVICE TRISC2 KERNEL DURATION [ns],DEVICE ERISC0 KERNEL DURATION [ns],DEVICE ERISC1 KERNEL DURATION "
            "[ns]\n",
            out);
    }
    const auto elapsed_ns = [](Time start, Time end) {
        return end > start ? std::chrono::duration<double, std::nano>(end - start).count() : 0.0;
    };
    for (const auto& [key, op] : ops_) {
        const auto& [chip, runtime_id, execution] = key;
        const auto processor = [&](Processor p) -> const Span& { return op.processors[static_cast<uint32_t>(p)]; };
        const auto processor_ns = [&](Processor p) { return elapsed_ns(processor(p).start, processor(p).end); };
        Time start = Time::max(), end = Time::min();
        for (const Span& span : op.processors) {
            start = std::min(start, span.start);
            end = std::max(end, span.end);
        }
        const Time dm_start = std::min(
            {processor(Processor::BRISC).start,
             processor(Processor::NCRISC).start,
             processor(Processor::ERISC0).start,
             processor(Processor::ERISC1).start});
        double core_min = 0.0, core_max = 0.0, core_sum = 0.0;
        uint32_t core_n = 0;
        for (const auto& [core, span] : op.cores) {
            const double core_ns = elapsed_ns(span.start, span.end);
            core_min = core_n == 0 ? core_ns : std::min(core_min, core_ns);
            core_max = std::max(core_max, core_ns);
            core_sum += core_ns;
            core_n++;
        }
        std::fprintf(out, "%u,%u,%u,%u,", chip, runtime_id, execution, core_n);
        // An op with no Tensix kernel zone leaves the CYCLE columns empty.
        if (op.start_cycles != UINT64_MAX) {
            std::fprintf(
                out,
                "%llu,%llu,",
                static_cast<unsigned long long>(op.start_cycles),
                static_cast<unsigned long long>(op.end_cycles));
        } else {
            std::fputs(",,", out);
        }
        std::fprintf(
            out,
            "%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,%.0f,",
            elapsed_ns(start, end),
            elapsed_ns(dm_start, end),
            core_min,
            core_max,
            core_sum / core_n,
            elapsed_ns(start, op.last_kernel_start),
            processor_ns(Processor::BRISC),
            processor_ns(Processor::NCRISC),
            processor_ns(Processor::TRISC0),
            processor_ns(Processor::TRISC1),
            processor_ns(Processor::TRISC2));
        // An ERISC with no kernel zone in the op leaves its column empty.
        for (Processor erisc : {Processor::ERISC0, Processor::ERISC1}) {
            if (processor(erisc).end != Time::min()) {
                std::fprintf(out, "%.0f", processor_ns(erisc));
            }
            std::fputc(erisc == Processor::ERISC0 ? ',' : '\n', out);
        }
    }
    std::fflush(out);
    ops_.clear();
}

}  // namespace tt::tt_metal::streaming_profiler
