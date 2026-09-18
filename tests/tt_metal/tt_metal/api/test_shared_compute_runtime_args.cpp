// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/experimental/mock_device/mock_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "device_fixture.hpp"
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"
#include "impl/program/dispatch.hpp"
#include "impl/program/program_impl.hpp"
#include "llrt/rtoptions.hpp"

namespace tt::tt_metal::shared_compute_runtime_test {

const CoreCoord core0{0, 0};
const CoreCoord core1{1, 0};

KernelDescriptor compute_descriptor(std::optional<ComputeProcessor> processor = std::nullopt) {
    KernelDescriptor kernel;
    kernel.kernel_source = "void kernel_main() {}";
    kernel.source_type = KernelDescriptor::SourceType::SOURCE_CODE;
    kernel.core_ranges = CoreRangeSet(CoreRange(core0, core1));
    ComputeConfigDescriptor config;
    config.processor = processor;
    kernel.config = config;
    return kernel;
}

// Returns owner as an UNPACK kernel followed by MATH and PACK kernels that borrow its runtime arguments.
ProgramDescriptor split_per_trisc(const KernelDescriptor& owner) {
    ProgramDescriptor descriptor{.kernels = {owner}};
    std::get<ComputeConfigDescriptor>(descriptor.kernels[0].config).processor = ComputeProcessor::UNPACK;
    for (auto processor : {ComputeProcessor::MATH, ComputeProcessor::PACK}) {
        auto borrower = compute_descriptor(processor);
        borrower.core_ranges = owner.core_ranges;
        borrower.runtime_args_owner = 0;
        descriptor.kernels.push_back(std::move(borrower));
    }
    return descriptor;
}

uint32_t tensix_index() {
    return MetalContext::instance().hal().get_programmable_core_type_index(HalProgrammableCoreType::TENSIX);
}

std::pair<uint32_t, uint32_t> pack_runtime_args(Program& program) {
    auto& kernels = program.impl().get_kernels(tensix_index());
    auto& groups = program.impl().get_kernel_groups(tensix_index());
    const auto& context = MetalContext::instance();
    constexpr uint32_t base_offset = 32;
    const auto unique = program_dispatch::configure_rta_offsets_for_kernel_groups(
        context, tensix_index(), kernels, groups, base_offset);
    const auto common = program_dispatch::configure_crta_offsets_for_kernel_groups(
        context, tensix_index(), kernels, groups, base_offset + unique);
    return {unique, common};
}

using DescriptorEdit = std::pair<std::string, std::function<void(ProgramDescriptor&)>>;

void expect_each_edit_rejected(const ProgramDescriptor& valid, const std::vector<DescriptorEdit>& edits) {
    for (const auto& [name, edit] : edits) {
        SCOPED_TRACE(name);
        auto invalid = valid;
        edit(invalid);
        EXPECT_THROW({ Program program(invalid); }, std::runtime_error);
    }
}

// Runs with Watcher off and on: Watcher prefixes each runtime-argument vector with its count.
class SharedComputeRuntimeArgs : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        experimental::configure_mock_mode(tt::ARCH::BLACKHOLE, 1);
        auto& options = MetalContext::instance().rtoptions();
        previous_watcher_ = options.get_watcher_enabled();
        previous_assert_disabled_ = options.watcher_assert_disabled();
        options.set_watcher_enabled(GetParam());
        options.enable_watcher_assert();
    }

    void TearDown() override {
        auto& options = MetalContext::instance().rtoptions();
        options.set_watcher_enabled(previous_watcher_);
        if (previous_assert_disabled_) {
            options.disable_watcher_assert();
        }
        experimental::disable_mock_mode();
    }

private:
    bool previous_watcher_ = false;
    bool previous_assert_disabled_ = false;
};

TEST_P(SharedComputeRuntimeArgs, CPU_SplitKernelPacksRuntimeArgsLikeUnsplitKernel) {
    for (const auto& [unique_count, common_count] :
         std::vector<std::pair<uint32_t, uint32_t>>{{0, 0}, {8, 1}, {84, 17}}) {
        SCOPED_TRACE(::testing::Message() << "unique=" << unique_count << " common=" << common_count);
        auto owner = compute_descriptor();
        if (unique_count) {
            owner.runtime_args = {
                {core0, std::vector<uint32_t>(unique_count, 7)}, {core1, std::vector<uint32_t>(unique_count / 2, 8)}};
        }
        owner.common_runtime_args = std::vector<uint32_t>(common_count, 9);
        auto split_descriptor = split_per_trisc(owner);
        ProgramDescriptor unsplit_descriptor{.kernels = {owner}};

        // The same compute owner participates in two kernel groups with different preceding DM payloads.
        auto reader = compute_descriptor();
        reader.core_ranges = CoreRangeSet(CoreRange(core1));
        reader.config = ReaderConfigDescriptor{};
        reader.runtime_args = {{core1, {1, 2, 3, 4, 5}}};
        reader.common_runtime_args = {6};
        unsplit_descriptor.kernels.push_back(reader);
        split_descriptor.kernels.push_back(reader);
        Program unsplit(unsplit_descriptor);
        Program split(split_descriptor);
        EXPECT_EQ(pack_runtime_args(split), pack_runtime_args(unsplit));

        const auto& groups = split.impl().get_kernel_groups(tensix_index());
        ASSERT_EQ(groups.size(), 2);
        for (const auto& group : groups) {
            auto* baseline =
                unsplit.impl().kernels_on_core(group->core_ranges.ranges().begin()->start_coord, tensix_index());
            ASSERT_NE(baseline, nullptr);
            EXPECT_EQ(group->total_rta_size, baseline->total_rta_size);
            auto offsets = group->launch_msg.view().kernel_config().rta_offset();
            auto baseline_offsets = baseline->launch_msg.view().kernel_config().rta_offset();
            for (uint32_t index = 0; index < group->kernel_ids.size(); ++index) {
                auto kernel = split.impl().get_kernel(group->kernel_ids[index]);
                if (kernel->runtime_args_owner()) {
                    EXPECT_EQ(group->rta_sizes[index], 0);
                    EXPECT_EQ(group->crta_sizes[index], 0);
                }
                for (int binary = 0; binary < kernel->expected_num_binaries(); ++binary) {
                    for (auto processor : kernel->get_processor_indices_for_binary(binary)) {
                        EXPECT_EQ(offsets[processor].rta_offset(), baseline_offsets[processor].rta_offset());
                        EXPECT_EQ(offsets[processor].crta_offset(), baseline_offsets[processor].crta_offset());
                    }
                }
            }
        }
    }
}

TEST_P(SharedComputeRuntimeArgs, CPU_BorrowersAliasOwnerStorageAndSchema) {
    auto owner = compute_descriptor();
    owner.runtime_args = {{core0, {10}}, {core1, {20}}};
    owner.common_runtime_args = {30};
    owner.blaze_named_args.named_per_core_runtime_args = {{"op.scalar", {{core0, 11}, {core1, 21}}}};
    Program program(split_per_trisc(owner));
    auto owner_kernel = program.impl().get_kernel(0);
    for (KernelHandle borrower : {1, 2}) {
        auto kernel = program.impl().get_kernel(borrower);
        EXPECT_EQ(kernel->runtime_args_owner(), owner_kernel);
        EXPECT_EQ(&GetRuntimeArgs(program, borrower, core0), &GetRuntimeArgs(program, 0, core0));
        EXPECT_EQ(&GetRuntimeArgs(program, borrower), &GetRuntimeArgs(program, 0));
        EXPECT_EQ(&GetCommonRuntimeArgs(program, borrower), &GetCommonRuntimeArgs(program, 0));
        EXPECT_EQ(
            kernel->named_runtime_arg_namespaces().at("op").size(),
            owner_kernel->named_runtime_arg_namespaces().at("op").size());
    }
    SetRuntimeArgs(program, 1, core0, {50, 51});
    EXPECT_EQ(GetRuntimeArgs(program, 2, core0)[0], 50);
    EXPECT_THROW(SetCommonRuntimeArgs(program, 1, {1}), std::runtime_error);
}

TEST_P(SharedComputeRuntimeArgs, CPU_RejectsOwnerThatIsNotAnEarlierUnpackKernelOnSameCores) {
    const auto valid = split_per_trisc(compute_descriptor());
    auto reordered_cores = valid;
    reordered_cores.kernels[1].core_ranges = CoreRangeSet(std::vector{CoreRange(core1), CoreRange(core0)});
    EXPECT_NO_THROW({ Program program(reordered_cores); });

    expect_each_edit_rejected(
        valid,
        {
            {"owner is itself", [](auto& d) { d.kernels[1].runtime_args_owner = 1; }},
            {"owner is later", [](auto& d) { d.kernels[1].runtime_args_owner = 2; }},
            {"owner is out of range", [](auto& d) { d.kernels[1].runtime_args_owner = 99; }},
            {"owner is MATH", [](auto& d) { d.kernels[2].runtime_args_owner = 1; }},
            {"fewer cores than owner", [](auto& d) { d.kernels[1].core_ranges = CoreRangeSet(CoreRange(core0)); }},
            {"owner is not compute", [](auto& d) { d.kernels[0].config = ReaderConfigDescriptor{}; }},
            {"borrower is not compute", [](auto& d) { d.kernels[1].config = ReaderConfigDescriptor{}; }},
            {"owner builds all TRISCs",
             [](auto& d) { std::get<ComputeConfigDescriptor>(d.kernels[0].config).processor.reset(); }},
        });
}

TEST_P(SharedComputeRuntimeArgs, CPU_RejectsBorrowerWithItsOwnRuntimeArgsOrBindings) {
    expect_each_edit_rejected(
        split_per_trisc(compute_descriptor()),
        {
            {"runtime_args", [](auto& d) { d.kernels[1].runtime_args = {{core0, {1}}}; }},
            {"common_runtime_args", [](auto& d) { d.kernels[1].common_runtime_args = {1}; }},
            {"named_common_runtime_args",
             [](auto& d) { d.kernels[1].blaze_named_args.named_common_runtime_args = {{"op.value", 1}}; }},
            {"named_per_core_runtime_args",
             [](auto& d) { d.kernels[1].blaze_named_args.named_per_core_runtime_args = {{"op.value", {{core0, 1}}}}; }},
            {"named_common_runtime_arg_arrays",
             [](auto& d) { d.kernels[1].blaze_named_args.named_common_runtime_arg_arrays = {{"op.array", {1, 2}}}; }},
            {"named_per_core_runtime_arg_arrays",
             [](auto& d) {
                 d.kernels[1].blaze_named_args.named_per_core_runtime_arg_arrays = {{"op.array", {{core0, {1, 2}}}}};
             }},
            {"buffer_bindings", [](auto& d) { d.kernels[1].buffer_bindings.push_back({core0, 0, nullptr}); }},
            {"common_buffer_bindings", [](auto& d) { d.kernels[1].common_buffer_bindings.push_back({0, nullptr}); }},
        });
}

TEST_P(SharedComputeRuntimeArgs, CPU_MergeRebasesRuntimeArgsOwner) {
    auto prefix = compute_descriptor();
    prefix.core_ranges = CoreRangeSet(CoreRange(CoreCoord{2, 0}));

    auto merged =
        merge_program_descriptors({ProgramDescriptor{.kernels = {prefix}}, split_per_trisc(compute_descriptor())});

    ASSERT_EQ(merged.kernels.size(), 4);
    EXPECT_EQ(merged.kernels[2].runtime_args_owner, 1);
    EXPECT_EQ(merged.kernels[3].runtime_args_owner, 1);
    Program program(merged);
    EXPECT_EQ(program.impl().get_kernel(2)->runtime_args_owner(), program.impl().get_kernel(1));
    EXPECT_EQ(program.impl().get_kernel(3)->runtime_args_owner(), program.impl().get_kernel(1));
}

TEST_P(SharedComputeRuntimeArgs, CPU_ProgramDescriptorHashIncludesRuntimeArgsOwner) {
    auto descriptor = split_per_trisc(compute_descriptor());
    const auto shared_hash = std::hash<ProgramDescriptor>{}(descriptor);

    descriptor.kernels[1].runtime_args_owner.reset();

    EXPECT_NE(shared_hash, std::hash<ProgramDescriptor>{}(descriptor));
}

INSTANTIATE_TEST_SUITE_P(Watcher, SharedComputeRuntimeArgs, ::testing::Bool());

using SharedComputeRuntimeArgsDevice = UnitMeshAnyDispatchFixture;

TEST_F(SharedComputeRuntimeArgsDevice, TensixAllTriscsSeeOwnerRuntimeArgsAcrossUpdates) {
    if (arch_ == tt::ARCH::QUASAR ||
        MetalContext::instance().get_cluster().get_target_device_type() == tt::TargetDevice::Emule) {
        GTEST_SKIP() << "Physical TRISC selection requires Wormhole or Blackhole silicon";
    }
    auto mesh_device = devices_.front();
    auto* device = mesh_device->get_devices()[0];
    const auto output_address = mesh_device->allocator()->get_base_allocator_addr(HalMemType::L1);
    auto owner = compute_descriptor(ComputeProcessor::UNPACK);
    owner.runtime_args = {{core0, {10}}, {core1, {20}}};
    owner.common_runtime_args = {30};
    owner.blaze_named_args.named_per_core_runtime_args = {{"probe.unique", {{core0, 11}, {core1, 21}}}};
    owner.blaze_named_args.named_common_runtime_args = {{"probe.common", 31}};
    auto descriptor = split_per_trisc(owner);
    for (auto& kernel : descriptor.kernels) {
        kernel.compile_time_args = {output_address};
        kernel.kernel_source = R"(
#include "api/compute/common.h"
#include "experimental/blaze_named_args.h"

void kernel_main() {
#if defined(TRISC_UNPACK)
    constexpr uint32_t processor = 0;
#elif defined(TRISC_MATH)
    constexpr uint32_t processor = 1;
#elif defined(TRISC_PACK)
    constexpr uint32_t processor = 2;
#endif
    auto* out = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_compile_time_arg_val(0));
    out[processor * 6 + 0] = get_arg_val<uint32_t>(0);
    out[processor * 6 + 1] = get_common_arg_val<uint32_t>(0);
    out[processor * 6 + 2] = blaze_rt_args::get<blaze_ct_args::probe::unique>();
    out[processor * 6 + 3] = blaze_rt_args::get<blaze_ct_args::probe::common>();
    out[processor * 6 + 4] = get_arg_addr(0);
    out[processor * 6 + 5] = get_common_arg_addr(0);
}
)";
    }
    const auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    distributed::MeshWorkload workload;
    workload.add_program(device_range, Program(descriptor));
    auto& program = workload.get_programs().at(device_range);

    auto run_and_check = [&](uint32_t increment) {
        SCOPED_TRACE(increment);
        std::vector<uint32_t> cleared(18, 0);
        for (auto core : {core0, core1}) {
            ASSERT_TRUE(detail::WriteToDeviceL1(device, core, output_address, cleared));
        }
        RunProgram(mesh_device, workload);
        for (auto core : {core0, core1}) {
            SCOPED_TRACE(::testing::Message() << "core x=" << core.x);
            std::vector<uint32_t> output;
            ASSERT_TRUE(detail::ReadFromDeviceL1(device, core, output_address, 18 * sizeof(uint32_t), output));
            ASSERT_EQ(output.size(), 18);
            const uint32_t unique = 10 + 10 * core.x + increment;
            for (uint32_t processor = 0; processor < 3; ++processor) {
                SCOPED_TRACE(processor);
                EXPECT_EQ(output[processor * 6 + 0], unique);
                EXPECT_EQ(output[processor * 6 + 1], 30 + increment);
                EXPECT_EQ(output[processor * 6 + 2], unique + 1);
                EXPECT_EQ(output[processor * 6 + 3], 31 + increment);
                EXPECT_EQ(output[processor * 6 + 4], output[4]);
                EXPECT_EQ(output[processor * 6 + 5], output[5]);
            }
        }
    };
    ASSERT_NO_FATAL_FAILURE(run_and_check(0));

    // Update the same Program after it has run, when fast dispatch has moved its runtime args into commands.
    auto& updated = descriptor.kernels[0];
    updated.runtime_args = {{core0, {110}}, {core1, {120}}};
    updated.common_runtime_args = {130};
    updated.blaze_named_args.named_per_core_runtime_args = {{"probe.unique", {{core0, 111}, {core1, 121}}}};
    updated.blaze_named_args.named_common_runtime_args = {{"probe.common", 131}};
    apply_descriptor_runtime_args(program, descriptor);
    ASSERT_NO_FATAL_FAILURE(run_and_check(100));

    for (auto core : {core0, core1}) {
        const uint32_t unique = 210 + 10 * core.x;
        SetRuntimeArgs(program, 1, core, {unique, unique + 1});
    }
    GetCommonRuntimeArgs(program, 2)[0] = 230;
    GetCommonRuntimeArgs(program, 2)[1] = 231;
    ASSERT_NO_FATAL_FAILURE(run_and_check(200));
}

}  // namespace tt::tt_metal::shared_compute_runtime_test
