// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

#include "llrt/tt_cluster.hpp"

namespace tt::test {
namespace {

// An empty simulator build directory, removed when the test ends.
class SimulatorBuildDir : public ::testing::Test {
protected:
    void SetUp() override {
        dir_ = std::filesystem::temp_directory_path() / ("tt_metal_sim_build_" + std::to_string(::getpid()));
        std::filesystem::create_directories(dir_);
    }
    void TearDown() override { std::filesystem::remove_all(dir_); }

    std::filesystem::path dir_;
};

TEST_F(SimulatorBuildDir, CPU_SingleChipBuildOpensChipZero) {
    EXPECT_EQ(Cluster::simulator_target_devices(dir_, tt::ARCH::WORMHOLE_B0), (std::unordered_set<ChipId>{0}));
}

TEST_F(SimulatorBuildDir, CPU_PartitionedBuildLeavesTheDevicesToUmd) {
    std::ofstream(dir_ / "ip_layout.yaml") << "access_points: []\n";
    EXPECT_TRUE(Cluster::simulator_target_devices(dir_, tt::ARCH::QUASAR).empty());
}

TEST_F(SimulatorBuildDir, CPU_SharedLibraryLeavesTheDevicesToUmd) {
    EXPECT_TRUE(Cluster::simulator_target_devices(dir_ / "libttsim.so", tt::ARCH::WORMHOLE_B0).empty());
    EXPECT_TRUE(Cluster::simulator_target_devices(dir_ / "libttsim.so", tt::ARCH::BLACKHOLE).empty());
}

TEST_F(SimulatorBuildDir, CPU_QuasarSharedLibraryOpensChipZero) {
    EXPECT_EQ(
        Cluster::simulator_target_devices(dir_ / "libttsim.so", tt::ARCH::QUASAR), (std::unordered_set<ChipId>{0}));
}

}  // namespace
}  // namespace tt::test
