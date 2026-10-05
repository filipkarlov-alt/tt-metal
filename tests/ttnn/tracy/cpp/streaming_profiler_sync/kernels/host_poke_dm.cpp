// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// For each round the host writes the round number to the flag, and this kernel records a HOST_RX zone when it sees it
// and then acks the round.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "workload_layout.hpp"

void kernel_main() {
    const uint32_t flag_addr = get_arg_val<uint32_t>(0);
    const uint32_t rounds = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    for (uint32_t round = 1; round <= rounds; round++) {
        if (!wait_for_round(flag, round)) {
            return;
        }
        {
            DeviceZoneScopedN("HOST_RX");
        }
        flag[kAckWord] = round;
    }
}
