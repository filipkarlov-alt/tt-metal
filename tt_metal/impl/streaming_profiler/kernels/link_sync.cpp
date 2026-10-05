// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "eth_l1_address_map.h"
#include "internal/ethernet/dataflow_api.h"
#include "tt_metal/impl/streaming_profiler/kernels/link_sync.hpp"

constexpr bool kTransmitter =
    static_cast<kernel_profiler::LinkSyncRole>(get_named_compile_time_arg_val("LINK_SYNC_ROLE")) ==
    kernel_profiler::LinkSyncRole::Transmitter;
static link_sync::LinkEnd<kTransmitter> g_link;
constexpr uint32_t kHandshake = eth_l1_mem::address_map::ERISC_L1_UNRESERVED_BASE;
constexpr uint32_t kHandshakeBytes = 16;
// The base firmware shares ERISC0 and only runs when a kernel context-switches to it. This kernel does so every this
// many loops, the fabric router's default_firmware_context_switch_interval.
constexpr uint32_t kLoopsPerContextSwitch = 10000;

void kernel_main() {
    g_link.start(get_named_compile_time_arg_val("LINK_SYNC_L1_ADDR"));
    if constexpr (kTransmitter) {
        eth_send_bytes(kHandshake, kHandshake, kHandshakeBytes);
        eth_wait_for_receiver_done();
    } else {
        eth_wait_for_bytes(kHandshakeBytes);
        eth_receiver_channel_done(0);
    }
    uint32_t until_switch = kLoopsPerContextSwitch;
    while (g_link.l1->ctl != kernel_profiler::LinkSyncCtl::Stop) {
        if (g_link.due()) {
            g_link.serve();
        }
        if (--until_switch == 0) {
            until_switch = kLoopsPerContextSwitch;
            run_routing();
        }
    }
    g_link.stop();
    g_link.l1->done = kernel_profiler::kResidentDoneWord;
}
