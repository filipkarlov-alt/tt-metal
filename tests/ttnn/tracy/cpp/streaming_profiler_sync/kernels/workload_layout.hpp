// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// A source's value is the NoC it multicasts on.
enum class MulticastRole : std::uint32_t { Noc0Source, Noc1Source, Receiver };

// A ping-pong end's value is the parity of the rounds it sends first on.
enum class PingpongRole : std::uint32_t { EvenRoundSender, OddRoundSender };

// The words of a kernel's flag: the round it has reached and the host poke kernel's ack.
enum FlagWord : std::uint32_t { kRoundWord, kAckWord };

// The polls a kernel spends on a round before it gives up and exits.
constexpr std::uint32_t kSpinLimit = 1u << 26;

#if defined(KERNEL_BUILD)
#include "api/dataflow/dataflow_api.h"

FORCE_INLINE bool wait_for_round(volatile tt_l1_ptr std::uint32_t* flag, std::uint32_t round) {
    for (std::uint32_t polls = 0; flag[kRoundWord] < round; polls++) {
        invalidate_l1_cache();
        if (polls == kSpinLimit) {
            return false;
        }
    }
    return true;
}
#endif

// The multicast's round values sit this far apart because an L1-to-L1 NoC write needs source and destination congruent
// mod 16.
constexpr std::uint32_t kRoundValueStrideBytes = 16;
