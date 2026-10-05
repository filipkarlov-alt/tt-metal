// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Register access for the Blackhole Ethernet IEEE 1588 (PTP) hardware, and reads of the refclk and the ERISC wall clock.
//
// The MAC stamps frames with the tile's PTP time. On send, a TX queue armed for stamping writes each frame's send time
// into the frame itself, or into the TX timestamp FIFO. On receive, the RX classifier matches frames against TCAM rows,
// and a row's flow-table entry can record the frame's arrival time and the row's label in the RX timestamp FIFO. The TX
// header table sets the destination address outgoing frames carry, which is what a TCAM row on the far end matches.
// Restarting the PTP timer onto the refclk makes every stamp the refclk count times 20 ns.
//
// These blocks are shared by everything running on the tile. A second user would consume the first one's FIFO entries,
// and restarting the PTP timer changes the PTP time for every user.

#pragma once

#if !defined(ARCH_BLACKHOLE)
#error "eth_ptp.hpp is Blackhole only"
#endif

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <type_traits>

#include "internal/ethernet/tt_eth_ss_regs.h"
#include "internal/risc_attribs.h"

namespace eth_ptp {

constexpr uint32_t kRefclkHz = 50'000'000u;
constexpr uint32_t kNsPerRefclkTick = 1'000'000'000u / kRefclkHz;

constexpr uint64_t join_words(uint32_t hi, uint32_t lo) { return (static_cast<uint64_t>(hi) << 32) | lo; }

template <typename T>
constexpr uint32_t word_of(const T& value) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    return __builtin_bit_cast(uint32_t, value);
}

// A memory-mapped register whose bits are laid out by T.
template <typename T = uint32_t>
struct Reg {
    static_assert(sizeof(T) == sizeof(uint32_t));
    static_assert(std::has_unique_object_representations_v<T>, "a register layout names every bit");
    uint32_t addr;
    FORCE_INLINE T read() const { return __builtin_bit_cast(T, *reinterpret_cast<volatile uint32_t*>(addr)); }
    FORCE_INLINE void write(T value) const { *reinterpret_cast<volatile uint32_t*>(addr) = word_of(value); }
};

// ---------------------------------------------------------------------------------------------------------------------
// The refclk and the ERISC wall clock

// The PTP timer block counts refclk ticks since power-on. Reading the low word of any PTP timer counter latches its
// high word until that counter's low word is read again.
constexpr Reg<> kRefclkLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_LO};
constexpr Reg<> kRefclkHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CFR_HI};
// Reading the wall clock's low word copies its high word into WALL_CLOCK_1_AT, which holds it only until the next load
// (measured). WALL_CLOCK_1 is the live high word.
constexpr Reg<> kWallClockLo{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_0};
constexpr Reg<> kWallClockHi{ETH_RISC_REGS_START + ETH_RISC_WALL_CLOCK_1_AT};

FORCE_INLINE uint64_t read_latched(Reg<> lo_reg, Reg<> hi_reg) {
    const uint32_t lo = lo_reg.read();
    return join_words(hi_reg.read(), lo);
}
FORCE_INLINE uint64_t read_refclk() { return read_latched(kRefclkLo, kRefclkHi); }

// The wall clock and the refclk, read together.
struct Instant {
    uint64_t wall, refclk;
};
// Returns the wall clock and the refclk. It reads the wall clock's high word before and after its low word, and retries
// if they differ, since the low word may have wrapped in between.
FORCE_INLINE Instant read_instant() {
    while (true) {
        const uint32_t wall_hi_before = kWallClockHi.read();
        const uint32_t wall_lo = kWallClockLo.read();
        const uint32_t refclk_lo = kRefclkLo.read();
        const uint32_t wall_hi = kWallClockHi.read();
        const uint32_t refclk_hi = kRefclkHi.read();
        if (wall_hi == wall_hi_before) {
            return {join_words(wall_hi, wall_lo), join_words(refclk_hi, refclk_lo)};
        }
    }
}

// As the cores read it, the refclk count advances by 4 every 80 ns, because it crosses into the tile's clock domain
// through a handshake.
constexpr uint32_t kRefclkTicksPerUpdate = 4;

struct ClocksLo {
    uint32_t wall, refclk;
};
// Spins until the refclk's low word changes and returns the low words of the wall clock and the refclk read at the
// change, in that order.
FORCE_INLINE ClocksLo await_refclk_update() {
    const uint32_t before = kRefclkLo.read();
    while (true) {
        const ClocksLo now{kWallClockLo.read(), kRefclkLo.read()};
        if (now.refclk != before) {
            return now;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The PTP timer

struct PtpTimerCtrl {
    uint32_t enable : 1;
    uint32_t rsvd : 31;
};
// PTP nanoseconds added per refclk tick, in 8.16 fixed point.
struct PtpTickIncrement {
    uint32_t ns_fixed_point : 24;
    uint32_t rsvd : 8;
};
constexpr uint32_t kPtpTickIncrementFractionBits = 16;
struct PtpUpdateRequest {
    uint32_t request : 1;
    uint32_t rsvd : 31;
};
struct PtpUpdateStatus {
    uint32_t tick_increment_pending : 1;
    uint32_t time_pending : 1;
    uint32_t rsvd0 : 6;
    uint32_t tick_increment_ack : 1;
    uint32_t time_ack : 1;
    uint32_t rsvd1 : 6;
    uint32_t tick_increment_late : 1;
    uint32_t time_late : 1;
    uint32_t rsvd2 : 14;
};

constexpr Reg<PtpTimerCtrl> kPtpTimerCtrl{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_CTRL};
// When the refclk count reaches kPtpFutureRefclkLo/Hi, the timer loads kPtpFutureTickIncrement if
// kPtpUpdateTickIncrement is set, and kPtpFutureTimeSeconds/Ns if kPtpUpdateTime is set.
constexpr Reg<> kPtpFutureRefclkLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_LO};
constexpr Reg<> kPtpFutureRefclkHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_CFR_HI};
constexpr Reg<PtpTickIncrement> kPtpFutureTickIncrement{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_PTI};
constexpr Reg<> kPtpFutureTimeNs{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_LO};
constexpr Reg<> kPtpFutureTimeSeconds{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_FUTURE_TIMESTAMP_HI};
constexpr Reg<PtpUpdateRequest> kPtpUpdateTickIncrement{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_PTI};
constexpr Reg<PtpUpdateRequest> kPtpUpdateTime{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_TIMESTAMP};
constexpr Reg<PtpUpdateStatus> kPtpUpdateStatus{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_UPDATE_STAT};
// The PTP time in nanoseconds.
constexpr Reg<> kPtpNsLo{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_LO};
constexpr Reg<> kPtpNsHi{ETH_PTP_TIMER_REGS_START + ETH_PTP_TIMER_64NS_HI};

FORCE_INLINE uint64_t read_ptp_ns() { return read_latched(kPtpNsLo, kPtpNsHi); }

// The restart is scheduled 100 us ahead. That is far longer than its register writes take, so the scheduled time is
// still in the future when they finish.
constexpr uint32_t kPtpRestartLeadTicks = kRefclkHz / 10'000;

// Restarts the tile's PTP time so it reads the refclk count times kNsPerRefclkTick, and returns once the restart has
// taken effect. Every PTP stamp the tile takes afterwards is on that time.
__attribute__((noinline, cold)) inline void restart_ptp_timer() {
    kPtpTimerCtrl.write({.enable = 1});
    const uint64_t restart_at = read_refclk() + kPtpRestartLeadTicks;
    // The restart takes effect one tick after the scheduled time. Stamps taken before then still use the old PTP time.
    const uint64_t landed_at = restart_at + 1;
    // The timer loads a time as whole seconds and nanoseconds. They are split by shift-and-subtract because a 64-bit
    // division, or a 64-bit shift by a variable amount, would pull a library routine into the ERISC's code.
    uint64_t nanoseconds = landed_at * kNsPerRefclkTick;
    uint32_t seconds = 0;
    uint64_t chunk = uint64_t{1'000'000'000} << 31;
    for (uint32_t bit = 1u << 31; bit != 0; bit >>= 1, chunk >>= 1) {
        if (nanoseconds >= chunk) {
            nanoseconds -= chunk;
            seconds |= bit;
        }
    }
    kPtpFutureRefclkLo.write(static_cast<uint32_t>(restart_at));
    kPtpFutureRefclkHi.write(static_cast<uint32_t>(restart_at >> 32));
    kPtpFutureTickIncrement.write({.ns_fixed_point = kNsPerRefclkTick << kPtpTickIncrementFractionBits});
    kPtpFutureTimeNs.write(static_cast<uint32_t>(nanoseconds));
    kPtpFutureTimeSeconds.write(seconds);
    kPtpUpdateTickIncrement.write({.request = 1});
    kPtpUpdateTime.write({.request = 1});
    PtpUpdateStatus status;
    do {
        status = kPtpUpdateStatus.read();
    } while (!(status.tick_increment_ack && status.time_ack));
    kPtpUpdateTickIncrement.write({});
    kPtpUpdateTime.write({});
    while (read_refclk() <= landed_at) {
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// TX queue timestamping

enum class StampMode : uint32_t { None = 0, OneStepCorrection = 1, OneStepInFrame = 2, TwoStepFifo = 3 };

// The queue's offset setting for one-step stamps, in 2-byte units. The stamp lands 2 bytes past it (measured).
// Keepalives get stamped like any other packet. With the setting at 2, the stamp lands in their padding. At 40 it lands
// past their end, and the link loses every keepalive.
constexpr uint32_t kFrameStampOffsetSetting = 2;

struct TxqStampControl {
    StampMode mode : 3;
    uint32_t rsvd0 : 13;
    uint32_t in_frame_offset : 6;
    uint32_t rsvd1 : 10;
};

// The TX header row used by each kind of software-issued packet.
struct TxqHeaderSelect {
    uint32_t raw : 4;
    uint32_t reg_write : 4;
    uint32_t packet : 4;
    uint32_t rsvd : 20;
};

template <typename T = uint32_t>
constexpr Reg<T> txq_reg(uint32_t queue, uint32_t offset) {
    return {ETH_TXQ0_REGS_START + queue * ETH_TXQ_REGS_SIZE + offset};
}

constexpr Reg<TxqHeaderSelect> txq_header_select(uint32_t queue) {
    return txq_reg<TxqHeaderSelect>(queue, ETH_TXQ_TXPKT_CFG_SEL_SW);
}
// The queue reads its stamp control 24 to 32 cycles after a send command (measured), so arming it just after the
// command still stamps that packet. The queue reports idle before the MAC has latched the setting, though, so to stamp
// only selected packets, disarm after the last one's timestamp has arrived.
constexpr Reg<TxqStampControl> txq_stamp_control(uint32_t queue) {
    return txq_reg<TxqStampControl>(queue, ETH_TXQ_TIMESTAMP);
}
// The tag a two-step stamp carries into the TX timestamp FIFO. txq_arm_two_step() and tx_stamp_fifo::pop() use only its
// low word.
constexpr Reg<> txq_two_step_tag_lo(uint32_t queue) { return txq_reg(queue, ETH_TXQ_RX_TIMESTAMP_LO); }
constexpr Reg<> txq_two_step_tag_hi(uint32_t queue) { return txq_reg(queue, ETH_TXQ_RX_TIMESTAMP_HI); }
// The queue's word count is in 96-byte units on the wire, not the 16-byte words the ETH_TXQ_WORD_CNT description says
// (measured). A keepalive, or a packet with up to 64 bytes of payload, is one unit. A packet with up to 128 bytes is
// two.
constexpr Reg<> txq_word_count(uint32_t queue) { return txq_reg(queue, ETH_TXQ_WORD_CNT); }

// Arms the queue to put each packet's send time into the TX timestamp FIFO with `tag`. `tag` must not be
// tx_stamp_fifo::kEmptyTag, which is what an empty FIFO reads.
FORCE_INLINE void txq_arm_two_step(uint32_t queue, uint32_t tag) {
    txq_two_step_tag_lo(queue).write(tag);
    txq_stamp_control(queue).write(TxqStampControl{.mode = StampMode::TwoStepFifo});
}
FORCE_INLINE void txq_arm_in_frame(uint32_t queue) {
    txq_stamp_control(queue).write(
        TxqStampControl{.mode = StampMode::OneStepInFrame, .in_frame_offset = kFrameStampOffsetSetting});
}
FORCE_INLINE void txq_disarm(uint32_t queue) {
    txq_stamp_control(queue).write(TxqStampControl{.mode = StampMode::None});
}

// The start of a frame, where the MAC writes a one-step stamp. The stamp is 16 high bits followed by the 64-bit PTP
// time in ns, big-endian.
struct FrameStampSlot {
    uint32_t head[2];  // frame bytes the stamp leaves alone, then the stamp's 16 high bits
    uint32_t time_big_endian[2];
};
static_assert(offsetof(FrameStampSlot, time_big_endian) == 2 * kFrameStampOffsetSetting + 4);

FORCE_INLINE uint64_t frame_stamp_ns(const volatile FrameStampSlot& slot) {
    return join_words(__builtin_bswap32(slot.time_big_endian[0]), __builtin_bswap32(slot.time_big_endian[1]));
}
// A frame the MAC didn't stamp keeps whatever its slot held before.
FORCE_INLINE void clear_frame_stamp(volatile FrameStampSlot& slot) {
    slot.time_big_endian[0] = 0;
    slot.time_big_endian[1] = 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// The TX header table

constexpr uint32_t kTxHeaderRows = 10;

// The base firmware sends only on TX queue 0 and points TX queues 0 to 2 at TX header rows 0 to 2, leaving rows 3 to 9
// free. install() points TX queue Queue at header row Row, which it fills with a copy of the queue's current packet row
// and a new destination address. restore() puts back the queue's saved selection and is valid only after install().
template <uint32_t Queue, uint32_t Row>
class TxHeaderRow {
public:
    static_assert(Queue < NUM_ETH_QUEUES && Row >= NUM_ETH_QUEUES && Row < kTxHeaderRows);

    __attribute__((noinline, cold)) void install(uint64_t destination) {
        const auto field = [](uint32_t row, uint32_t offset) {
            return Reg<>{ETH_TXPKT_CFG_REGS_START + row * ETH_TXPKT_CFG_REGS_SIZE + offset};
        };
        select_before_ = txq_header_select(Queue).read();
        for (uint32_t offset :
             {ETH_TXPKT_CFG_INSERT_CTL,
              ETH_TXPKT_CFG_CUSTOM_HDR,
              ETH_TXPKT_CFG_MAC_SA_LO,
              ETH_TXPKT_CFG_MAC_SA_HI,
              ETH_TXPKT_CFG_ETHERTYPE,
              ETH_TXPKT_CFG_VLAN1,
              ETH_TXPKT_CFG_VLAN2}) {
            field(Row, offset).write(field(select_before_.packet, offset).read());
        }
        field(Row, ETH_TXPKT_CFG_MAC_DA_LO).write(static_cast<uint32_t>(destination));
        field(Row, ETH_TXPKT_CFG_MAC_DA_HI).write(static_cast<uint32_t>(destination >> 32));
        txq_header_select(Queue).write({.raw = Row, .reg_write = Row, .packet = Row});
    }
    __attribute__((noinline, cold)) void restore() const { txq_header_select(Queue).write(select_before_); }

private:
    TxqHeaderSelect select_before_{};
};

// ---------------------------------------------------------------------------------------------------------------------
// The RX classifier. Each TCAM row matches frames, and its flow-table row says what to do with them.

struct RxFlowActions {
    uint32_t rx_queue : 2;
    uint32_t drop : 1;
    uint32_t strip_headers : 1;
    uint32_t record_rx_time : 1;
    uint32_t prepend_software_metadata : 1;
    uint32_t prepend_hardware_metadata : 1;
    uint32_t rsvd : 25;
};

struct RxTcamRowMapping {
    uint32_t priority : 3;  // higher wins
    uint32_t rsvd0 : 13;
    uint32_t flow_table_row : 6;
    uint32_t rsvd1 : 10;
};
constexpr uint32_t kRxTcamTopPriority = 7;
constexpr uint32_t kRxTcamRows = 64;

struct RxTcamRowUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t enable : 1;
    uint32_t rsvd1 : 7;
    uint32_t write : 1;
    uint32_t rsvd2 : 14;
    uint32_t go : 1;
};

enum class RxTcamTupleType : uint32_t { NonIp = 0 };

// Each update_* bit makes the row take that field from the write registers. Fields whose bit is clear keep their
// values.
struct RxTcamUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t mask : 1;  // write the row's mask rather than its values
    uint32_t write : 1;
    uint32_t not_ip : 1;
    uint32_t rsvd1 : 5;
    uint32_t update_protocol : 1;
    uint32_t update_destination_port : 1;
    uint32_t update_source_port : 1;
    uint32_t update_destination : 1;
    uint32_t update_source : 1;
    uint32_t update_tuple_type : 1;
    uint32_t update_ethertype : 1;
    uint32_t update_l2_priority : 1;
    uint32_t rsvd2 : 7;
    uint32_t go : 1;
};

struct RxTcamNonIpAddrFlags {
    uint32_t augmented_destination : 4;
    uint32_t rsvd0 : 12;
    uint32_t augmented_source : 4;
    uint32_t rsvd1 : 12;
};

struct RxTcamEthertype {
    uint32_t value : 16;
    uint32_t augmented : 4;
    uint32_t rsvd : 12;
};

struct RxTcamPriority {
    uint32_t priority_code_point : 3;
    uint32_t rsvd : 29;
};

struct RxFlowTableLabel {
    uint32_t label : 5;
    uint32_t rsvd : 27;
};

struct RxFlowTableUpdate {
    uint32_t row : 6;
    uint32_t rsvd0 : 2;
    uint32_t write : 1;
    uint32_t rsvd1 : 22;
    uint32_t go : 1;
};

// The actions of flow-table row kRxTcamRows. That row applies to frames no TCAM row matches.
constexpr Reg<RxFlowActions> kRxNoMatchActions{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_NO_MATCH_ACTIONS};
constexpr Reg<RxTcamRowMapping> rx_tcam_row_mapping(uint32_t row) {
    return {ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_ROW_MAPPING + row * sizeof(uint32_t)};
}
constexpr Reg<RxTcamRowUpdate> kRxTcamRowUpdate{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_ROW_UPDATE};
constexpr Reg<RxTcamTupleType> kRxTcamTupleTypeWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_TUPLE_TYPE_WRITE};
constexpr Reg<> rx_tcam_source_write(uint32_t word) {
    return {ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_SA_WRITE + word * sizeof(uint32_t)};
}
constexpr Reg<> rx_tcam_destination_write(uint32_t word) {
    return {ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_DA_WRITE + word * sizeof(uint32_t)};
}
constexpr Reg<RxTcamNonIpAddrFlags> kRxTcamNonIpAddrFlagsWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_NON_IP_ADDR_FLAGS_WRITE};
constexpr Reg<RxTcamEthertype> kRxTcamEthertypeWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_ETHERTYPE_WRITE};
constexpr Reg<RxTcamPriority> kRxTcamPriorityWrite{
    ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_PRIORITY_WRITE};
constexpr Reg<RxTcamUpdate> kRxTcamUpdate{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_TCAM_UPDATE};
constexpr Reg<RxFlowTableLabel> kRxFlowTableLabel{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_LABELS};
constexpr Reg<RxFlowActions> kRxFlowTableActions{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_ACTIONS};
constexpr Reg<> kRxFlowTableVlan{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_VLAN};
constexpr Reg<> kRxFlowTableSoftwareMetadata{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_SW_METADATA};
constexpr Reg<RxFlowTableUpdate> kRxFlowTableUpdate{ETH_RX_CLASSIFIER_REGS_START + ETH_RX_CLASSIFIER_FTABLE_UPDATE};

// A non-IP TCAM pattern. Its address fields are four words wide to fit an IPv6 address.
struct RxTcamNonIpPattern {
    uint32_t source[4];
    uint32_t destination[4];
    RxTcamNonIpAddrFlags addr_flags;
    RxTcamEthertype ethertype;
    RxTcamPriority priority;
};
// A TCAM row's values and mask. A frame matches if it equals `values` on every bit where `mask` is 0.
struct RxTcamNonIpMatch {
    RxTcamNonIpPattern values, mask;
};
// Returns a match for frames whose destination address equals `destination` on the bits set in `compared`, regardless
// of the rest of the frame. The base firmware sends only broadcast, 01:00:.. multicast and 02:00:.. unicast frames.
// Addresses use the layout TxHeaderRow::install() writes, with the low word first and the high 16 bits second.
constexpr RxTcamNonIpMatch rx_tcam_match_destination(uint64_t destination, uint64_t compared) {
    const auto lo = [](uint64_t address) { return static_cast<uint32_t>(address); };
    const auto hi = [](uint64_t address) { return static_cast<uint32_t>(address >> 32); };
    return {
        .values = {.destination = {lo(destination & compared), hi(destination & compared)}},
        .mask = {
            .source = {~0u, ~0u, ~0u, ~0u},
            .destination = {~lo(compared), ~hi(compared), ~0u, ~0u},
            .addr_flags = {.augmented_destination = 0xF, .augmented_source = 0xF},
            .ethertype = {.value = 0xFFFF, .augmented = 0xF},
            .priority = {.priority_code_point = 7}}};
}

// ---------------------------------------------------------------------------------------------------------------------
// The RX timestamp FIFO. Each entry holds a received frame's arrival time and the label of the rule that recorded it.

struct RxStampLabel {
    uint32_t label : 5;
    uint32_t recorded_by_rule : 1;  // set on every entry a rule records (measured), so it isn't part of the label
    uint32_t rsvd : 25;
    uint32_t valid : 1;
};
struct RxStampStatus {
    uint32_t entries : 4;
    uint32_t rsvd0 : 12;
    uint32_t full : 1;
    uint32_t nearly_full : 1;
    uint32_t empty : 1;
    uint32_t rsvd1 : 13;
};
// The RxStampStatus register's write layout.
struct RxStampCommand {
    uint32_t rsvd : 30;
    uint32_t flush : 1;
    uint32_t pop : 1;
};

constexpr Reg<> kRxStampTimeLo{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_LOW};
constexpr Reg<> kRxStampTimeHi{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_HIGH};
constexpr Reg<RxStampLabel> kRxStampLabel{ETH_RX_TH_REGS_START + ETH_RX_TH_TS_LABEL};
constexpr Reg<RxStampStatus> kRxStampStatus{ETH_RX_TH_REGS_START + ETH_RX_TH_STATUS};
constexpr Reg<RxStampCommand> kRxStampCommand{ETH_RX_TH_REGS_START + ETH_RX_TH_STATUS};

// A frame's time is recorded when it starts to arrive, so by the time its data is visible in L1, its timestamp is
// already in the FIFO. A frame the link resends (Go-back-N) is recorded again, even though the receiver drops it as a
// duplicate.
namespace rx_stamp_fifo {
constexpr uint32_t kDepth = 16;
// Removes the head entry, which must exist, and returns its arrival time. kRxStampLabel shows the head's label until
// then. A pop needs a write of 1 followed by a write of 0 (measured).
FORCE_INLINE uint64_t pop() {
    const uint32_t lo = kRxStampTimeLo.read();
    const uint32_t hi = kRxStampTimeHi.read();
    kRxStampCommand.write({.pop = 1});
    kRxStampCommand.write({});
    return join_words(hi, lo);
}
template <uint32_t Count>
FORCE_INLINE bool holds_exactly() {
    static_assert(0 < Count && Count < kDepth, "a full FIFO reads as 0 entries");
    return kRxStampStatus.read().entries == Count;
}
// A single write of 1 flushes the FIFO, with no write of 0 after it (measured).
FORCE_INLINE void flush() { kRxStampCommand.write({.flush = 1}); }
}  // namespace rx_stamp_fifo

// Records the arrival time of every frame that `match` matches, labelled Label, using TCAM row Row and its flow-table
// row. Matched frames are otherwise handled like unmatched ones. Call remove() only after install().
template <uint32_t Row, uint32_t Label>
class RxStampRule {
public:
    static_assert(Row < kRxTcamRows);
    static_assert(Label < 32, "flow-table labels are five bits");

    // Also stops recording arrival times for unmatched frames, then flushes the FIFO.
    __attribute__((noinline, cold)) void install(const RxTcamNonIpMatch& match) {
        no_match_before_ = kRxNoMatchActions.read();
        RxFlowActions actions = no_match_before_;
        actions.record_rx_time = 0;
        kRxNoMatchActions.write(actions);
        rx_stamp_fifo::flush();
        write_tcam_row(match.values, false);
        write_tcam_row(match.mask, true);
        rx_tcam_row_mapping(Row).write({.priority = kRxTcamTopPriority, .flow_table_row = Row});
        actions.record_rx_time = 1;
        write_flow(actions, Label);
        kRxTcamRowUpdate.write({.row = Row, .enable = 1, .write = 1, .go = 1});
    }
    // Disables the row, clears its flow-table row, flushes the FIFO and restores the actions for unmatched frames.
    __attribute__((noinline, cold)) void remove() const {
        kRxTcamRowUpdate.write({.row = Row, .write = 1, .go = 1});
        write_flow({}, 0);
        rx_stamp_fifo::flush();
        kRxNoMatchActions.write(no_match_before_);
    }

private:
    FORCE_INLINE static void write_tcam_row(const RxTcamNonIpPattern& pattern, bool mask) {
        kRxTcamTupleTypeWrite.write(RxTcamTupleType::NonIp);
        kRxTcamEthertypeWrite.write(pattern.ethertype);
        kRxTcamPriorityWrite.write(pattern.priority);
        for (uint32_t word = 0; word < std::size(pattern.source); word++) {
            rx_tcam_source_write(word).write(pattern.source[word]);
            rx_tcam_destination_write(word).write(pattern.destination[word]);
        }
        kRxTcamNonIpAddrFlagsWrite.write(pattern.addr_flags);
        kRxTcamUpdate.write(
            {.row = Row,
             .mask = mask,
             .write = 1,
             .not_ip = 1,
             .update_destination = 1,
             .update_source = 1,
             .update_tuple_type = 1,
             .update_ethertype = 1,
             .update_l2_priority = 1,
             .go = 1});
    }
    FORCE_INLINE static void write_flow(RxFlowActions actions, uint32_t label) {
        kRxFlowTableActions.write(actions);
        kRxFlowTableVlan.write(0);
        kRxFlowTableLabel.write({.label = label});
        kRxFlowTableSoftwareMetadata.write(0);
        kRxFlowTableUpdate.write({.row = Row, .write = 1, .go = 1});
    }

    RxFlowActions no_match_before_{};
};

// ---------------------------------------------------------------------------------------------------------------------
// The MAC's send-time stamping, and the TX timestamp FIFO that two-step stamps go to

struct MacTxCfg {
    uint32_t rsvd0 : 1;
    uint32_t tx_enable : 1;
    uint32_t rsvd1 : 2;
    uint32_t stamp_fifo_every_packet : 1;  // when 0, only packets armed for a two-step stamp
    uint32_t rsvd2 : 6;
    uint32_t origin_timestamp_mode : 1;
    uint32_t rsvd3 : 20;
};
struct MacTxInterrupts {
    uint32_t rsvd0 : 2;
    uint32_t stamp_fifo_full : 1;
    uint32_t stamp_fifo_not_empty : 1;
    uint32_t stamp_offset_error : 1;
    uint32_t rsvd1 : 27;
};

// PTP nanoseconds added to every send-time stamp.
struct MacTxDelay {
    uint32_t ns : 16;
    uint32_t rsvd : 16;
};

constexpr Reg<MacTxCfg> kMacTxCfg{ETH_MAC_REGS_START + ETH_MAC_TX_CFG};
constexpr Reg<MacTxDelay> kMacTxDelay{ETH_MAC_REGS_START + ETH_MAC_TX_DELAY};
// Write 1 to a bit to clear it.
constexpr Reg<MacTxInterrupts> kMacTxInterrupts{ETH_MAC_REGS_START + ETH_MAC_TX_INT};
constexpr Reg<MacTxInterrupts> kMacTxInterruptsRaw{ETH_MAC_REGS_START + ETH_MAC_TX_INT_RAW};
constexpr Reg<> kMacTxStampFifoFullThreshold{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_FULL_THRESH};
// An entry's four words are the tag's low and high words followed by the time's low and high words (measured). The
// vendor guide gives a different order. Reading the first word pops the entry, and an empty FIFO reads all ones there.
constexpr Reg<> kMacTxStampTagLo{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_0};
constexpr Reg<> kMacTxStampTagHi{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_1};
constexpr Reg<> kMacTxStampTimeLo{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_2};
constexpr Reg<> kMacTxStampTimeHi{ETH_MAC_REGS_START + ETH_MAC_TS_FIFO_3};

namespace tx_stamp_fifo {
constexpr uint32_t kEmptyTag = 0xFFFFFFFFu;
struct Entry {
    uint32_t tag;
    uint64_t time_ns;
};
FORCE_INLINE std::optional<Entry> pop() {
    const uint32_t tag = kMacTxStampTagLo.read();
    if (tag == kEmptyTag) {
        return std::nullopt;
    }
    const uint32_t time_lo = kMacTxStampTimeLo.read();
    return Entry{.tag = tag, .time_ns = join_words(kMacTxStampTimeHi.read(), time_lo)};
}
FORCE_INLINE bool empty() { return !kMacTxInterruptsRaw.read().stamp_fifo_not_empty; }
__attribute__((noinline, cold)) inline void clear() {
    while (pop()) {
    }
}
}  // namespace tx_stamp_fifo

}  // namespace eth_ptp
