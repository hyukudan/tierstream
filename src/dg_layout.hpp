// Device-gated decode mode: what an open needs from the descriptor, the plan and the options (design rev 3.2 §1-§2),
// with every refusal. Backend-free, so the CPU tests check it; the CUDA context allocates exactly this.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "desc.hpp"
#include "plan.hpp"
#include "strix_stream/ss.h"

namespace ts {

struct DgLayout {
  std::vector<int> gated;           // descriptor layer of each gated ordinal
  std::vector<int> ordinal;         // per descriptor layer: its gated ordinal, -1 when not gated
  std::vector<uint32_t> unit_base;  // G + 1 (keys = unit_base[g] + unit)
  std::vector<uint64_t> part_bytes, part_offset;  // per part (offset inside the unit's bytes)
  uint64_t unit_bytes = 0, host_stride = 0;
  uint32_t keys = 0;
  uint32_t R = 0;  // device residents (the plan's tier R)
  uint32_t D = 0;  // spare slots = the plan's billed dynamic slots, and the boundary's free target
  uint32_t B = 0;  // bank size = dg_max_ids
  uint32_t P = 0;  // pinned units (tier P)
  uint32_t Q = 0;  // spare blocks the planner granted and billed
  // three-tier (dg_disk_tier): F = file-homed gated units (the plan's tier DISK), their per-key flags (keys order;
  // empty without the disk tier), S = staging buffers (pinned, one unit each); I = the inclusive share; disk_queue =
  // file reads in flight (1 = synchronous)
  uint32_t F = 0, S = 0, I = 0, disk_queue = 0;
  bool disk_tier = false, disk_prefetch = false;
  std::vector<uint8_t> disk;
  uint32_t persistent_slots = 0, slots = 0, guard_slot = 0, blocks = 0;
  std::vector<uint32_t> residents;  // keys of the tier-R units, in key order
  uint64_t need = 0, billed = 0;    // device bytes: the arena + tables + controls, and what the plan billed for them
  uint64_t mapped_bytes = 0;        // the mapped control area
  uint64_t device_ctl_bytes = 0;    // seq, abort, err_count, err_code, err_aux, any_miss[G], unit_base[G + 1]
  uint64_t bounce_bytes = 0;        // pageable, per reader: the covering aligned reads of unaligned parts
  uint32_t bounce_readers = 0;      // readers with their own bounce buffer: the context's thread + the disk readers
  uint64_t staging_bytes = 0;       // pinned: one resident's bytes on their way to the arena at open
  bool load_serial = false;
  // scored eviction (policy SCORED): the resolved weights, the profile's identity (empty without a profile)
  bool scored = false;
  uint32_t w_trans = 700, w_freq = 200, w_rec = 100;
  std::string score_identity;
  uint32_t load_readers = 0, load_buffers = 0;
  uint64_t load_buffer_bytes = 0;  // additional temporary pinned ring, admitted at open
  // DG_LOAD_OVERLAP_ALLOC (pipeline only, default on; 0 / false / off restores the serial order, as DG_LOAD_SERIAL
  // takes 0 / false / off or 1 / true / on): the ring first, then the pinned blocks on a helper thread while the
  // loader reads the residents
  bool load_overlap_alloc = false;
  // DG_UNIT_SUBREAD_KIB: the sub-read size of one unit's in-place parts (dg_read_unit; 0 = one request per part)
  uint64_t unit_subread = 0;
  // Exactly what the context allocates, so admission bills what it allocates.
  // seq, abort, err_count, err_code, err_aux, any_miss[G], unit_base[G + 1], absent_err
  static uint64_t ctl_words(uint32_t gated) { return 5ull + gated + (uint64_t(gated) + 1) + 1; }
};

// Bytes of the mapped control area for G gated layers (64-byte aligned arrays).
uint64_t dg_mapped_bytes(uint32_t gated);
// Throws ts::Error naming the refused rule. Checks every v1 scope rule except the backend (the context's).
DgLayout dg_layout(const Desc& d, const Plan& p, const ss_options& o);

// The device's own error word (err_code, err_aux; dg_proto.hpp) as text, empty for kDgOk.
std::string dg_device_error(uint32_t code, uint32_t aux);
// A failed token's message: the device's error word first (when it holds one), then what the host saw (the end edge,
// the service's error), then the poisoning.
std::string dg_token_failure(uint32_t seq, const std::string& device, const std::string& host);
// The CUDA driver's module loading mode (ss_dg_stats.cuda_module_loading: 0 unknown, 1 eager, 2 lazy) by name, and the
// open's notice for it (empty unless lazy with gated layers).
const char* dg_module_loading_name(uint32_t mode);
std::string dg_module_loading_notice(uint32_t mode, size_t gated);

}  // namespace ts
