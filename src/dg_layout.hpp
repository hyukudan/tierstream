// Device-gated decode mode: what an open needs from the descriptor, the plan and the options (design rev 3.2 §1-§2),
// with every refusal. Backend-free, so the CPU tests check it; the CUDA context allocates exactly this.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "desc.hpp"
#include "dg_proto.hpp"
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
  // file reads in flight (1 = synchronous; SS_DG_DISK_QUEUE_AUTO resolved, dg_resolve_disk_queue); disk_locations =
  // the most read locations of a gated source (dg_disk_locations; counted only for AUTO, else 0)
  uint32_t F = 0, S = 0, I = 0, disk_queue = 0, disk_locations = 0;
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
  // DG_PF_LEGACY (0 / false / off or 1 / true / on, default on until the new admission is measured): the prediction
  // admission before SS_DG_PF_ADMIT (no ahead skip of plain predictions, every prefetch read of a layer ended before
  // its ready, a full read queue ending the oldest reads); DG_PF_LEGACY=0 selects the new admission
  bool pf_legacy = true;
  // DG_UNIT_SUBREAD_KIB: the sub-read size of one unit's in-place parts (dg_read_unit; 0 = one request per part)
  uint64_t unit_subread = 0;
  // Exactly what the context allocates, so admission bills what it allocates.
  // seq, abort, err_count, err_code, err_aux, any_miss[G], unit_base[G + 1], absent_err
  // seq, abort, err_count, err_code, err_aux, any_miss[G], unit_base[G + 1], absent_err, then the layer tokens'
  // any_miss[kDgMaxParts]
  static uint64_t ctl_words(uint32_t gated) { return 5ull + gated + (uint64_t(gated) + 1) + 1 + kDgMaxParts; }
};

// Bytes of the mapped control area for G gated layers (64-byte aligned arrays).
uint64_t dg_mapped_bytes(uint32_t gated);

// The context's own host memory beside the plan's blocks, as the open admits and allocates it: the mapped control area,
// one bounce buffer per file reader (pageable), the one-unit staging buffer and the file tier's staging buffers. The
// host admission below and the whole-device planner (ss_budget_in.dg_gated_layers, SS_BUDGET_LIBRARY) both take it
// from here, so a plan charges what an open admits.
struct DgHostBytes {
  uint64_t mapped = 0;          // pinned
  uint64_t bounce = 0;          // pageable, per reader
  uint32_t bounce_readers = 0;  // the context's thread + the disk readers (dg_disk_queue > 1)
  uint64_t staging = 0;         // pinned: one unit, 4 KiB rounded
  uint64_t disk_staging = 0;    // pinned: the file tier's staging buffers, one host stride each
  uint64_t bounce_total() const { return bounce * bounce_readers; }  // (checked by dg_host_bytes)
  uint64_t total() const { return mapped + bounce_total() + staging + disk_staging; }
};
// gated = G; unit_bytes = one unit; part_max = its largest part; max_align = the largest source alignment (4096 at
// least); staging_request = ss_options.staging_bytes (0 = 16 MiB, at most 16 MiB); disk_queue / disk_staging = file
// reads in flight and staging buffers, resolved (0 without the disk tier). Throws SS_E_INVALID on overflow.
DgHostBytes dg_host_bytes(uint32_t gated, uint64_t unit_bytes, uint64_t part_max, uint64_t max_align,
                          uint64_t staging_request, uint32_t disk_queue, uint32_t disk_staging);
// The read locations (directories) serving the gated sources' bytes: the most over the sources a gated layer's parts
// read of stripe_locations (the primary plus each weight_dirs directory holding that source's stripe manifest; 1 = not
// striped). Directories, not volumes: two directories on one drive count twice. A stat per directory, before any
// source is opened (the context opens them after the layout).
uint32_t dg_disk_locations(const Desc& d, const ss_options& o);
// SS_DG_DISK_QUEUE_AUTO for that many locations: 3 reads in flight per location, at least 4 (one drive saturates at 4)
// and at most 8 (three drives peaked at 8), then at most an explicit dg_disk_staging (0 = not given): reads in flight
// are bounded by the staging buffers anyway, so no reader or bounce buffer beyond them is made.
uint32_t dg_auto_disk_queue(uint32_t locations, uint32_t staging_option);
// The staging buffers for a resolved AUTO queue: an explicit dg_disk_staging as given, else the default 8 raised to the
// queue (so the default never caps an AUTO queue).
uint32_t dg_auto_disk_staging(uint32_t staging_option, uint32_t queue);
// dg_disk_queue / dg_disk_staging as the open resolves them (dg_layout and ss_budget_in_dg_library both call this):
// without dg_disk_tier all 0 (AUTO included: no reader, no pool); otherwise staging = dg_disk_staging (0 = 8), queue
// = dg_disk_queue (0 = 1, 2..64 as given), or for SS_DG_DISK_QUEUE_AUTO the two above from dg_disk_locations.
// locations is counted only for AUTO with the disk tier (0 otherwise: an explicit queue does no file system work);
// it is reported in ss_dg_stats.disk_locations. Values are not range-checked here (dg_layout checks the options).
struct DgDiskQueue {
  uint32_t queue = 0, staging = 0, locations = 0;
};
DgDiskQueue dg_resolve_disk_queue(const Desc& d, const ss_options& o);
// The planner's inputs for that memory (ss_budget_in_dg_library): the gated layers, the largest part, the source
// alignment, the bounce request and the disk queue of this descriptor and these options (AUTO resolved as the open
// resolves it, dg_resolve_disk_queue), and staging = the open's staging buffers (dg_disk_staging, 0 = 8, or the AUTO
// raise).
void dg_budget_inputs(const Desc& d, const ss_options& o, ss_budget_in* in);
// Throws ts::Error naming the refused rule. Checks every v1 scope rule except the backend (the context's).
DgLayout dg_layout(const Desc& d, const Plan& p, const ss_options& o);

// The device's own error word (err_code, err_aux; dg_proto.hpp) as text, empty for kDgOk. layer_token: the failed token
// was a layer token, whose end kernel also reports a stray full export with kDgDesync (err_aux = that gated layer).
std::string dg_device_error(uint32_t code, uint32_t aux, bool layer_token = false);
// A failed token's message: the device's error word first (when it holds one), then what the host saw (the end edge,
// the service's error), then the poisoning.
std::string dg_token_failure(uint32_t seq, const std::string& device, const std::string& host);
// The CUDA driver's module loading mode (ss_dg_stats.cuda_module_loading: 0 unknown, 1 eager, 2 lazy) by name, and the
// open's notice for it (empty unless lazy with gated layers).
const char* dg_module_loading_name(uint32_t mode);
std::string dg_module_loading_notice(uint32_t mode, size_t gated);

}  // namespace ts
