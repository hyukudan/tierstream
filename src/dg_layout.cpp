// Device-gated decode mode: layout and admission (dg_layout.hpp).

#include "dg_layout.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "dg_proto.hpp"
#include "dg_score.hpp"
#include "errors.hpp"
#include "source_file.hpp"

namespace ts {

namespace {

uint64_t add_ck(uint64_t a, uint64_t b, const char* what) {
  require(a <= UINT64_MAX - b, SS_E_INVALID, std::string("device-gated: ") + what + " overflows");
  return a + b;
}
uint64_t mul_ck(uint64_t a, uint64_t b, const char* what) {
  require(b == 0 || a <= UINT64_MAX / b, SS_E_INVALID, std::string("device-gated: ") + what + " overflows");
  return a * b;
}
uint64_t align64(uint64_t v) { return (v + 63) / 64 * 64; }
std::string gib(uint64_t b) { return std::to_string(double(b) / double(kGiB)); }

// The startup environment switches. A count: decimal digits only (no sign, no leading whitespace), at most max; a
// flag: exactly 0 / false / off or 1 / true / on (unset or empty = def). Anything else refuses the open (SS_E_INVALID).
bool env_count(const char* name, unsigned long max, unsigned long* out) {
  const char* v = std::getenv(name);
  if (!v) return false;
  char* end = nullptr;
  const unsigned long n = std::strtoul(v, &end, 10);
  require(*v >= '0' && *v <= '9' && end && !*end && n <= max, SS_E_INVALID,
          std::string("device-gated: invalid ") + name);
  *out = n;
  return true;
}
bool env_flag(const char* name, bool def) {
  const char* v = std::getenv(name);
  if (!v || !*v) return def;
  const std::string s = v;
  if (s == "1" || s == "true" || s == "on") return true;
  require(s == "0" || s == "false" || s == "off", SS_E_INVALID,
          std::string("device-gated: invalid ") + name + " (use 0, 1, false, true, off or on)");
  return false;
}

}  // namespace

std::string dg_device_error(uint32_t code, uint32_t aux, bool layer_token) {
  if (code == kDgOk) return std::string();
  const char* what = code == kDgTimeout  ? "a wait timed out"
                     : code == kDgBadId  ? "an id was out of range"
                     : code == kDgDesync ? "a step ran outside ss_begin / ss_end (sequence desync)"
                     : code == kDgAbsent ? "a slot lookup met an absent entry"
                                         : "the host aborted the token";
  const std::string where = code == kDgDesync && layer_token
                                ? " (that step's seq, or the gated layer exported in full inside the layer token: " +
                                      std::to_string(aux) + ")"
                            : code == kDgDesync ? " (that step's seq " + std::to_string(aux) + ")"
                            : code == kDgAbsent ? std::string()
                                                : " at gated layer " + std::to_string(aux);
  return what + where;
}

std::string dg_token_failure(uint32_t seq, const std::string& device, const std::string& host) {
  std::string m = "device-gated: token " + std::to_string(seq) + ": ";
  if (!device.empty()) m += "the device: " + device + (host.empty() ? "" : "; ");
  m += host;
  return m + " (the context is poisoned, the token's outputs must be discarded)";
}

const char* dg_module_loading_name(uint32_t mode) {
  return mode == 1 ? "eager" : mode == 2 ? "lazy" : "unknown";
}

std::string dg_module_loading_notice(uint32_t mode, size_t gated) {
  if (mode != 2 || gated == 0) return std::string();
  return "CUDA module loading is lazy: the gated layers' wait kernels can stall behind the first-use kernel loads of "
         "the engine (CUDA_MODULE_LOADING=EAGER loads every module at startup and avoids it)";
}

uint64_t dg_mapped_bytes(uint32_t G) {
  // Record[G], PredRecord[G], flag[G], pflag[G], ready[G], then done / err_code / err_aux / host_abort / host_seq /
  // began / err_count, then the layer tokens' part records, part flags and part ready words
  return align64(uint64_t(G) * sizeof(DgRecord)) + align64(uint64_t(G) * sizeof(DgPredRecord)) +
         3 * align64(uint64_t(G) * 4) + align64(7 * 4) + align64(uint64_t(kDgMaxParts) * sizeof(DgRecord)) +
         2 * align64(uint64_t(kDgMaxParts) * 4);
}

DgHostBytes dg_host_bytes(uint32_t gated, uint64_t unit_bytes, uint64_t part_max, uint64_t max_align,
                          uint64_t staging_request, uint32_t disk_queue, uint32_t disk_staging,
                          uint32_t landing_blocks) {
  // A bounce buffer per reader (the context's own thread, and each disk reader with dg_disk_queue > 1: none waits for
  // another's): min(staging_bytes, 16 MiB), at least 4 x the largest source alignment; with disk readers each is cut
  // to what the largest part's covering read needs (one read per part, never more than that size).
  DgHostBytes h;
  max_align = std::max<uint64_t>(max_align, 4096);
  const uint64_t want = std::min<uint64_t>(staging_request ? staging_request : (uint64_t(16) << 20), uint64_t(16) << 20);
  const auto aligned_up = [max_align](uint64_t v) {
    return (add_ck(v, max_align - 1, "the bounce buffer") / max_align) * max_align;
  };
  h.bounce = std::max<uint64_t>(aligned_up(want), mul_ck(4, max_align, "the bounce buffer"));
  h.bounce_readers = 1 + (disk_queue > 1 ? disk_queue : 0);
  if (h.bounce_readers > 1) {
    const uint64_t cover = aligned_up(add_ck(part_max, 2 * max_align, "the bounce buffer"));
    h.bounce = std::max<uint64_t>(std::min(h.bounce, cover), mul_ck(4, max_align, "the bounce buffer"));
  }
  const uint64_t bounce_total = mul_ck(h.bounce, h.bounce_readers, "the bounce buffers");
  h.staging = (add_ck(unit_bytes, 4095, "the staging buffer") / 4096) * 4096;
  h.mapped = dg_mapped_bytes(gated);
  h.disk_staging = mul_ck(disk_staging, host_stride(unit_bytes), "the disk staging buffers");
  h.landing = mul_ck(landing_blocks, host_stride(unit_bytes), "the landing blocks");
  // (total() cannot overflow once this sum did not)
  (void)add_ck(add_ck(add_ck(add_ck(h.mapped, bounce_total, "the host extras"), h.staging, "the host extras"),
                      h.disk_staging, "the host extras"),
               h.landing, "the host extras");
  return h;
}

void dg_budget_inputs(const Desc& d, const ss_options& o, ss_budget_in* in) {
  uint32_t gated = 0;
  uint64_t part_max = 0, max_align = 4096;
  for (const Layer& l : d.layers) {
    if (l.n_units == 0) continue;  // the gated layers: every layer with units (dg_layout)
    ++gated;
    for (const Part& part : l.parts) part_max = std::max(part_max, part.bytes);
  }
  for (const Source& src : d.sources) max_align = std::max(max_align, src.align);
  in->dg_gated_layers = gated;
  in->dg_disk_queue = o.dg_disk_queue;
  in->dg_part_max_bytes = part_max;
  in->dg_source_align = max_align;
  in->dg_bounce_request = o.staging_bytes;
  in->staging = o.dg_disk_staging ? o.dg_disk_staging : 8u;
  if (o.dg_disk_queue == SS_DG_DISK_QUEUE_AUTO) {  // resolved as the open resolves it (no disk tier: 0, no reader)
    const DgDiskQueue dq = dg_resolve_disk_queue(d, o);
    in->dg_disk_queue = dq.queue;
    if (dq.staging) in->staging = dq.staging;
  }
  in->dg_landing_blocks = o.dg_disk_tier ? o.dg_landing_blocks : 0;  // (charged only with a file tier in the plan)
}

uint32_t dg_disk_locations(const Desc& d, const ss_options& o) {
  std::vector<bool> read(d.sources.size(), false);
  for (const Layer& l : d.layers)
    if (l.n_units != 0)  // the gated layers (dg_layout)
      for (const Part& part : l.parts)
        if (part.source >= 0 && size_t(part.source) < read.size()) read[size_t(part.source)] = true;
  uint32_t most = 1;
  for (size_t i = 0; i < read.size(); ++i)
    if (read[i]) most = std::max(most, stripe_locations(d.sources[i].path, o.weight_dirs));
  return most;
}

uint32_t dg_auto_disk_queue(uint32_t locations, uint32_t staging_option) {
  const uint32_t q = uint32_t(std::min<uint64_t>(std::max<uint64_t>(uint64_t(3) * locations, 4), 8));  // 4..8
  return staging_option ? std::min(q, staging_option) : q;  // reads in flight are bounded by the staging buffers
}

uint32_t dg_auto_disk_staging(uint32_t staging_option, uint32_t queue) {
  return staging_option ? staging_option : std::max<uint32_t>(8, queue);
}

DgDiskQueue dg_resolve_disk_queue(const Desc& d, const ss_options& o) {
  DgDiskQueue r;
  if (!o.dg_disk_tier) return r;  // no disk tier: no staging, no reader (SS_DG_DISK_QUEUE_AUTO included)
  if (o.dg_disk_queue == SS_DG_DISK_QUEUE_AUTO) {  // only AUTO touches the file system (an explicit queue: as before)
    r.locations = dg_disk_locations(d, o);
    r.queue = dg_auto_disk_queue(r.locations, o.dg_disk_staging);
    r.staging = dg_auto_disk_staging(o.dg_disk_staging, r.queue);
  } else {
    r.queue = o.dg_disk_queue ? o.dg_disk_queue : 1u;
    r.staging = o.dg_disk_staging ? o.dg_disk_staging : 8u;
  }
  return r;
}

DgLayout dg_layout(const Desc& d, const Plan& p, const ss_options& o) {
  auto refuse = [](ss_status st, const std::string& why) { fail(st, "device-gated: " + why); };
  if (o.forward_mode != SS_FORWARD_DEVICE_GATED) refuse(SS_E_INVALID, "forward_mode is not SS_FORWARD_DEVICE_GATED");
  const ss_budget& b = p.budget;
  if (b.tier_mode != SS_TIER_MODE_EXCLUSIVE) refuse(SS_E_INVALID, "needs tier_mode EXCLUSIVE");
  if (b.pinned_backing) refuse(SS_E_INVALID, "pinned_backing is refused (each unit has one home)");
  if (o.policy != SS_POLICY_LRU && o.policy != SS_POLICY_SCORED && o.policy != SS_POLICY_SCORED_AUTO)
    refuse(SS_E_INVALID, "policy must be LRU or SCORED or SCORED_AUTO");
  if ((o.dg_score_profile_len || o.dg_score_profile) && o.policy != SS_POLICY_SCORED && o.policy != SS_POLICY_SCORED_AUTO)
    refuse(SS_E_INVALID, "dg_score_profile needs policy SCORED");
  if (o.pin_planned) refuse(SS_E_INVALID, "needs pin_planned = 0 (planned residents are a warm start)");
  if (!o.fill_residents) refuse(SS_E_INVALID, "needs fill_residents = 1 (a resident's slot is its only copy)");
  if (o.readahead_units) refuse(SS_E_INVALID, "readahead_units must be 0 (no gated unit is read from its file)");
  if (o.spec_persistent) refuse(SS_E_INVALID, "spec_persistent is refused (prediction is the service's)");
  if (o.dg_max_ids < 1 || o.dg_max_ids > kDgMaxIds) refuse(SS_E_INVALID, "dg_max_ids must be 1..64 (rows x top_k)");
  if (o.dg_wait_timeout_ms < 1) refuse(SS_E_INVALID, "dg_wait_timeout_ms must be >= 1");
  if (o.dg_wb_defer > 1) refuse(SS_E_INVALID, "dg_wb_defer must be 0 or 1");
  if (o.dg_wb_stream > 1) refuse(SS_E_INVALID, "dg_wb_stream must be 0 or 1");
  if (o.dg_disk_tier > 1 || o.dg_disk_prefetch > 1) refuse(SS_E_INVALID, "dg_disk_tier / dg_disk_prefetch must be 0 or 1");
  // (SS_DG_DISK_QUEUE_AUTO without the disk tier is accepted: it resolves to no reader, as 0 does)
  const bool queue_auto = o.dg_disk_queue == SS_DG_DISK_QUEUE_AUTO;
  if (!o.dg_disk_tier && (o.dg_disk_prefetch || o.dg_disk_staging || (o.dg_disk_queue && !queue_auto)))
    refuse(SS_E_INVALID, "dg_disk_prefetch / dg_disk_staging / dg_disk_queue need dg_disk_tier = 1");
  if (o.dg_disk_staging > 256) refuse(SS_E_INVALID, "dg_disk_staging must be 0..256");
  if (o.dg_disk_queue > 64 && !queue_auto)
    refuse(SS_E_INVALID, "dg_disk_queue must be 0..64 or SS_DG_DISK_QUEUE_AUTO");
  if (o.dg_no_admit > 1) refuse(SS_E_INVALID, "dg_no_admit must be 0 or 1");
  if (o.dg_clean_evict > 1) refuse(SS_E_INVALID, "dg_clean_evict must be 0 or 1");
  if (o.dg_bank_keep > 2) refuse(SS_E_INVALID, "dg_bank_keep must be 0, 1 or 2");
  if (o.dg_disk_queue_adapt > 1) refuse(SS_E_INVALID, "dg_disk_queue_adapt must be 0 or 1");
  if (o.dg_disk_queue_adapt && (!o.dg_disk_tier || !queue_auto))
    refuse(SS_E_INVALID, "dg_disk_queue_adapt needs dg_disk_tier = 1 and dg_disk_queue = SS_DG_DISK_QUEUE_AUTO");
  if (o.dg_pf_revert_unused > 1) refuse(SS_E_INVALID, "dg_pf_revert_unused must be 0 or 1");
  if (o.dg_pf_strict_admit > 1) refuse(SS_E_INVALID, "dg_pf_strict_admit must be 0 or 1");
  if (o.dg_landing_blocks > 4096) refuse(SS_E_INVALID, "dg_landing_blocks must be 0..4096");
  if (!o.dg_landing_blocks && (o.dg_landing_max_queued || o.dg_landing_ttl || o.dg_landing_chunk))
    refuse(SS_E_INVALID, "dg_landing_max_queued / dg_landing_ttl / dg_landing_chunk need dg_landing_blocks > 0");
  if (o.dg_landing_max_queued > 64) refuse(SS_E_INVALID, "dg_landing_max_queued must be 0..64");
  if (o.dg_landing_ttl > 1024) refuse(SS_E_INVALID, "dg_landing_ttl must be 0..1024");
  if (o.dg_landing_chunk != 0xFFFFFFFFu && o.dg_landing_chunk % 4096)
    refuse(SS_E_INVALID, "dg_landing_chunk must be 0, a multiple of 4096 or 0xFFFFFFFF (no chunks)");
  if (o.dg_landing_blocks && !o.dg_disk_tier) refuse(SS_E_INVALID, "dg_landing_blocks needs dg_disk_tier = 1");
  if (o.dg_landing_blocks && o.dg_disk_queue <= 1)
    refuse(SS_E_INVALID, "dg_landing_blocks needs dg_disk_queue > 1 (the landing reads run on the read pool)");

  DgLayout x;
  x.B = o.dg_max_ids;
  x.ordinal.assign(d.layers.size(), -1);
  const Layer* shape = nullptr;
  for (size_t L = 0; L < d.layers.size(); ++L) {
    const Layer& l = d.layers[L];
    if (l.n_units == 0) continue;
    if (!l.streamable)
      refuse(SS_E_UNSUPPORTED, "layer " + std::to_string(l.index) + " has non-streamable units (v1 gates every unit)");
    if (!shape) {
      shape = &l;
      if (l.parts.size() > SS_DG_MAX_PARTS)
        refuse(SS_E_UNSUPPORTED, "more than " + std::to_string(SS_DG_MAX_PARTS) + " parts per unit");
      uint64_t off = 0;
      for (const Part& part : l.parts) {
        if (part.bytes % 256 != 0)
          refuse(SS_E_UNSUPPORTED, "part " + part.name + " is " + std::to_string(part.bytes) +
                                       " bytes, not a multiple of 256 (the arena stride equals the part bytes)");
        x.part_bytes.push_back(part.bytes);
        x.part_offset.push_back(off);
        off += part.bytes;
      }
      x.unit_bytes = l.unit_bytes;
    } else {
      bool same = l.parts.size() == shape->parts.size();
      for (size_t i = 0; same && i < l.parts.size(); ++i) same = l.parts[i].bytes == shape->parts[i].bytes;
      if (!same) refuse(SS_E_UNSUPPORTED, "layer " + std::to_string(l.index) + " has another part shape (one arena)");
    }
    x.ordinal[L] = int(x.gated.size());
    x.gated.push_back(int(L));
  }
  if (x.gated.empty()) refuse(SS_E_INVALID, "the descriptor has no streamable units");
  const uint32_t G = uint32_t(x.gated.size());
  x.host_stride = host_stride(x.unit_bytes);

  // Homes from the plan: every gated unit resident or pinned (coverage), keys in descriptor order.
  x.unit_base.assign(1, 0);
  uint64_t keys = 0;
  for (uint32_t g = 0; g < G; ++g) {
    const int L = x.gated[g];
    const Layer& l = d.layers[size_t(L)];
    uint32_t planned = 0;
    for (int u = 0; u < l.n_units; ++u) {
      const uint8_t t = p.tier[size_t(d.unit_base[size_t(L)] + u)];
      if (t == SS_TIER_RESIDENT) {
        x.residents.push_back(uint32_t(keys + uint64_t(u)));
        ++planned;
      } else if (t == SS_TIER_PINNED) {
        ++x.P;
      } else if (t == SS_TIER_DISK && o.dg_disk_tier) {
        ++x.F;
      } else {
        refuse(SS_E_CAPACITY, "unit (" + std::to_string(l.index) + ", " + std::to_string(u) +
                                  ") is not resident or pinned (no gated unit is read from its file)");
      }
    }
    // the planner's billed dynamic slots: never more than the layer's non-resident units
    x.D += std::min<uint32_t>(b.dynamic_slots_per_layer, uint32_t(l.n_units) - planned);
    if (o.dg_disk_tier)
      for (int u = 0; u < l.n_units; ++u)
        x.disk.push_back(p.tier[size_t(d.unit_base[size_t(L)] + u)] == SS_TIER_DISK ? 1 : 0);
    keys += uint64_t(l.n_units);
    require(keys < 0x7fffffffull, SS_E_INVALID, "device-gated: more than 2^31 - 1 gated units");
    x.unit_base.push_back(uint32_t(keys));
  }
  x.keys = uint32_t(keys);
  x.R = uint32_t(x.residents.size());
  x.disk_tier = o.dg_disk_tier != 0;
  x.disk_prefetch = o.dg_disk_prefetch != 0;
  if (o.dg_spare_slots) {  // one global count, as the engine caches (the plan bills per layer)
    if (o.dg_spare_slots > x.P + x.F)
      refuse(SS_E_INVALID, "dg_spare_slots " + std::to_string(o.dg_spare_slots) + " above the non-resident units (" +
                               std::to_string(x.P + x.F) + ")");
    x.D = o.dg_spare_slots;
  }
  if (x.D < 1)
    refuse(SS_E_INVALID, "needs dynamic_slots_per_layer > 0 or dg_spare_slots > 0 (the spare slots; the boundary keeps "
                         "them free)");

  // Banks + guard from the billed ring of the expert size.
  const uint64_t stride = d.unit_stride(x.gated[0]);
  uint32_t ring_slots = 0;
  for (const RingClass& rc : p.rings)
    if (rc.stride == stride) ring_slots = rc.slots;
  if (o.dg_no_admit && uint64_t(ring_slots) < 2ull * x.B + 1)  // (the same slots: named for the miss path)
    refuse(SS_E_CAPACITY, "dg_no_admit serves every miss through a bank slot: it needs " + std::to_string(2 * x.B) +
                              " bank slots (2 banks x dg_max_ids " + std::to_string(x.B) + ") + the guard, the plan "
                              "bills " + std::to_string(ring_slots) + " ring slots (set ring_slots = 2 x dg_max_ids + 1)");
  if (uint64_t(ring_slots) < 2ull * x.B + 1)
    refuse(SS_E_CAPACITY, "the plan bills " + std::to_string(ring_slots) + " ring slots, the banks and the guard need " +
                              std::to_string(2 * x.B + 1) + " (set ring_slots = 2 x dg_max_ids + 1)");

  // Spare blocks the planner granted and billed for this host stride.
  for (const SpareClass& sc : p.spares)
    if (sc.block_bytes == x.host_stride) x.Q = sc.blocks;
  // Q may be below D (the engine caches run 96 spare slots on 48 spare blocks): a boundary then writes back at most
  // the free blocks and keeps the other victims resident (evict_short), so fewer slots are free, never a wrong one.
  if (x.Q < 1)
    refuse(SS_E_CAPACITY, "the plan granted no spare blocks (host_spare_units asked " +
                              std::to_string(b.host_spare_units) + "): a write-back needs a free block (raise "
                              "host_spare_units or the host budget)");
  require(uint64_t(x.P) + x.Q < 0x7fffffffull, SS_E_INVALID, "device-gated: more than 2^31 - 1 host blocks");
  x.blocks = x.P + x.Q;
  // the inclusive share comes out of the spare blocks: at least one stays for write-backs
  x.I = o.dg_inclusive_blocks;
  if (x.I && x.I >= x.Q)
    refuse(SS_E_CAPACITY, "dg_inclusive_blocks " + std::to_string(x.I) + " must stay below the spare blocks (" +
                              std::to_string(x.Q) + "): raise host_spare_units by the inclusive share");
  // 0 = 1 (synchronous reads): a measured A/B ran depth 16 slower while the reads shared one bounce buffer and ended in
  // fetch order; the default stays 1 until an A/B with per-reader bounce buffers and independent read ends says otherwise.
  // SS_DG_DISK_QUEUE_AUTO resolves here, before the staging buffers, readers and bounce buffers are sized (the same
  // function as the planner's, so ss_budget_in_dg_library bills the same readers).
  const DgDiskQueue dq = dg_resolve_disk_queue(d, o);
  x.S = dq.staging;
  x.disk_queue = dq.queue;
  x.disk_locations = dq.locations;
  // landing blocks only when the plan has a file tier (nothing to land otherwise; the planner charges them so too)
  x.L = x.F ? o.dg_landing_blocks : 0u;
  x.landing_queued = o.dg_landing_max_queued ? o.dg_landing_max_queued : 1u;
  x.landing_ttl = o.dg_landing_ttl;  // (0: expiry at the target layer)
  x.landing_chunk = o.dg_landing_chunk == 0xFFFFFFFFu ? 0u : o.dg_landing_chunk ? o.dg_landing_chunk : 4u << 20;

  // Scored eviction: the weights, the per-layer bound and the table size (checked), the profile (every rule of its
  // format; the context loads it before the service starts).
  if (o.policy == SS_POLICY_SCORED || o.policy == SS_POLICY_SCORED_AUTO) {
    x.scored = true;
    if (o.dg_score_w_trans || o.dg_score_w_freq || o.dg_score_w_recency) {
      if (o.dg_score_w_trans > 1000 || o.dg_score_w_freq > 1000 || o.dg_score_w_recency > 1000 ||
          o.dg_score_w_trans + o.dg_score_w_freq + o.dg_score_w_recency != 1000)
        refuse(SS_E_INVALID, "dg_score_w_trans / _freq / _recency must each be 0..1000 and sum to 1000");
      x.w_trans = o.dg_score_w_trans;
      x.w_freq = o.dg_score_w_freq;
      x.w_rec = o.dg_score_w_recency;
    }
    std::vector<uint32_t> units;
    for (int L : x.gated) units.push_back(uint32_t(d.layers[size_t(L)].n_units));
    const uint64_t cells = dg_score_cells(units);
    if (cells > uint64_t(SIZE_MAX / 8)) refuse(SS_E_INVALID, "the score tables do not fit the address space");
    if (o.dg_score_profile_len || o.dg_score_profile) {
      if (!o.dg_score_profile || !o.dg_score_profile_len)
        refuse(SS_E_INVALID, "dg_score_profile and dg_score_profile_len must be given together");
      x.score_identity = dg_score_identity(d, x.gated);
      (void)dg_profile_check(o.dg_score_profile, o.dg_score_profile_len, units, x.score_identity);
    }
  }

  x.persistent_slots = x.R + x.D;
  const uint64_t S = uint64_t(x.persistent_slots) + 2ull * x.B + 1;
  require(S < 0x7fffffffull, SS_E_INVALID, "device-gated: more than 2^31 - 1 slots");
  x.slots = uint32_t(S);
  x.guard_slot = x.slots - 1;

  // Device admission: replace what the plan billed for the gated slots by what the arena needs.
  uint64_t per_slot = 0;
  for (uint64_t pb : x.part_bytes) per_slot = add_ck(per_slot, pb, "slot bytes");
  x.need = mul_ck(S, per_slot, "the arena");
  x.need = add_ck(x.need, mul_ck(keys, 4, "slot_of"), "the arena");
  x.device_ctl_bytes = DgLayout::ctl_words(G) * 4;
  x.need = add_ck(x.need, x.device_ctl_bytes, "the device controls");
  x.billed = add_ck(add_ck(p.resident, p.dynamic, "billed"), p.ring, "billed");
  const uint64_t limit = p.cap > p.margin ? p.cap - p.margin : 0;
  const uint64_t other = p.peak >= x.billed ? p.peak - x.billed : 0;
  if (add_ck(other, x.need, "the device total") > limit)
    refuse(SS_E_CAPACITY, "the arena needs " + gib(x.need) + " GiB where the plan billed " + gib(x.billed) +
                              " GiB; with the rest of the plan (" + gib(other) + " GiB) that is above the cap - margin (" +
                              gib(limit) + " GiB)");
  // Host admission: on top of what the plan bills (blocks of P + Q, host_fixed): the mapped control area, the bounce
  // buffers, the staging buffer (one unit, 4 KiB rounded; kept in pipeline mode too: ss_dg_reset_open's one-unit spare
  // for placement cycles), the disk staging buffers and the landing blocks, as dg_host_bytes computes them for the
  // planner too. The context allocates exactly these.
  uint64_t max_align = 4096, part_max = 0;
  for (const Source& src : d.sources) max_align = std::max(max_align, src.align);
  for (int L : x.gated)
    for (const Part& part : d.layers[size_t(L)].parts) part_max = std::max(part_max, part.bytes);
  const DgHostBytes hb = dg_host_bytes(G, x.unit_bytes, part_max, max_align, o.staging_bytes, x.disk_queue, x.S, x.L);
  x.bounce_bytes = hb.bounce;
  x.bounce_readers = hb.bounce_readers;
  const uint64_t bounce_total = hb.bounce_total();
  x.load_serial = env_flag("DG_LOAD_SERIAL", false);
  x.pf_legacy = env_flag("DG_PF_LEGACY", false);  // the new admission (measured not slower); 1 = the previous one
  x.staging_bytes = hb.staging;
  x.mapped_bytes = hb.mapped;
  const uint64_t host_extra = hb.total();
  if (add_ck(p.pinned, host_extra, "the host total") > b.host_pinned_bytes)
    refuse(SS_E_CAPACITY, "the plan's pinned bytes (" + std::to_string(p.pinned) + ") plus the mapped control area (" +
                              std::to_string(x.mapped_bytes) + "), the bounce buffers (" +
                              std::to_string(bounce_total) + ") and the staging buffer (" +
                              std::to_string(x.staging_bytes) + "), the disk staging buffers (" +
                              std::to_string(uint64_t(x.S) * x.host_stride) + "), the landing blocks (" +
                              std::to_string(hb.landing) + ") exceed the host budget (" +
                              std::to_string(b.host_pinned_bytes) + ")");
  // A unit's in-place parts as concurrent sub-reads (the file misses, the prefetch reads, the serial loader), e.g.
  // 1024 = 1 MiB pieces (a medium request: a multi-location session balances each across the homes of its chunk); a
  // multiple of 4 KiB keeps every piece aligned; 0 (default, until it measures faster) = one request per part.
  unsigned long subread = 0;
  if (env_count("DG_UNIT_SUBREAD_KIB", 65536, &subread))
    require(subread % 4 == 0, SS_E_INVALID, "device-gated: invalid DG_UNIT_SUBREAD_KIB (0..65536, a multiple of 4)");
  x.unit_subread = uint64_t(subread) << 10;
  if (!x.load_serial) {
    auto option = [](const char* name, uint32_t def, uint32_t max) {
      unsigned long n = def;
      if (env_count(name, max, &n))
        require(n >= 1, SS_E_INVALID, std::string("device-gated: invalid ") + name);
      return uint32_t(n);
    };
    x.load_readers = option("DG_LOAD_READERS", 8, 32);
    x.load_buffers = option("DG_LOAD_BUFFERS", 4, 16);
    const uint64_t target = uint64_t(option("DG_LOAD_BUFFER_MIB", 64, 256)) << 20;
    const uint64_t available = b.host_pinned_bytes - p.pinned - host_extra;
    // Defaults shrink to the admitted host headroom (small fixtures need only KiB).
    // Explicit sizes are exact and fail admission rather than silently shrinking.
    x.load_buffer_bytes = std::getenv("DG_LOAD_BUFFER_MIB") ? target :
        std::min(std::max(target, x.unit_bytes), available / x.load_buffers / 4096 * 4096);
    require(x.load_buffer_bytes >= x.unit_bytes &&
            x.load_buffer_bytes <= available / x.load_buffers, SS_E_CAPACITY,
            "device-gated: startup ring exceeds the host budget (reduce DG_LOAD_BUFFERS or DG_LOAD_BUFFER_MIB)");
    x.load_overlap_alloc = env_flag("DG_LOAD_OVERLAP_ALLOC", true);
  }
  return x;
}

}  // namespace ts
