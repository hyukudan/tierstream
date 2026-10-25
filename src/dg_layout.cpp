// Device-gated decode mode: layout and admission (dg_layout.hpp).

#include "dg_layout.hpp"

#include <algorithm>

#include "dg_proto.hpp"
#include "errors.hpp"

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

}  // namespace

uint64_t dg_mapped_bytes(uint32_t G) {
  // Record[G], PredRecord[G], flag[G], pflag[G], ready[G], then done / err_code / err_aux / host_abort / host_seq /
  // began / err_count
  return align64(uint64_t(G) * sizeof(DgRecord)) + align64(uint64_t(G) * sizeof(DgPredRecord)) +
         3 * align64(uint64_t(G) * 4) + align64(7 * 4);
}

DgLayout dg_layout(const Desc& d, const Plan& p, const ss_options& o) {
  auto refuse = [](ss_status st, const std::string& why) { fail(st, "device-gated: " + why); };
  if (o.forward_mode != SS_FORWARD_DEVICE_GATED) refuse(SS_E_INVALID, "forward_mode is not SS_FORWARD_DEVICE_GATED");
  const ss_budget& b = p.budget;
  if (b.tier_mode != SS_TIER_MODE_EXCLUSIVE) refuse(SS_E_INVALID, "needs tier_mode EXCLUSIVE");
  if (b.pinned_backing) refuse(SS_E_INVALID, "pinned_backing is refused (each unit has one home)");
  if (o.policy != SS_POLICY_LRU) refuse(SS_E_UNSUPPORTED, "only policy LRU (SCORED is not ported)");
  if (o.pin_planned) refuse(SS_E_INVALID, "needs pin_planned = 0 (planned residents are a warm start)");
  if (!o.fill_residents) refuse(SS_E_INVALID, "needs fill_residents = 1 (a resident's slot is its only copy)");
  if (o.readahead_units) refuse(SS_E_INVALID, "readahead_units must be 0 (no gated unit is read from its file)");
  if (o.spec_persistent) refuse(SS_E_INVALID, "spec_persistent is refused (prediction is the service's)");
  if (o.dg_max_ids < 1 || o.dg_max_ids > kDgMaxIds) refuse(SS_E_INVALID, "dg_max_ids must be 1..64 (rows x top_k)");
  if (o.dg_wait_timeout_ms < 1) refuse(SS_E_INVALID, "dg_wait_timeout_ms must be >= 1");

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
      } else {
        refuse(SS_E_CAPACITY, "unit (" + std::to_string(l.index) + ", " + std::to_string(u) +
                                  ") is not resident or pinned (no gated unit is read from its file)");
      }
    }
    // the planner's billed dynamic slots: never more than the layer's non-resident units
    x.D += std::min<uint32_t>(b.dynamic_slots_per_layer, uint32_t(l.n_units) - planned);
    keys += uint64_t(l.n_units);
    require(keys < 0x7fffffffull, SS_E_INVALID, "device-gated: more than 2^31 - 1 gated units");
    x.unit_base.push_back(uint32_t(keys));
  }
  x.keys = uint32_t(keys);
  x.R = uint32_t(x.residents.size());
  if (o.dg_spare_slots) {  // one global count, as the engine caches (the plan bills per layer)
    if (o.dg_spare_slots > x.P)
      refuse(SS_E_INVALID, "dg_spare_slots " + std::to_string(o.dg_spare_slots) + " above the non-resident units (" +
                               std::to_string(x.P) + ")");
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
  // buffer (min(staging_bytes, 16 MiB), at least 4 x the largest source alignment) and the open-time staging buffer
  // (one unit, 4 KiB rounded). The context allocates exactly these.
  uint64_t max_align = 4096;
  for (const Source& src : d.sources) max_align = std::max(max_align, src.align);
  const uint64_t want = std::min<uint64_t>(o.staging_bytes ? o.staging_bytes : (uint64_t(16) << 20), uint64_t(16) << 20);
  x.bounce_bytes = std::max<uint64_t>((add_ck(want, max_align - 1, "the bounce buffer") / max_align) * max_align,
                                      mul_ck(4, max_align, "the bounce buffer"));
  x.staging_bytes = (add_ck(x.unit_bytes, 4095, "the staging buffer") / 4096) * 4096;
  x.mapped_bytes = dg_mapped_bytes(G);
  const uint64_t host_extra =
      add_ck(add_ck(x.mapped_bytes, x.bounce_bytes, "the host extras"), x.staging_bytes, "the host extras");
  if (add_ck(p.pinned, host_extra, "the host total") > b.host_pinned_bytes)
    refuse(SS_E_CAPACITY, "the plan's pinned bytes (" + std::to_string(p.pinned) + ") plus the mapped control area (" +
                              std::to_string(x.mapped_bytes) + "), the bounce buffer (" +
                              std::to_string(x.bounce_bytes) + ") and the staging buffer (" +
                              std::to_string(x.staging_bytes) + ") exceed the host budget (" +
                              std::to_string(b.host_pinned_bytes) + ")");
  return x;
}

}  // namespace ts
