// Whole-device budgets: the planner and the ledger (budget.hpp).
#include "budget.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "errors.hpp"

struct ss_budget_ledger {
  std::mutex mu;
  uint64_t cap[SS_BUDGET_KINDS] = {};  // PINNED and PAGEABLE hold the one RAM cap (charged against their sum)
  uint64_t cur[SS_BUDGET_KINDS] = {}, peak[SS_BUDGET_KINDS] = {};
  uint64_t host_peak = 0;
  std::map<std::string, uint64_t> comp[SS_BUDGET_KINDS];
  std::vector<std::pair<std::string, uint64_t>> top[SS_BUDGET_KINDS];  // at each kind's peak, largest first
  uint64_t measured_last = 0, measured_peak = 0, observations = 0;
  uint32_t refusals = 0;
  // ss_budget_ledger_observe: host samples, samples above the caps, the stages of the peaks
  uint64_t host_last = 0, host_measured_peak = 0, host_observations = 0;
  uint32_t device_over = 0, host_over = 0;
  std::string device_peak_at, host_peak_at;
};

namespace ts {
namespace {

constexpr uint64_t kAuto = SS_BUDGET_AUTO;
constexpr uint64_t kGiB = uint64_t(1) << 30;

uint64_t add_sat(uint64_t a, uint64_t b) { return a > UINT64_MAX - b ? UINT64_MAX : a + b; }
uint64_t sub_sat(uint64_t a, uint64_t b) { return a > b ? a - b : 0; }
uint64_t mul_sat(uint64_t a, uint64_t b) { return a && b > UINT64_MAX / a ? UINT64_MAX : a * b; }

// GiB with two decimals, for messages only (no decision reads it).
std::string gib(uint64_t b) {
  char s[32];
  std::snprintf(s, sizeof s, "%.2f", double(b) / double(kGiB));
  return s;
}
uint64_t gib_ceil(uint64_t b) { return b / kGiB + (b % kGiB ? 1 : 0); }
// GiB with two decimals rounded UP: a minimum the caller can pass as it reads and that holds (integer arithmetic)
std::string gib_up(uint64_t b) {
  const uint64_t h = b / kGiB * 100 + ((b % kGiB) * 100 + kGiB - 1) / kGiB;
  char s[32];
  std::snprintf(s, sizeof s, "%llu.%02llu", static_cast<unsigned long long>(h / 100),
                static_cast<unsigned long long>(h % 100));
  return s;
}

// The named components of one kind: their sum (saturating) and "name size + name size" for refusals.
struct Parts {
  uint64_t sum = 0;
  std::string text;
  void add(const char* name, uint64_t bytes) {
    sum = add_sat(sum, bytes);
    if (!text.empty()) text += " + ";
    text += std::string(name && *name ? name : "unnamed") + " " + gib(bytes);
  }
};
Parts parts(const ss_budget_component* c, uint32_t n, const char* what) {
  require(n == 0 || c != nullptr, SS_E_INVALID, std::string("ss_budget_plan: ") + what + " is NULL with a count");
  Parts p;
  for (uint32_t i = 0; i < n; ++i) p.add(c[i].name, c[i].bytes);
  return p;
}

// The message, cut to the field at a UTF-8 boundary (a component's name is never split mid-character).
void set_refuse(ss_budget_out& out, const std::string& msg) {
  out.refused = 1;
  size_t n = std::min(msg.size(), sizeof out.refuse - 1);
  if (n < msg.size())
    while (n > 0 && (static_cast<unsigned char>(msg[n]) & 0xC0) == 0x80) --n;
  std::memcpy(out.refuse, msg.data(), n);
  out.refuse[n] = '\0';
}

const char* kind_name(ss_budget_kind k) {
  return k == SS_BUDGET_DEVICE ? "device" : k == SS_BUDGET_PINNED ? "pinned" : "pageable";
}

}  // namespace

bool unified_plan(const ss_budget_in& in, ss_budget_out& out);

void budget_in_defaults(ss_budget_in* in) {
  *in = ss_budget_in{};
  in->struct_size = uint32_t(sizeof *in);
  in->file_tier = 1;
  in->staging = 8;
  in->arena_units = kAuto;
  in->ram_expert_bytes = kAuto;
  in->inclusive_bytes = 0;
  in->inclusive_auto_cap_bytes = 16 * kGiB;
  in->host_avail_bytes = kAuto;
  in->max_pinned_bytes = kAuto;
}

bool budget_plan(const ss_budget_in& in, ss_budget_out& out) {
  if (in.unified && (in.vram_total_bytes || in.ram_total_bytes)) return unified_plan(in, out);
  out = ss_budget_out{};
  out.struct_size = uint32_t(sizeof out);
  const Parts dev = parts(in.device_fixed, in.n_device_fixed, "device_fixed");
  const Parts pin = parts(in.pinned_fixed, in.n_pinned_fixed, "pinned_fixed");
  const Parts page = parts(in.pageable_fixed, in.n_pageable_fixed, "pageable_fixed");
  const uint64_t unit = in.unit_bytes;
  const auto units_of = [unit](uint64_t bytes) { return unit ? bytes / unit : 0; };
  const uint64_t keys = in.keys;
  // the host placement's sums (non-residents + spares + staging + one block) are exact below this: a larger key count
  // is refused as a bad argument rather than saturated into a plan
  require(add_sat(add_sat(keys, in.host_spares), uint64_t(in.staging) + 1) != UINT64_MAX, SS_E_INVALID,
          "ss_budget_plan: keys " + std::to_string(keys) + " is out of range");

  // Device: context + components + the arena's spares (+ the smallest arena accepted), then the residents.
  out.device_fixed_bytes = add_sat(in.device_context_bytes, dev.sum);
  out.arena_spares = in.arena_spares;
  // (the margins only count against a cap: with none they are not read)
  const uint64_t vmargin = in.vram_total_bytes ? in.vram_margin_bytes : 0;
  const uint64_t rmargin = in.ram_total_bytes ? in.ram_margin_bytes : 0;
  out.min_vram_total_bytes = add_sat(add_sat(out.device_fixed_bytes, in.vram_margin_bytes),
                                     mul_sat(add_sat(in.arena_spares, in.min_resident), in.slot_bytes));
  uint64_t resident = in.arena_units == kAuto ? keys : std::min(in.arena_units, keys);
  if (in.vram_total_bytes) {
    if (out.min_vram_total_bytes > in.vram_total_bytes) {
      std::string m = "VRAM cap " + gib(in.vram_total_bytes) + " GiB cannot hold the fixed parts; needs at least " +
                      gib_up(out.min_vram_total_bytes) + " GiB (--vram-total-gib " +
                      std::to_string(gib_ceil(out.min_vram_total_bytes)) + "): context " +
                      gib(in.device_context_bytes);
      if (!dev.text.empty()) m += " + " + dev.text;
      if (vmargin) m += " + margin " + gib(vmargin);
      m += " + arena spares " + gib(mul_sat(in.arena_spares, in.slot_bytes));
      if (in.min_resident) m += " + minimal arena " + gib(mul_sat(in.min_resident, in.slot_bytes));
      set_refuse(out, m + " GiB");
      resident = 0;
    } else {
      const uint64_t room = sub_sat(sub_sat(sub_sat(in.vram_total_bytes, out.device_fixed_bytes), vmargin),
                                    mul_sat(in.arena_spares, in.slot_bytes));
      const uint64_t fit = in.slot_bytes ? room / in.slot_bytes : keys;
      out.device_room_bytes = sub_sat(sub_sat(in.vram_total_bytes, out.device_fixed_bytes), vmargin);
      out.device_capped = fit < resident ? 1 : 0;
      resident = std::min(resident, fit);
    }
  }
  out.resident = resident;
  out.arena_slots = add_sat(resident, in.arena_spares);
  out.device_planned_bytes = add_sat(out.device_fixed_bytes, mul_sat(out.arena_slots, in.slot_bytes));

  // Host: the pinned room (available now, the pinned ceiling, the RAM cap), then the three-tier placement.
  const uint64_t nonres = keys - resident;
  const uint64_t extra = in.load_staging_bytes;
  out.host_fixed_bytes = add_sat(add_sat(pin.sum, page.sum), extra);
  const uint64_t room_host =
      in.host_avail_bytes == kAuto ? UINT64_MAX : sub_sat(sub_sat(in.host_avail_bytes, in.host_margin_bytes), extra);
  const uint64_t room_pin =
      in.max_pinned_bytes == kAuto ? UINT64_MAX : sub_sat(sub_sat(in.max_pinned_bytes, pin.sum), extra);
  uint64_t room_bytes = std::min(room_host, room_pin);
  const uint64_t room_uncapped = room_bytes;
  if (in.ram_total_bytes)
    room_bytes = std::min(room_bytes, sub_sat(sub_sat(in.ram_total_bytes, out.host_fixed_bytes), rmargin));
  out.room_bytes = room_bytes;
  const uint64_t room = units_of(room_bytes);
  out.ram_auto = in.ram_expert_bytes == kAuto ? 1 : 0;
  uint64_t cap = out.ram_auto ? room : units_of(in.ram_expert_bytes);
  if (in.file_tier && cap > room) cap = room;  // never more than the host can pin now
  const uint64_t spares = in.host_spares;
  out.host_cap_units = cap;
  {  // the ceiling without the RAM cap: the cap binds when it lowers it below what the non-residents and spares want
    const uint64_t room_nc = units_of(room_uncapped);
    uint64_t cap_nc = out.ram_auto ? room_nc : units_of(in.ram_expert_bytes);
    if (in.file_tier && cap_nc > room_nc) cap_nc = room_nc;
    out.host_capped = in.ram_total_bytes && cap < cap_nc && cap < add_sat(nonres, spares) ? 1 : 0;
  }
  // the smallest host placement: the spare blocks, and the staging and one block (file tier) or every non-resident
  // expert (no file tier)
  const uint64_t min_units = nonres == 0 ? spares
                             : in.file_tier ? add_sat(spares, uint64_t(in.staging) + 1)
                                            : add_sat(spares, nonres);
  out.min_ram_total_bytes = add_sat(add_sat(out.host_fixed_bytes, in.ram_margin_bytes), mul_sat(min_units, unit));
  const bool ram_short = in.ram_total_bytes && out.min_ram_total_bytes > in.ram_total_bytes;
  if (ram_short && !out.refused) {
    std::string m = "RAM cap " + gib(in.ram_total_bytes) + " GiB cannot hold the fixed parts; needs at least " +
                    gib_up(out.min_ram_total_bytes) + " GiB (--ram-total-gib " +
                    std::to_string(gib_ceil(out.min_ram_total_bytes)) + "):";
    std::string t;
    if (!pin.text.empty()) t += " + " + pin.text;
    if (!page.text.empty()) t += " + " + page.text;
    if (extra) t += " + load staging " + gib(extra);
    if (rmargin) t += " + margin " + gib(rmargin);
    if (spares) t += " + spare blocks " + gib(mul_sat(spares, unit));
    if (nonres && in.file_tier) t += " + file staging and one block " + gib(mul_sat(uint64_t(in.staging) + 1, unit));
    if (nonres && !in.file_tier) t += " + non-resident experts (no file tier) " + gib(mul_sat(nonres, unit));
    set_refuse(out, m + (t.empty() ? std::string(" nothing") : " " + t.substr(3)) + " GiB");
  }
  if (in.ram_total_bytes && !in.file_tier && !out.refused && add_sat(nonres, spares) > std::min(cap, room))
    set_refuse(out, "RAM cap " + gib(in.ram_total_bytes) + " GiB: the " + std::to_string(nonres) +
                        " non-resident experts and the spare blocks do not fit the pinned room (" + gib(room_bytes) +
                        " GiB) and there is no file tier");

  // tier_plan's placement (the uncapped arithmetic, unchanged).
  const uint64_t want = add_sat(nonres, spares);
  uint64_t pinned = 0, disk = 0, staging = 0;
  bool three = false;
  if (want <= cap || !in.file_tier) {
    pinned = nonres;
  } else {
    three = true;
    staging = in.staging;
    if (cap < add_sat(add_sat(spares, staging), 1)) {
      if (!out.refused)
        set_refuse(out, "not enough host memory even for the spare blocks and the file tier's staging buffers");
      pinned = 0;
    } else {
      pinned = std::min(cap - spares - staging, nonres);
    }
    disk = nonres - pinned;
    if (disk == 0) three = false, staging = 0;
  }
  const uint64_t used = add_sat(add_sat(pinned, spares), staging);
  uint64_t left = cap > used ? cap - used : 0;
  // under a RAM cap the clean copies only take what the capped room leaves (without the file tier an explicit RAM
  // request is not clamped to the room above, and the copies would otherwise fill the request past the cap)
  if (in.ram_total_bytes) left = std::min(left, room > used ? room - used : 0);
  uint64_t inc = in.inclusive_bytes == kAuto ? units_of(in.inclusive_auto_cap_bytes) : units_of(in.inclusive_bytes);
  if (inc > left) inc = left;
  if (spares == 0) inc = 0;  // the library keeps at least one spare block for write-backs
  out.pinned = pinned;
  out.disk = disk;
  out.staging = staging;
  out.three_tier = three ? 1 : 0;
  out.inclusive = inc;
  out.spare_blocks = spares + inc;
  const uint64_t blocks = mul_sat(add_sat(add_sat(pinned, out.spare_blocks), staging), unit);
  out.host_planned_bytes = add_sat(out.host_fixed_bytes, blocks);
  out.need_host_bytes = add_sat(add_sat(blocks, extra), in.host_margin_bytes);
  out.pinned_peak_bytes = add_sat(add_sat(blocks, extra), pin.sum);
  return !out.refused;
}

// One pool (SS_BUDGET_UNIFIED). The joint placement: the most residents r (at most the arena request, at least
// min_resident) such that the device part (context + components + VRAM margin + (r + arena spares) slots) and the
// host part (components + load staging + RAM margin + the smallest host placement for keys - r non-residents: the
// spare blocks, plus the file tier's staging and one block, or every non-resident without a file tier - the same
// minimum the two-cap planner checks) fit the pool together. The pool's minimum is the smallest such sum over r (a
// plan at exactly that pool succeeds). Then the two-cap arithmetic above runs with derived caps: the device's cap
// = pool - host part (arena request r), the host's = pool - planned device bytes - VRAM margin, so the pinned tier
// and the clean copies get what the device leaves; the sum is checked against the pool at the end.
bool unified_plan(const ss_budget_in& in, ss_budget_out& out) {
  const uint64_t pool = in.vram_total_bytes && in.ram_total_bytes ? std::min(in.vram_total_bytes, in.ram_total_bytes)
                        : in.vram_total_bytes                     ? in.vram_total_bytes
                                                                  : in.ram_total_bytes;
  const Parts dev = parts(in.device_fixed, in.n_device_fixed, "device_fixed");
  const Parts pin = parts(in.pinned_fixed, in.n_pinned_fixed, "pinned_fixed");
  const Parts page = parts(in.pageable_fixed, in.n_pageable_fixed, "pageable_fixed");
  const uint64_t host_fixed = add_sat(add_sat(add_sat(pin.sum, page.sum), in.load_staging_bytes), in.ram_margin_bytes);
  const uint64_t dev_fixed = add_sat(add_sat(in.device_context_bytes, dev.sum), in.vram_margin_bytes);
  const uint64_t keys = in.keys;
  const uint64_t rq = in.arena_units == kAuto ? keys : std::min(in.arena_units, keys);
  const auto host_part = [&](uint64_t r) {
    const uint64_t nonres = keys - r;
    const uint64_t units = nonres == 0 ? in.host_spares
                           : in.file_tier ? add_sat(in.host_spares, uint64_t(in.staging) + 1)
                                          : add_sat(in.host_spares, nonres);
    return add_sat(host_fixed, mul_sat(units, in.unit_bytes));
  };
  const auto device_part = [&](uint64_t r) {
    return add_sat(dev_fixed, mul_sat(add_sat(r, in.arena_spares), in.slot_bytes));
  };
  const auto total = [&](uint64_t r) { return add_sat(device_part(r), host_part(r)); };
  // total(r) is linear on r < keys (the host part is constant with the file tier, falls by one unit per resident
  // without it) plus the point r = keys (no non-resident): its minimum is at an end, the largest fitting r is solved.
  uint64_t least = UINT64_MAX, pick = UINT64_MAX;
  const uint64_t lo = in.min_resident;
  if (lo <= rq) {
    if (rq == keys) {
      least = total(keys);
      if (least <= pool) pick = keys;
    }
    if (keys > 0 && lo <= keys - 1 && lo <= rq) {
      const uint64_t hi = std::min(rq, keys - 1);
      least = std::min(least, std::min(total(lo), total(hi)));
      if (pick == UINT64_MAX) {
        const uint64_t down = in.file_tier ? 0 : in.unit_bytes;  // host bytes one more resident saves
        if (in.slot_bytes <= down) {
          if (total(hi) <= pool) pick = hi;
        } else if (total(lo) <= pool) {
          uint64_t c = lo + std::min(hi - lo, (pool - total(lo)) / (in.slot_bytes - down));
          while (c > lo && total(c) > pool) --c;  // saturation guard
          pick = c;
        }
      }
    }
  }
  const uint64_t r = pick != UINT64_MAX ? pick : std::min(in.min_resident, rq);
  ss_budget_in a = in;
  a.unified = 0;
  a.arena_units = r;
  a.vram_total_bytes = std::max<uint64_t>(1, sub_sat(pool, host_part(r)));
  a.ram_total_bytes = 0;
  ss_budget_out d;
  budget_plan(a, d);
  a.ram_total_bytes = std::max<uint64_t>(1, sub_sat(sub_sat(pool, d.device_planned_bytes), in.vram_margin_bytes));
  budget_plan(a, out);
  out.device_capped = out.resident < rq ? 1 : 0;
  out.min_vram_total_bytes = out.min_ram_total_bytes = least;
  const uint64_t used = add_sat(add_sat(out.device_planned_bytes, in.vram_margin_bytes),
                                add_sat(out.host_planned_bytes, in.ram_margin_bytes));
  std::string why;
  if (pick == UINT64_MAX)
    why = least == UINT64_MAX ? "min_resident " + std::to_string(in.min_resident) + " is above the arena request " +
                                    std::to_string(rq)
                              : "the fixed parts and the smallest placement need " + gib_up(least) + " GiB";
  else if (!out.refused && (used == UINT64_MAX || used > pool))
    why = "the plan would use " + gib(used) + " GiB";  // a guard: the derivation above never gets here
  if (!why.empty() || out.refused) {
    const std::string m = "unified pool " + gib(pool) + " GiB (one cap for device and host): " +
                          (why.empty() ? std::string(out.refuse) : why);
    set_refuse(out, m);
  }
  return !out.refused;
}

namespace {
void json_text(std::string& s, const char* v) {
  s += '"';
  for (const char* p = v ? v : ""; *p; ++p) {
    const unsigned char c = static_cast<unsigned char>(*p);
    if (c == '"' || c == '\\') {
      s += '\\';
      s += char(c);
    } else if (c < 0x20) {
      char b[8];
      std::snprintf(b, sizeof b, "\\u%04x", c);
      s += b;
    } else {
      s += char(c);
    }
  }
  s += '"';
}
}  // namespace

std::string budget_plan_json(const ss_budget_in& in, const ss_budget_out& out) {
  std::string s;
  const auto key = [&](const char* k) {
    if (!s.empty()) s += ',';
    s += '"';
    s += k;
    s += "\":";
  };
  const auto num = [&](const char* k, uint64_t v) {
    key(k);
    s += std::to_string(v);
  };
  const auto knob = [&](const char* k, uint64_t v) {  // SS_BUDGET_AUTO = null
    key(k);
    s += v == kAuto ? std::string("null") : std::to_string(v);
  };
  const auto flag = [&](const char* k, bool v) {
    key(k);
    s += v ? "true" : "false";
  };
  const auto comps = [&](const char* k, const ss_budget_component* c, uint32_t n) {
    key(k);
    s += '{';
    for (uint32_t i = 0; c && i < n; ++i) {
      if (i) s += ',';
      json_text(s, c[i].name && *c[i].name ? c[i].name : "unnamed");
      s += ':' + std::to_string(c[i].bytes);
    }
    s += '}';
  };
  const auto minimum = [&](const char* bytes_key, const char* gib_key, const char* flag_key, uint64_t v) {
    // the bytes, GiB rounded up to 0.01 (a value that holds), then the whole-GiB flag value; null when no minimum
    // exists (UINT64_MAX: the arithmetic saturated)
    const bool none = v == UINT64_MAX;
    key(bytes_key);
    s += none ? std::string("null") : std::to_string(v);
    key(gib_key);
    s += none ? std::string("null") : gib_up(v);
    key(flag_key);
    s += none ? std::string("null") : std::to_string(gib_ceil(v));
  };
  num("vram_total_bytes", in.vram_total_bytes);
  num("ram_total_bytes", in.ram_total_bytes);
  num("vram_margin_bytes", in.vram_margin_bytes);
  num("ram_margin_bytes", in.ram_margin_bytes);
  if (in.unified && (in.vram_total_bytes || in.ram_total_bytes)) flag("unified", true);  // only when it planned
  num("device_context_bytes", in.device_context_bytes);
  comps("device_fixed", in.device_fixed, in.n_device_fixed);
  comps("pinned_fixed", in.pinned_fixed, in.n_pinned_fixed);
  comps("pageable_fixed", in.pageable_fixed, in.n_pageable_fixed);
  num("load_staging_bytes", in.load_staging_bytes);
  num("keys", in.keys);
  num("slot_bytes", in.slot_bytes);
  num("unit_bytes", in.unit_bytes);
  knob("arena_units", in.arena_units);
  num("min_resident", in.min_resident);
  knob("ram_expert_bytes", in.ram_expert_bytes);
  knob("host_avail_bytes", in.host_avail_bytes);
  num("host_margin_bytes", in.host_margin_bytes);
  knob("max_pinned_bytes", in.max_pinned_bytes);
  knob("inclusive_bytes", in.inclusive_bytes);
  num("host_spares", in.host_spares);
  num("staging_request", in.staging);
  flag("file_tier", in.file_tier != 0);
  num("resident", out.resident);
  num("arena_spares", out.arena_spares);
  num("arena_slots", out.arena_slots);
  num("pinned", out.pinned);
  num("disk", out.disk);
  num("inclusive", out.inclusive);
  num("spare_blocks", out.spare_blocks);
  num("staging", out.staging);
  flag("three_tier", out.three_tier != 0);
  flag("ram_auto", out.ram_auto != 0);
  num("device_fixed_bytes", out.device_fixed_bytes);
  num("device_planned_bytes", out.device_planned_bytes);
  num("device_room_bytes", out.device_room_bytes);
  flag("device_capped", out.device_capped != 0);
  num("host_fixed_bytes", out.host_fixed_bytes);
  num("host_planned_bytes", out.host_planned_bytes);
  num("room_bytes", out.room_bytes);
  num("host_cap_units", out.host_cap_units);
  flag("host_capped", out.host_capped != 0);
  num("need_host_bytes", out.need_host_bytes);
  num("pinned_peak_bytes", out.pinned_peak_bytes);
  minimum("min_vram_total_bytes", "min_vram_total_gib", "min_vram_total_flag_gib", out.min_vram_total_bytes);
  minimum("min_ram_total_bytes", "min_ram_total_gib", "min_ram_total_flag_gib", out.min_ram_total_bytes);
  flag("refused", out.refused != 0);
  key("refuse");
  char r[SS_BUDGET_REFUSE + 1];
  std::memcpy(r, out.refuse, SS_BUDGET_REFUSE);
  r[SS_BUDGET_REFUSE] = '\0';
  json_text(s, r);
  return s;
}

ss_budget_ledger* ledger_create(uint64_t vram_total_bytes, uint64_t ram_total_bytes) {
  auto* l = new ss_budget_ledger;
  l->cap[SS_BUDGET_DEVICE] = vram_total_bytes;
  l->cap[SS_BUDGET_PINNED] = ram_total_bytes;
  l->cap[SS_BUDGET_PAGEABLE] = ram_total_bytes;
  return l;
}

void ledger_free(ss_budget_ledger* l) { delete l; }

namespace {
void check_kind(const ss_budget_ledger* l, ss_budget_kind kind, const char* fn) {
  require(l != nullptr, SS_E_INVALID, std::string(fn) + ": null ledger");
  require(int(kind) >= 0 && int(kind) < SS_BUDGET_KINDS, SS_E_INVALID, std::string(fn) + ": unknown kind");
}
}  // namespace

void ledger_add(ss_budget_ledger* l, ss_budget_kind kind, const char* name, uint64_t bytes) {
  check_kind(l, kind, "ss_budget_ledger_add");
  const std::string n = name && *name ? name : "unnamed";
  std::lock_guard<std::mutex> lock(l->mu);
  const size_t k = size_t(kind);
  const bool host = kind != SS_BUDGET_DEVICE;
  const uint64_t in_use = host ? add_sat(l->cur[SS_BUDGET_PINNED], l->cur[SS_BUDGET_PAGEABLE]) : l->cur[k];
  if (l->cap[k] && (bytes > l->cap[k] || in_use > l->cap[k] - bytes)) {
    ++l->refusals;
    fail(SS_E_CAPACITY, std::string("budget: ") + kind_name(kind) + " allocation '" + n + "' of " + gib(bytes) +
                            " GiB (" + std::to_string(bytes) + " bytes) would exceed the " + (host ? "RAM" : "VRAM") +
                            " cap: " + gib(in_use) + " of " + gib(l->cap[k]) + " GiB in use");
  }
  // no cap: the counts must still fit (a component's and a kind's sum are at most the host or device sum)
  require(in_use <= UINT64_MAX - bytes, SS_E_INVALID,
          std::string("ss_budget_ledger_add: ") + kind_name(kind) + " allocation '" + n + "' overflows the ledger");
  l->cur[k] += bytes;
  l->comp[k][n] += bytes;
  l->host_peak = std::max(l->host_peak, l->cur[SS_BUDGET_PINNED] + l->cur[SS_BUDGET_PAGEABLE]);
  if (l->cur[k] > l->peak[k] || (l->cur[k] == l->peak[k] && l->top[k].empty())) {
    l->peak[k] = l->cur[k];
    auto& top = l->top[k];
    top.assign(l->comp[k].begin(), l->comp[k].end());
    const size_t keep = std::min<size_t>(top.size(), SS_BUDGET_TOP);
    std::partial_sort(top.begin(), top.begin() + std::ptrdiff_t(keep), top.end(), [](const auto& a, const auto& b) {
      return a.second > b.second || (a.second == b.second && a.first < b.first);
    });
    top.resize(keep);
  }
}

void ledger_sub(ss_budget_ledger* l, ss_budget_kind kind, const char* name, uint64_t bytes) {
  check_kind(l, kind, "ss_budget_ledger_sub");
  const std::string n = name && *name ? name : "unnamed";
  std::lock_guard<std::mutex> lock(l->mu);
  const size_t k = size_t(kind);
  auto it = l->comp[k].find(n);
  const uint64_t held = it == l->comp[k].end() ? 0 : it->second;
  require(bytes <= held, SS_E_INVALID,
          std::string("ss_budget_ledger_sub: ") + kind_name(kind) + " component '" + n + "' holds " +
              std::to_string(held) + " bytes, " + std::to_string(bytes) + " released");
  if (it == l->comp[k].end()) return;  // 0 of nothing
  it->second -= bytes;
  if (it->second == 0) l->comp[k].erase(it);
  l->cur[k] -= bytes;
}

namespace {
// one side's sample (the lock held): last, peak with its stage (the first sample sets it), above the cap
void take_sample(uint64_t v, uint64_t cap, const char* stage, uint64_t& last, uint64_t& peak, uint64_t& count,
                 uint32_t& over, std::string& at) {
  if (count == 0 || v > peak) {
    peak = v;
    at = stage ? stage : "";
  }
  last = v;
  ++count;
  if (cap && v > cap) ++over;
}
}  // namespace

void ledger_observe_device(ss_budget_ledger* l, uint64_t used) {
  require(l != nullptr, SS_E_INVALID, "ss_budget_ledger_observe_device: null ledger");
  std::lock_guard<std::mutex> lock(l->mu);
  take_sample(used, l->cap[SS_BUDGET_DEVICE], nullptr, l->measured_last, l->measured_peak, l->observations,
              l->device_over, l->device_peak_at);
}

void ledger_observe(ss_budget_ledger* l, const char* stage, uint64_t device_used, uint64_t host_used) {
  require(l != nullptr, SS_E_INVALID, "ss_budget_ledger_observe: null ledger");
  std::lock_guard<std::mutex> lock(l->mu);
  if (device_used != SS_BUDGET_NO_SAMPLE)
    take_sample(device_used, l->cap[SS_BUDGET_DEVICE], stage, l->measured_last, l->measured_peak, l->observations,
                l->device_over, l->device_peak_at);
  if (host_used != SS_BUDGET_NO_SAMPLE)
    take_sample(host_used, l->cap[SS_BUDGET_PINNED], stage, l->host_last, l->host_measured_peak,
                l->host_observations, l->host_over, l->host_peak_at);
}

namespace {
bool within_caps_locked(const ss_budget_ledger* l) {
  const uint64_t vcap = l->cap[SS_BUDGET_DEVICE], rcap = l->cap[SS_BUDGET_PINNED];
  return (!vcap || (l->measured_peak <= vcap && l->device_over == 0)) &&
         (!rcap || (l->host_measured_peak <= rcap && l->host_over == 0));
}
}  // namespace

bool ledger_within_caps(const ss_budget_ledger* l) {
  require(l != nullptr, SS_E_INVALID, "ss_budget_ledger_within_caps: null ledger");
  auto* m = const_cast<ss_budget_ledger*>(l);  // the mutex only
  std::lock_guard<std::mutex> lock(m->mu);
  return within_caps_locked(l);
}

ss_budget_stats ledger_stats(const ss_budget_ledger* l) {
  require(l != nullptr, SS_E_INVALID, "ss_budget_ledger_stats: null ledger");
  auto* m = const_cast<ss_budget_ledger*>(l);  // the mutex only
  std::lock_guard<std::mutex> lock(m->mu);
  ss_budget_stats s{};
  s.struct_size = uint32_t(sizeof s);
  s.refusals = l->refusals;
  for (size_t k = 0; k < SS_BUDGET_KINDS; ++k) {
    ss_budget_kind_stats& o = s.kind[k];
    o.cap = l->cap[k];
    o.current = l->cur[k];
    o.peak = l->peak[k];
    o.n_top = uint32_t(l->top[k].size());
    for (size_t i = 0; i < l->top[k].size(); ++i) {
      std::snprintf(o.top[i].name, sizeof o.top[i].name, "%s", l->top[k][i].first.c_str());
      o.top[i].bytes = l->top[k][i].second;
    }
  }
  s.host_current = l->cur[SS_BUDGET_PINNED] + l->cur[SS_BUDGET_PAGEABLE];
  s.host_peak = l->host_peak;
  s.measured_device_last = l->measured_last;
  s.measured_device_peak = l->measured_peak;
  s.observations = l->observations;
  s.measured_host_last = l->host_last;
  s.measured_host_peak = l->host_measured_peak;
  s.host_observations = l->host_observations;
  s.device_over = l->device_over;
  s.host_over = l->host_over;
  std::snprintf(s.measured_device_peak_at, sizeof s.measured_device_peak_at, "%s", l->device_peak_at.c_str());
  std::snprintf(s.measured_host_peak_at, sizeof s.measured_host_peak_at, "%s", l->host_peak_at.c_str());
  s.within_caps = within_caps_locked(l) ? 1 : 0;
  return s;
}

void BudgetCharges::charge(ss_budget_kind kind, const char* name, uint64_t bytes) {
  if (!l_) return;
  ledger_add(l_, kind, name, bytes);
  charges_.push_back(Charge{kind, name, bytes});
}

void BudgetCharges::release() noexcept {
  for (auto it = charges_.rbegin(); it != charges_.rend(); ++it) {
    try {
      ledger_sub(l_, it->kind, it->name.c_str(), it->bytes);
    } catch (...) {  // the owner released it by hand: nothing to give back
    }
  }
  charges_.clear();
}

namespace {

class BudgetBackend final : public Backend {
 public:
  BudgetBackend(std::unique_ptr<Backend> inner, ss_budget_ledger* l) : in_(std::move(inner)), l_(l) {}
  ~BudgetBackend() override {
    for (const auto& a : live_) {  // left allocated by the context (or freed through another path): give back
      try {
        ledger_sub(l_, a.second.first, kind_label(a.second.first), a.second.second);
      } catch (...) {
      }
    }
  }
  const char* name() const override { return in_->name(); }
  bool unified() const override { return in_->unified(); }
  void bind_thread() override { in_->bind_thread(); }
  void* alloc_device(size_t bytes) override { return alloc(SS_BUDGET_DEVICE, bytes); }
  void free_device(void* p) override {
    forget(p);
    in_->free_device(p);
  }
  void* alloc_pinned(size_t bytes) override { return alloc(SS_BUDGET_PINNED, bytes); }
  void free_pinned(void* p) override {
    forget(p);
    in_->free_pinned(p);
  }
  void* event_create() override { return in_->event_create(); }
  void event_destroy(void* ev) override { in_->event_destroy(ev); }
  void event_record(void* ev, void* stream) override { in_->event_record(ev, stream); }
  void event_sync(void* ev) override { in_->event_sync(ev); }
  void stream_wait(void* stream, void* ev) override { in_->stream_wait(stream, ev); }
  void copy_h2d(void* dst, const void* src, size_t n, void* s) override { in_->copy_h2d(dst, src, n, s); }
  void copy_d2h(void* dst, const void* src, size_t n, void* s) override { in_->copy_d2h(dst, src, n, s); }
  void copy_d2d(void* dst, const void* src, size_t n, void* s) override { in_->copy_d2d(dst, src, n, s); }
  void stream_sync(void* stream) override { in_->stream_sync(stream); }
  void* writeback_stream() override { return in_->writeback_stream(); }

 private:
  static const char* kind_label(ss_budget_kind k) {
    return k == SS_BUDGET_DEVICE ? "tierstream device" : "tierstream pinned";
  }
  void* alloc(ss_budget_kind kind, size_t bytes) {
    ledger_add(l_, kind, kind_label(kind), uint64_t(bytes));
    void* p = nullptr;
    try {
      p = kind == SS_BUDGET_DEVICE ? in_->alloc_device(bytes) : in_->alloc_pinned(bytes);
    } catch (...) {
      ledger_sub(l_, kind, kind_label(kind), uint64_t(bytes));
      throw;
    }
    std::lock_guard<std::mutex> lock(mu_);
    live_[p] = {kind, uint64_t(bytes)};
    return p;
  }
  void forget(void* p) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = live_.find(p);
    if (it == live_.end()) return;
    try {
      ledger_sub(l_, it->second.first, kind_label(it->second.first), it->second.second);
    } catch (...) {
    }
    live_.erase(it);
  }

  std::unique_ptr<Backend> in_;
  ss_budget_ledger* l_;
  std::mutex mu_;
  std::unordered_map<void*, std::pair<ss_budget_kind, uint64_t>> live_;
};

}  // namespace

std::unique_ptr<Backend> budget_backend(std::unique_ptr<Backend> inner, ss_budget_ledger* l) {
  if (!l) return inner;
  return std::make_unique<BudgetBackend>(std::move(inner), l);
}

}  // namespace ts
