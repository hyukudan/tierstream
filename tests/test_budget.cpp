// Whole-device budgets (ss.h "Whole-device budgets"): the planner against the engines' three-tier arithmetic it
// replaces (DeepSeek's rtx::tier_plan, copied below unchanged as the oracle), the capped rules and refusals, the
// ledger (peaks, components at the peak, caps, threads), sized-struct compatibility, and a host-mode context charging
// its allocations to a ledger on the CPU backend.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <string>
#include <random>
#include <thread>
#include <vector>

#include "check.hpp"
#include "fixture.hpp"
#include "strix_stream/ss.h"

namespace oracle {
// ---- copied from the DeepSeek engine (include/strix/ds41_rtx_schedule.hpp, rtx::tier_plan), unchanged ----
struct TierPlanIn {
    std::uint64_t keys = 0;        // routed experts (layers x experts)
    std::uint64_t unit_bytes = 0;  // one expert's host stride
    std::uint64_t residents = 0;   // device experts (the arena's slots minus its spares)
    std::uint32_t host_spares = 0; // free pinned blocks for write-backs (--host-spares)
    double ram_gib = -1.0;         // --ram-expert-gib: < 0 = auto (what fits: available - margin, max pinned)
    double inclusive_gib = 0.0;    // --ram-inclusive-gib: < 0 = auto (RAM left over, at most inclusive_auto_cap_gib), 0 = off
    double inclusive_auto_cap_gib = 16.0;
    double host_avail_gib = 0.0, host_margin_gib = 0.0;  // available host memory now; the safety margin kept free
    double engine_pinned_gib = 0.0;                      // the engine's own pinned buffers (outside the expert tiers)
    double max_pinned_gib = 0.0;                         // --max-pinned-gib (engine + experts)
    double extra_gib = 0.0;                              // load staging the budget always adds (16 x down part)
    bool file_tier = true;         // false: no file tier (the pinned tier must hold every non-resident expert)
    std::uint32_t staging = 8;     // file reads' pinned staging buffers (three-tier only)
};
struct TierPlan {
    std::uint64_t resident = 0, pinned = 0, disk = 0;  // experts per tier
    std::uint64_t inclusive = 0;     // pinned blocks device experts may keep as clean copies (out of spare_blocks)
    std::uint64_t spare_blocks = 0;  // host_spares + inclusive (the library's host_spare_units)
    std::uint64_t staging = 0;       // staging buffers (three-tier)
    double vram_gib = 0, ram_gib = 0, file_gib = 0, inclusive_gib = 0;
    double room_gib = 0;             // pinned bytes the host can give the expert tiers now (auto's ceiling)
    double need_host_gib = 0;        // available host memory the load needs (blocks + extra + margin), as before
    double pinned_peak_gib = 0;      // engine + expert pinned bytes at peak, as before
    bool three_tier = false;         // disk > 0: the library's file tier is on
    bool ram_auto = true;
    const char* refuse = nullptr;    // set: the load cannot go on (the message)
};
inline TierPlan tier_plan(const TierPlanIn& in) {
    constexpr double gib = 1073741824.0;
    TierPlan p;
    p.ram_auto = in.ram_gib < 0.0;
    const double unit = static_cast<double>(in.unit_bytes) / gib;
    p.resident = in.residents < in.keys ? in.residents : in.keys;
    const std::uint64_t nonres = in.keys - p.resident;
    const auto units_of = [&](double g) -> std::uint64_t { return g > 0.0 && unit > 0.0 ? static_cast<std::uint64_t>(g / unit) : 0U; };
    const double room_host = in.host_avail_gib - in.host_margin_gib - in.extra_gib;
    const double room_pin = in.max_pinned_gib - in.engine_pinned_gib - in.extra_gib;
    p.room_gib = room_host < room_pin ? room_host : room_pin;
    const std::uint64_t room = units_of(p.room_gib);
    std::uint64_t cap = p.ram_auto ? room : units_of(in.ram_gib);
    if (in.file_tier && cap > room) cap = room;  // never more than the host can pin now
    const std::uint64_t want = nonres + in.host_spares;
    if (want <= cap || !in.file_tier) {
        p.pinned = nonres;  // the two-tier placement (without the file tier the budget checks refuse as before)
    } else {
        p.three_tier = true;
        p.staging = in.staging;
        if (cap < std::uint64_t{in.host_spares} + p.staging + 1U) {
            p.refuse = "not enough host memory even for the spare blocks and the file tier's staging buffers";
            p.pinned = 0;
        } else {
            p.pinned = cap - in.host_spares - p.staging;
            if (p.pinned > nonres) p.pinned = nonres;
        }
        p.disk = nonres - p.pinned;
        if (p.disk == 0) p.three_tier = false, p.staging = 0;
    }
    const std::uint64_t used = p.pinned + in.host_spares + p.staging;
    const std::uint64_t left = cap > used ? cap - used : 0U;
    std::uint64_t inc = 0;
    if (in.inclusive_gib < 0.0) inc = units_of(in.inclusive_auto_cap_gib);
    else if (in.inclusive_gib > 0.0) inc = units_of(in.inclusive_gib);
    if (inc > left) inc = left;
    if (in.host_spares == 0) inc = 0;  // the library keeps at least one spare block for write-backs
    p.inclusive = inc;
    p.spare_blocks = std::uint64_t{in.host_spares} + inc;
    p.vram_gib = static_cast<double>(p.resident) * unit;
    p.ram_gib = static_cast<double>(p.pinned) * unit;
    p.file_gib = static_cast<double>(p.disk) * unit;
    p.inclusive_gib = static_cast<double>(p.inclusive) * unit;
    p.need_host_gib = static_cast<double>(p.pinned + p.spare_blocks + p.staging) * unit + in.extra_gib + in.host_margin_gib;
    p.pinned_peak_gib = p.need_host_gib - in.host_margin_gib + in.engine_pinned_gib;
    return p;
}
// ---- end of the copy ----
}  // namespace oracle

namespace {

constexpr uint64_t kGiB = uint64_t(1) << 30;
constexpr double kGiBd = 1073741824.0;
constexpr uint64_t kAuto = SS_BUDGET_AUTO;

// One uncapped case in bytes: the oracle reads it in GiB (bytes / 2^30), the planner in bytes. -1 = auto.
struct Case {
  uint64_t keys, unit, residents;
  uint32_t spares;
  int64_t ram, inclusive;  // bytes; -1 = auto; inclusive 0 = off
  uint64_t inc_cap, avail, margin, engine_pinned, max_pinned, extra;
  bool file_tier;
  uint32_t staging;
};

oracle::TierPlanIn to_oracle(const Case& c) {
  oracle::TierPlanIn in;
  in.keys = c.keys;
  in.unit_bytes = c.unit;
  in.residents = c.residents;
  in.host_spares = c.spares;
  in.ram_gib = c.ram < 0 ? -1.0 : double(c.ram) / kGiBd;
  in.inclusive_gib = c.inclusive < 0 ? -1.0 : double(c.inclusive) / kGiBd;
  in.inclusive_auto_cap_gib = double(c.inc_cap) / kGiBd;
  in.host_avail_gib = double(c.avail) / kGiBd;
  in.host_margin_gib = double(c.margin) / kGiBd;
  in.engine_pinned_gib = double(c.engine_pinned) / kGiBd;
  in.max_pinned_gib = double(c.max_pinned) / kGiBd;
  in.extra_gib = double(c.extra) / kGiBd;
  in.file_tier = c.file_tier;
  in.staging = c.staging;
  return in;
}

// The migration: tier_plan's inputs as ss_budget_in (no caps).
ss_budget_in to_budget(const Case& c, ss_budget_component* engine_pinned) {
  ss_budget_in in;
  ss_budget_in_defaults(&in);
  in.keys = c.keys;
  in.unit_bytes = c.unit;
  in.slot_bytes = c.unit;
  in.arena_units = c.residents;
  in.host_spares = c.spares;
  in.ram_expert_bytes = c.ram < 0 ? kAuto : uint64_t(c.ram);
  in.inclusive_bytes = c.inclusive < 0 ? kAuto : uint64_t(c.inclusive);
  in.inclusive_auto_cap_bytes = c.inc_cap;
  in.host_avail_bytes = c.avail;
  in.host_margin_bytes = c.margin;
  *engine_pinned = ss_budget_component{"engine pinned", c.engine_pinned};
  in.pinned_fixed = engine_pinned;
  in.n_pinned_fixed = 1;
  in.max_pinned_bytes = c.max_pinned;
  in.load_staging_bytes = c.extra;
  in.file_tier = c.file_tier ? 1 : 0;
  in.staging = c.staging;
  return in;
}

bool near(double a, double b) { return std::fabs(a - b) <= 1e-6 * (1.0 + std::fabs(b)); }

// Every output tier_plan has, equal; returns the number of differences (printed).
int compare(const Case& c, const char* label) {
  const oracle::TierPlan o = oracle::tier_plan(to_oracle(c));
  ss_budget_component ep;
  const ss_budget_in in = to_budget(c, &ep);
  ss_budget_out out;
  out.struct_size = sizeof out;
  const ss_status st = ss_budget_plan(&in, &out);
  int bad = 0;
  auto diff = [&](bool ok, const char* what) {
    if (!ok) {
      std::printf("  case %s: %s differs\n", label, what);
      ++bad;
    }
  };
  diff((st == SS_OK) == (o.refuse == nullptr), "status");
  diff((out.refused != 0) == (o.refuse != nullptr), "refused");
  diff(!o.refuse || std::strcmp(out.refuse, o.refuse) == 0, "refusal message");
  diff(out.resident == o.resident, "resident");
  diff(out.pinned == o.pinned, "pinned");
  diff(out.disk == o.disk, "disk");
  diff(out.inclusive == o.inclusive, "inclusive");
  diff(out.spare_blocks == o.spare_blocks, "spare_blocks");
  diff(out.staging == o.staging, "staging");
  diff((out.three_tier != 0) == o.three_tier, "three_tier");
  diff((out.ram_auto != 0) == o.ram_auto, "ram_auto");
  diff(near(double(out.room_bytes) / kGiBd, o.room_gib > 0 ? o.room_gib : 0.0), "room");
  diff(near(double(out.need_host_bytes) / kGiBd, o.need_host_gib), "need_host");
  diff(near(double(out.pinned_peak_bytes) / kGiBd, o.pinned_peak_gib), "pinned_peak");
  diff(near(double(out.resident) * double(c.unit) / kGiBd, o.vram_gib), "vram_gib");
  diff(near(double(out.pinned) * double(c.unit) / kGiBd, o.ram_gib), "ram_gib");
  diff(near(double(out.disk) * double(c.unit) / kGiBd, o.file_gib), "file_gib");
  CHECK_EQ(bad, 0);
  return bad;
}

uint64_t g(double gib) { return uint64_t(gib * kGiBd); }

// The DeepSeek production shape (40 x 384 experts of 9953280 bytes, 74.8 GiB arena, 48 host spares).
Case production() {
  Case c{};
  c.keys = 40ULL * 384ULL;
  c.unit = 9953280ULL;
  c.residents = uint64_t(74.8 * kGiBd / 9953280.0);
  c.spares = 48;
  c.ram = -1;
  c.inclusive = 0;
  c.inc_cap = 16 * kGiB;
  c.margin = 6 * kGiB;
  c.engine_pinned = g(0.6);
  c.max_pinned = 74 * kGiB;
  c.extra = 16ULL * 3870720ULL;
  c.file_tier = true;
  c.staging = 8;
  return c;
}

}  // namespace

// tier_plan's own test cases (a)-(g) of ds41_rtx_schedule_tests.cpp, in its order, then a grid: identical outputs.
TEST(uncapped_equals_tier_plan_on_its_cases) {
  Case c = production();
  c.avail = 90 * kGiB;
  compare(c, "a: enough RAM");
  c.avail = 40 * kGiB;
  compare(c, "b: RAM short, file tier");
  c.file_tier = false;
  compare(c, "c: no file tier");
  c.file_tier = true;
  c.avail = 120 * kGiB;
  c.max_pinned = 100 * kGiB;
  c.residents = uint64_t(48.0 * kGiBd / 9953280.0);
  c.ram = int64_t(64 * kGiB);
  compare(c, "d: explicit budgets");
  c.ram = int64_t(500 * kGiB);
  compare(c, "e: explicit RAM above the room");
  c.ram = -1;
  c.residents = uint64_t(74.8 * kGiBd / 9953280.0);
  c.avail = 200 * kGiB;
  c.max_pinned = 200 * kGiB;
  c.inclusive = int64_t(8 * kGiB);
  compare(c, "f: inclusive 8 GiB");
  c.inclusive = -1;
  compare(c, "f2: inclusive auto");
  c.avail = 40 * kGiB;
  compare(c, "f3: RAM short, inclusive auto");
  c.avail = g(6.2);
  compare(c, "g: host cannot hold the spares");
}

TEST(uncapped_equals_tier_plan_on_a_grid) {
  int cases = 0, bad = 0;
  const uint64_t units[] = {9953280ULL, 3 * 4096 + 1, 2621440ULL};
  const uint64_t keys_v[] = {64, 15360};
  for (uint64_t unit : units)
    for (uint64_t keys : keys_v)
      for (double res_frac : {0.0, 0.25, 0.9, 1.0, 1.5})
        for (uint32_t spares : {0u, 1u, 48u})
          for (int ram_mode = 0; ram_mode < 3; ++ram_mode)
            for (int inc_mode = 0; inc_mode < 3; ++inc_mode)
              for (double avail : {0.0, 0.3, 7.7, 40.0, 150.0})
                for (bool file_tier : {true, false})
                  for (uint32_t staging : {8u, 2u}) {
                    Case c{};
                    c.keys = keys;
                    c.unit = unit;
                    c.residents = uint64_t(double(keys) * res_frac);
                    c.spares = spares;
                    c.ram = ram_mode == 0 ? -1 : ram_mode == 1 ? int64_t(g(5.5)) : int64_t(g(90.25));
                    c.inclusive = inc_mode == 0 ? 0 : inc_mode == 1 ? -1 : int64_t(g(1.75));
                    c.inc_cap = (keys == 64 ? 1 : 16) * kGiB;
                    c.avail = g(avail);
                    c.margin = g(avail > 10 ? 6.0 : 0.1);
                    c.engine_pinned = g(0.6);
                    c.max_pinned = g(avail > 100 ? 120.0 : 74.0);
                    c.extra = 16ULL * 3870720ULL;
                    c.file_tier = file_tier;
                    c.staging = staging;
                    ++cases;
                    bad += compare(c, "grid") ? 1 : 0;
                  }
  std::printf("  %d grid cases, %d different\n", cases, bad);
  CHECK(cases > 5000);
}

namespace {

// A 24 GiB card: CUDA context, a 14.4 GB trunk, KV for the context, workspaces; DeepSeek-like experts.
struct Card {
  ss_budget_component dev[3] = {{"trunk", 14400000000ULL}, {"kv", 2 * kGiB}, {"workspaces", 512ULL << 20}};
  ss_budget_component pin[1] = {{"engine pinned", 600ULL << 20}};
  ss_budget_component page[1] = {{"host weights", 4 * kGiB}};
  ss_budget_in in;
  Card() {
    ss_budget_in_defaults(&in);
    in.vram_total_bytes = 24 * kGiB;
    in.device_context_bytes = 600ULL << 20;
    in.device_fixed = dev;
    in.n_device_fixed = 3;
    in.pinned_fixed = pin;
    in.n_pinned_fixed = 1;
    in.pageable_fixed = page;
    in.n_pageable_fixed = 1;
    in.keys = 40ULL * 384ULL;
    in.slot_bytes = 9953280ULL;
    in.unit_bytes = 9953280ULL;
    in.arena_spares = 96;
    in.host_spares = 48;
    in.load_staging_bytes = 16ULL * 3870720ULL;
  }
};

}  // namespace

TEST(vram_cap_gives_the_arena_the_rest) {
  Card k;
  ss_budget_out out;
  out.struct_size = sizeof out;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  const uint64_t fixed = (600ULL << 20) + 14400000000ULL + 2 * kGiB + (512ULL << 20);
  const uint64_t slots = (24 * kGiB - fixed) / 9953280ULL;
  CHECK_EQ(out.device_fixed_bytes, fixed);
  CHECK_EQ(out.arena_slots, slots);
  CHECK_EQ(out.resident, slots - 96);
  CHECK_EQ(out.arena_spares, uint64_t(96));
  CHECK(out.device_planned_bytes <= 24 * kGiB);
  CHECK(out.device_planned_bytes + 9953280ULL > 24 * kGiB);  // one more slot would not fit
  CHECK_EQ(out.min_vram_total_bytes, fixed + 96 * 9953280ULL);
  // no RAM cap and no host limit: every other expert pinned (the two-tier placement)
  CHECK_EQ(out.pinned, k.in.keys - out.resident);
  CHECK_EQ(out.disk, uint64_t(0));
  CHECK_EQ(out.refused, 0u);
  CHECK_EQ(std::string(out.refuse), std::string());
  // an explicit arena below what the cap allows is kept; above it, the cap wins
  k.in.arena_units = 100;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.resident, uint64_t(100));
  k.in.arena_units = 1000000;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.resident, slots - 96);
  // the same plan uncapped: every expert resident (auto), as today
  k.in.arena_units = kAuto;
  k.in.vram_total_bytes = 0;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.resident, k.in.keys);
}

TEST(vram_cap_below_the_fixed_parts_is_refused_with_the_minimum) {
  Card k;
  k.in.vram_total_bytes = 12 * kGiB;  // the trunk alone is 13.41 GiB
  k.in.min_resident = 1000;
  ss_budget_out out;
  out.struct_size = sizeof out;
  CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
  CHECK_EQ(out.refused, 1u);
  const uint64_t need = (600ULL << 20) + 14400000000ULL + 2 * kGiB + (512ULL << 20) + (96 + 1000) * 9953280ULL;
  CHECK_EQ(out.min_vram_total_bytes, need);
  const std::string m = out.refuse;
  std::printf("  %s\n", m.c_str());
  CHECK_EQ(std::string(ss_last_error()), m);
  for (const char* part : {"VRAM cap 12.00 GiB", "context", "trunk 13.41", "kv 2.00", "workspaces 0.50",
                           "arena spares", "minimal arena", "--vram-total-gib 27"})
    CHECK(m.find(part) != std::string::npos);
  CHECK_EQ(out.resident, uint64_t(0));
  // the minimum it names is enough, one byte less is not
  k.in.vram_total_bytes = need;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.resident, uint64_t(1000));
  k.in.vram_total_bytes = need - 1;
  CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
}

TEST(ram_cap_leaves_the_rest_to_the_file_tier) {
  Card k;
  k.in.ram_total_bytes = 64 * kGiB;
  ss_budget_out out;
  out.struct_size = sizeof out;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  const uint64_t host_fixed = (600ULL << 20) + 4 * kGiB + 16ULL * 3870720ULL;
  CHECK_EQ(out.host_fixed_bytes, host_fixed);
  const uint64_t units = (64 * kGiB - host_fixed) / 9953280ULL;
  const uint64_t nonres = k.in.keys - out.resident;
  CHECK_EQ(out.three_tier, 1u);
  CHECK_EQ(out.staging, uint64_t(8));
  CHECK_EQ(out.pinned, units - 48 - 8);
  CHECK_EQ(out.pinned + out.disk, nonres);
  CHECK(out.host_planned_bytes <= 64 * kGiB);
  CHECK_EQ(out.min_ram_total_bytes, host_fixed + (48 + 8 + 1) * 9953280ULL);
  // an explicit RAM request above what the cap leaves is clamped to it, below it is kept
  k.in.ram_expert_bytes = 500 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.pinned, units - 48 - 8);
  k.in.ram_expert_bytes = 20 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.pinned, (20 * kGiB) / 9953280ULL - 48 - 8);
  // and what the host can pin now still bounds it (the uncapped rule)
  k.in.ram_expert_bytes = kAuto;
  k.in.host_avail_bytes = 30 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.pinned, (30 * kGiB - 16ULL * 3870720ULL) / 9953280ULL - 48 - 8);
  // a RAM cap below the fixed host parts: refused, naming them and the minimum
  k.in.host_avail_bytes = kAuto;
  k.in.ram_total_bytes = 4 * kGiB;
  CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
  const std::string m = out.refuse;
  std::printf("  %s\n", m.c_str());
  for (const char* part : {"RAM cap 4.00 GiB", "engine pinned", "host weights 4.00", "load staging", "spare blocks",
                           "--ram-total-gib 6"})
    CHECK(m.find(part) != std::string::npos);
}

TEST(no_file_tier_and_not_enough_ram_is_refused) {
  Card k;
  k.in.ram_total_bytes = 64 * kGiB;
  k.in.file_tier = 0;
  ss_budget_out out;
  out.struct_size = sizeof out;
  CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
  const std::string m = out.refuse;
  std::printf("  %s\n", m.c_str());
  CHECK(m.find("no file tier") != std::string::npos);
  CHECK(m.find("--ram-total-gib") != std::string::npos);
  CHECK_EQ(out.disk, uint64_t(0));
  // enough RAM for every non-resident expert: accepted, two-tier
  k.in.ram_total_bytes = 200 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.three_tier, 0u);
  CHECK_EQ(out.pinned, k.in.keys - out.resident);
  CHECK(out.min_ram_total_bytes <= 200 * kGiB);
  // with no RAM cap the plan is tier_plan's (no refusal: the budget checks decide, as before)
  k.in.ram_total_bytes = 0;
  k.in.host_avail_bytes = 10 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
}

TEST(budget_plan_arguments_and_saturation) {
  ss_budget_in in;
  ss_budget_in_defaults(&in);
  ss_budget_out out;
  out.struct_size = sizeof out;
  CHECK_EQ(int(ss_budget_plan(nullptr, &out)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_budget_plan(&in, nullptr)), int(SS_E_INVALID));
  in.n_device_fixed = 1;  // a count without its array
  CHECK_EQ(int(ss_budget_plan(&in, &out)), int(SS_E_INVALID));
  in.n_device_fixed = 0;
  // huge components saturate instead of wrapping: refused, never a small total
  ss_budget_component huge[2] = {{"a", UINT64_MAX - 5}, {nullptr, 100}};
  in.device_fixed = huge;
  in.n_device_fixed = 2;
  in.vram_total_bytes = 24 * kGiB;
  in.keys = 10;
  in.slot_bytes = in.unit_bytes = 1 << 20;
  CHECK_EQ(int(ss_budget_plan(&in, &out)), int(SS_E_CAPACITY));
  CHECK_EQ(out.min_vram_total_bytes, UINT64_MAX);
  CHECK(std::string(out.refuse).find("unnamed") != std::string::npos);
  // nothing to place: no experts, no refusal
  ss_budget_in_defaults(&in);
  CHECK_OK(ss_budget_plan(&in, &out));
  CHECK_EQ(out.resident + out.pinned + out.disk, uint64_t(0));
}

TEST(sized_structs_older_and_smaller_callers) {
  // defaults into a smaller struct: only its bytes are written
  struct {
    ss_budget_in in;
    unsigned char tail[8];
  } s;
  std::memset(&s, 0xAB, sizeof s);
  const uint32_t old_size = uint32_t(offsetof(ss_budget_in, min_resident));
  ss_budget_in_defaults_sized(&s.in, old_size);
  CHECK_EQ(s.in.struct_size, old_size);
  CHECK_EQ(s.in.staging, 8u);
  CHECK_EQ(reinterpret_cast<unsigned char*>(&s.in)[old_size], 0xAB);
  // an older caller's input (no min_resident): read up to its struct_size, the rest defaulted (min_resident = 0)
  Card k;
  ss_budget_in old;
  std::memset(&old, 0xCD, sizeof old);
  std::memcpy(&old, &k.in, old_size);
  old.struct_size = old_size;
  ss_budget_out out;
  out.struct_size = sizeof out;
  CHECK_OK(ss_budget_plan(&old, &out));
  ss_budget_out ref;
  ref.struct_size = sizeof ref;
  CHECK_OK(ss_budget_plan(&k.in, &ref));
  CHECK_EQ(out.resident, ref.resident);
  CHECK_EQ(out.min_vram_total_bytes, ref.min_vram_total_bytes);
  // a smaller output struct: the library writes only its bytes
  struct {
    ss_budget_out out;
    unsigned char tail[8];
  } o;
  std::memset(&o, 0xEE, sizeof o);
  const uint32_t small = uint32_t(offsetof(ss_budget_out, min_vram_total_bytes));
  o.out.struct_size = small;
  CHECK_OK(ss_budget_plan(&k.in, &o.out));
  CHECK_EQ(o.out.struct_size, small);
  CHECK_EQ(o.out.resident, ref.resident);
  CHECK_EQ(reinterpret_cast<unsigned char*>(&o.out)[small], 0xEE);
  o.out.struct_size = 4;  // below the two leading words
  CHECK_EQ(int(ss_budget_plan(&k.in, &o.out)), int(SS_E_INVALID));
  // ledger stats into a smaller struct
  ss_budget_ledger* l = nullptr;
  CHECK_OK(ss_budget_ledger_create(0, 0, &l));
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "x", 5));
  struct {
    ss_budget_stats st;
    unsigned char tail[8];
  } st;
  std::memset(&st, 0x11, sizeof st);
  const uint32_t ssmall = uint32_t(offsetof(ss_budget_stats, kind) + sizeof(ss_budget_kind_stats));
  st.st.struct_size = ssmall;
  CHECK_OK(ss_budget_ledger_stats(l, &st.st));
  CHECK_EQ(st.st.struct_size, ssmall);
  CHECK_EQ(st.st.kind[0].current, uint64_t(5));
  CHECK_EQ(reinterpret_cast<unsigned char*>(&st.st)[ssmall], 0x11);
  ss_budget_ledger_free(l);
}

TEST(margins_are_kept_free_under_the_caps) {
  constexpr uint64_t slot = 9953280ULL;
  Card k;
  ss_budget_out ref, out;
  ref.struct_size = out.struct_size = sizeof out;
  CHECK_EQ(k.in.vram_margin_bytes, uint64_t(0));  // defaults
  CHECK_EQ(k.in.ram_margin_bytes, uint64_t(0));
  // uncapped: the margins change no placement (only the minimum caps carry them)
  k.in.vram_total_bytes = 0;
  CHECK_OK(ss_budget_plan(&k.in, &ref));
  k.in.vram_margin_bytes = 1536ULL << 20;
  k.in.ram_margin_bytes = 1 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.resident, ref.resident);
  CHECK_EQ(out.pinned, ref.pinned);
  CHECK_EQ(out.disk, ref.disk);
  CHECK_EQ(out.room_bytes, ref.room_bytes);
  CHECK_EQ(out.device_planned_bytes, ref.device_planned_bytes);
  CHECK_EQ(out.min_vram_total_bytes, ref.min_vram_total_bytes + (1536ULL << 20));
  CHECK_EQ(out.min_ram_total_bytes, ref.min_ram_total_bytes + 1 * kGiB);
  // 24 GiB card, 1.5 GiB margin: the arena gets the rest of cap - fixed - margin; 64 GiB host, 1 GiB margin
  k.in.vram_total_bytes = 24 * kGiB;
  k.in.ram_total_bytes = 64 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  const uint64_t fixed = (600ULL << 20) + 14400000000ULL + 2 * kGiB + (512ULL << 20);
  const uint64_t slots = (24 * kGiB - fixed - (1536ULL << 20)) / slot;
  CHECK_EQ(out.arena_slots, slots);
  CHECK_EQ(out.resident, slots - 96);
  CHECK(out.device_planned_bytes + k.in.vram_margin_bytes <= 24 * kGiB);
  CHECK(out.device_planned_bytes + k.in.vram_margin_bytes + slot > 24 * kGiB);  // tight
  const uint64_t host_fixed = (600ULL << 20) + 4 * kGiB + 16ULL * 3870720ULL;
  const uint64_t blocks = (64 * kGiB - host_fixed - 1 * kGiB) / slot;
  CHECK_EQ(out.room_bytes, 64 * kGiB - host_fixed - 1 * kGiB);
  CHECK_EQ(out.pinned, blocks - 48 - 8);
  CHECK(out.host_planned_bytes + k.in.ram_margin_bytes <= 64 * kGiB);
  CHECK(out.host_planned_bytes + k.in.ram_margin_bytes + slot > 64 * kGiB);
  // the minimum caps carry the margins: exactly the minimum loads, one byte less is refused naming the margin
  k.in.min_resident = 36;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  const uint64_t vmin = fixed + (1536ULL << 20) + (96 + 36) * slot;
  CHECK_EQ(out.min_vram_total_bytes, vmin);
  const uint64_t rmin = host_fixed + 1 * kGiB + (48 + 8 + 1) * slot;
  CHECK_EQ(out.min_ram_total_bytes, rmin);
  k.in.vram_total_bytes = vmin;
  k.in.ram_total_bytes = rmin;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.resident, uint64_t(36));
  CHECK_EQ(out.pinned, uint64_t(1));
  k.in.vram_total_bytes = vmin - 1;
  CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
  std::printf("  %s\n", out.refuse);
  CHECK(std::string(out.refuse).find("margin 1.50") != std::string::npos);
  k.in.vram_total_bytes = vmin;
  k.in.ram_total_bytes = rmin - 1;
  CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
  std::printf("  %s\n", out.refuse);
  CHECK(std::string(out.refuse).find("margin 1.00") != std::string::npos);
  // a margin larger than the cap saturates: refused, never wrapped
  k.in.ram_total_bytes = 64 * kGiB;
  k.in.vram_margin_bytes = UINT64_MAX - 3;
  CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
  CHECK_EQ(out.min_vram_total_bytes, UINT64_MAX);
}

// The minimum a refusal names, in GiB with two decimals, is rounded UP: passing it as read loads.
TEST(refusal_minimum_is_rounded_up) {
  const auto named = [](const std::string& m) {  // "needs at least X GiB"
    const size_t at = m.find("needs at least ");
    CHECK(at != std::string::npos);
    return std::atof(m.c_str() + at + 15);
  };
  Card k;
  ss_budget_out out;
  out.struct_size = sizeof out;
  for (uint64_t extra : {uint64_t(1), uint64_t(5) << 20, uint64_t(10737418), uint64_t(1) << 29}) {
    k.in.vram_margin_bytes = extra;  // moves the minimum off the 0.01 GiB grid
    k.in.vram_total_bytes = 12 * kGiB;
    k.in.min_resident = 36;
    CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
    const double g = named(out.refuse);
    const uint64_t min = out.min_vram_total_bytes;
    CHECK(g * kGiBd >= double(min));
    CHECK(g * kGiBd < double(min) + 0.01 * kGiBd + 1.0);
    k.in.vram_total_bytes = uint64_t(std::ceil(g * kGiBd));
    CHECK_OK(ss_budget_plan(&k.in, &out));
  }
  k.in.vram_total_bytes = 0;
  k.in.ram_total_bytes = 2 * kGiB;
  k.in.ram_margin_bytes = 12345;
  CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
  const double g = named(out.refuse);
  CHECK(g * kGiBd >= double(out.min_ram_total_bytes));
  k.in.ram_total_bytes = uint64_t(std::ceil(g * kGiBd));
  CHECK_OK(ss_budget_plan(&k.in, &out));
}

TEST(capped_flags_room_and_ceiling) {
  constexpr uint64_t slot = 9953280ULL;
  Card k;
  ss_budget_out out;
  out.struct_size = sizeof out;
  // a 24 GiB card sizes the arena: capped, room = cap - fixed
  CHECK_OK(ss_budget_plan(&k.in, &out));
  const uint64_t fixed = (600ULL << 20) + 14400000000ULL + 2 * kGiB + (512ULL << 20);
  CHECK_EQ(out.device_capped, 1u);
  CHECK_EQ(out.device_room_bytes, 24 * kGiB - fixed);
  CHECK_EQ(out.host_capped, 0u);  // no RAM cap
  CHECK_EQ(out.host_cap_units, UINT64_MAX / slot);  // no host limit at all
  // an explicit arena below what the cap allows: the request sized it
  k.in.arena_units = 100;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.device_capped, 0u);
  // a 96 GiB card holds every expert: not capped; uncapped: room 0
  k.in.arena_units = kAuto;
  k.in.vram_total_bytes = 0;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.device_capped, 0u);
  CHECK_EQ(out.device_room_bytes, uint64_t(0));
  // a RAM cap below what the non-residents want: capped, the ceiling is the cap's room
  k.in.vram_total_bytes = 24 * kGiB;
  k.in.ram_total_bytes = 64 * kGiB;
  k.in.ram_margin_bytes = 1 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  const uint64_t host_fixed = (600ULL << 20) + 4 * kGiB + 16ULL * 3870720ULL;
  CHECK_EQ(out.host_capped, 1u);
  CHECK_EQ(out.host_cap_units, (64 * kGiB - host_fixed - 1 * kGiB) / slot);
  CHECK_EQ(out.pinned + out.spare_blocks + out.staging, out.host_cap_units);
  // a RAM cap above what is wanted: not capped
  k.in.ram_total_bytes = 400 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.host_capped, 0u);
  // the RAM cap above an explicit RAM request: the request is the ceiling, not capped
  k.in.ram_total_bytes = 64 * kGiB;
  k.in.ram_expert_bytes = 20 * kGiB;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.host_capped, 0u);
  CHECK_EQ(out.host_cap_units, (20 * kGiB) / slot);
}

TEST(plan_json) {
  Card k;
  k.in.ram_total_bytes = 32 * kGiB;
  k.in.vram_margin_bytes = 1536ULL << 20;
  k.in.ram_margin_bytes = 1 * kGiB;
  ss_budget_out out;
  out.struct_size = sizeof out;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  const size_t need = ss_budget_plan_json(&k.in, &out, nullptr, 0);
  CHECK(need > 200);
  std::string s(need + 1, '\0');
  CHECK_EQ(ss_budget_plan_json(&k.in, &out, &s[0], s.size()), need);
  s.resize(need);
  std::printf("  %.200s...\n", s.c_str());
  for (const char* m : {"\"vram_total_bytes\":25769803776", "\"ram_total_bytes\":34359738368",
                        "\"vram_margin_bytes\":1610612736", "\"ram_margin_bytes\":1073741824",
                        "\"device_context_bytes\":629145600", "\"device_fixed\":{\"trunk\":14400000000,\"kv\":",
                        "\"pinned_fixed\":{\"engine pinned\":", "\"pageable_fixed\":{\"host weights\":",
                        "\"arena_units\":null", "\"ram_expert_bytes\":null", "\"device_capped\":true",
                        "\"host_capped\":true", "\"three_tier\":true", "\"min_vram_total_gib\":",
                        "\"min_ram_total_flag_gib\":", "\"refused\":false", "\"refuse\":\"\""})
    CHECK(s.find(m) != std::string::npos);
  CHECK(s.find("\"arena_slots\":" + std::to_string(out.arena_slots)) != std::string::npos);
  CHECK(s.find("\"disk\":" + std::to_string(out.disk)) != std::string::npos);
  CHECK(s.front() == '"' && s.back() == '"');  // members only: the caller adds the braces
  // it parses as an object once braced (a minimal check: balanced braces and quotes outside strings)
  int depth = 0;
  bool in_str = false;
  for (size_t i = 0; i < s.size(); ++i) {
    if (in_str) {
      if (s[i] == '\\') ++i;
      else if (s[i] == '"') in_str = false;
    } else if (s[i] == '"') {
      in_str = true;
    } else if (s[i] == '{') {
      ++depth;
    } else if (s[i] == '}') {
      --depth;
    }
    CHECK(depth >= 0);
  }
  CHECK(depth == 0 && !in_str);
  // truncation: NUL-terminated, the whole length returned
  char small[16];
  CHECK_EQ(ss_budget_plan_json(&k.in, &out, small, sizeof small), need);
  CHECK_EQ(std::strlen(small), sizeof small - 1);
  CHECK_EQ(ss_budget_plan_json(nullptr, &out, small, sizeof small), size_t(0));
  CHECK_EQ(small[0], '\0');
  // a refusal: the message, the minimum rounded up and the flag value
  k.in.vram_total_bytes = 12 * kGiB;
  CHECK_EQ(int(ss_budget_plan(&k.in, &out)), int(SS_E_CAPACITY));
  std::string r(ss_budget_plan_json(&k.in, &out, nullptr, 0) + 1, '\0');
  ss_budget_plan_json(&k.in, &out, &r[0], r.size());
  CHECK(r.find("\"refused\":true") != std::string::npos);
  CHECK(r.find("\"refuse\":\"VRAM cap 12.00 GiB") != std::string::npos);
  CHECK(r.find("\"min_vram_total_flag_gib\":19") != std::string::npos);
  // names are escaped
  ss_budget_component odd[1] = {{"a\"b\\c\n", 1}};
  k.in.device_fixed = odd;
  k.in.n_device_fixed = 1;
  k.in.vram_total_bytes = 0;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  std::string t(ss_budget_plan_json(&k.in, &out, nullptr, 0) + 1, '\0');
  ss_budget_plan_json(&k.in, &out, &t[0], t.size());
  CHECK(t.find("\"device_fixed\":{\"a\\\"b\\\\c\\u000a\":1}") != std::string::npos);
  // an older caller's output struct (before the appended fields): read up to its size
  ss_budget_out old = out;
  old.struct_size = uint32_t(offsetof(ss_budget_out, device_room_bytes));
  old.device_capped = 1;  // past its size: not read
  std::string u(ss_budget_plan_json(&k.in, &old, nullptr, 0) + 1, '\0');
  ss_budget_plan_json(&k.in, &old, &u[0], u.size());
  CHECK(u.find("\"device_capped\":false") != std::string::npos);
}

// Without the file tier an explicit RAM request is not clamped to the pinned room; under a RAM cap the inclusive clean
// copies must still stop at the capped room (found by the random plans below).
TEST(inclusive_copies_stay_under_the_ram_cap_without_a_file_tier) {
  Card k;
  k.in.file_tier = 0;
  k.in.ram_expert_bytes = 500 * kGiB;
  k.in.inclusive_bytes = kAuto;  // up to 16 GiB of clean copies
  ss_budget_out out;
  out.struct_size = sizeof out;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  const uint64_t nonres = k.in.keys - out.resident;
  const uint64_t host_fixed = (600ULL << 20) + 4 * kGiB + 16ULL * 3870720ULL;
  k.in.ram_total_bytes = host_fixed + (nonres + 48 + 100) * 9953280ULL;  // room for the placement + 100 copies
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.pinned, nonres);
  CHECK_EQ(out.inclusive, uint64_t(100));
  CHECK(out.host_planned_bytes <= k.in.ram_total_bytes);
  // uncapped, the copies take what the request leaves (tier_plan's rule, unchanged)
  k.in.ram_total_bytes = 0;
  CHECK_OK(ss_budget_plan(&k.in, &out));
  CHECK_EQ(out.inclusive, (16 * kGiB) / 9953280ULL);
}

// ---- ts_capplan's cases (ts-budget e9e13e2, tests/test_capplan.cpp), on ss_budget's API ----
namespace {
constexpr uint64_t kMiB = uint64_t(1) << 20;
constexpr uint64_t kUnit = 9953280;  // one DS V4.1 Flash routed expert (gate + up + down, Q2)

// The DS RTX shape as the replica plans it: 40 x 384 experts, 96 arena spares, 48 host spares, 8 staging buffers,
// the arena's extras (bank ring 73 slots + reserves) as a named device component, the process commit as pageable.
struct DsShape {
  ss_budget_component dev[5] = {{"trunk", 10ULL * kGiB + 300 * kMiB},
                                {"drafter", 4219 * kMiB},
                                {"workspaces", 200 * kMiB},
                                {"arena reserves", 73ULL * kUnit + 256 * kMiB},
                                {nullptr, 0}};
  ss_budget_component page[1] = {{"process_commit", 3ULL * kGiB}};
  ss_budget_in in;
  uint64_t request = 0;  // the replica's arena without a cap (residents)
  DsShape() {
    ss_budget_in_defaults(&in);
    in.device_context_bytes = 600 * kMiB;
    in.device_fixed = dev;
    in.n_device_fixed = 4;
    in.pageable_fixed = page;
    in.n_pageable_fixed = 1;
    in.keys = 40ULL * 384ULL;
    in.slot_bytes = in.unit_bytes = kUnit;
    in.arena_spares = 96;
    request = uint64_t(74.8 * kGiBd / double(kUnit));
    in.arena_units = request;
    in.min_resident = 36;
    in.host_spares = 48;
    in.staging = 8;
    in.load_staging_bytes = 16ULL * 3870720ULL;
  }
  uint64_t device_fixed() const { return 600 * kMiB + 10ULL * kGiB + 300 * kMiB + 4219 * kMiB + 200 * kMiB +
                                         73ULL * kUnit + 256 * kMiB; }
};
uint64_t placed(const ss_budget_out& o) { return o.resident + o.pinned + o.disk; }
ss_budget_out plan_of(const ss_budget_in& in, ss_status* st = nullptr) {
  ss_budget_out o;
  o.struct_size = sizeof o;
  const ss_status r = ss_budget_plan(&in, &o);
  if (st) *st = r;
  return o;
}
}  // namespace

TEST(capplan_no_caps_is_the_request) {
  DsShape k;
  ss_status st;
  ss_budget_out p = plan_of(k.in, &st);
  CHECK_OK(st);
  CHECK_EQ(p.arena_slots, k.request + 96);
  CHECK_EQ(p.resident, k.request);
  CHECK_EQ(p.pinned, k.in.keys - k.request);
  CHECK_EQ(p.disk, uint64_t(0));
  CHECK_EQ(p.three_tier, 0u);
  CHECK_EQ(p.staging, uint64_t(0));
  CHECK_EQ(p.device_capped, 0u);
  CHECK_EQ(p.host_capped, 0u);
  CHECK_EQ(p.refused, 0u);
  // a RAM request below the non-residents (the caller's RAM budget) gives the three-tier split without any cap
  k.in.ram_expert_bytes = 3000 * kUnit;
  p = plan_of(k.in, &st);
  CHECK_OK(st);
  CHECK_EQ(p.pinned, uint64_t(3000 - 48 - 8));
  CHECK_EQ(p.disk, k.in.keys - p.resident - p.pinned);
  CHECK_EQ(p.three_tier, 1u);
  CHECK_EQ(p.host_capped, 0u);
  // no file tier and no cap: every non-resident pinned (the caller's own checks decide), as the engine did
  k.in.file_tier = 0;
  p = plan_of(k.in, &st);
  CHECK_OK(st);
  CHECK_EQ(p.pinned, k.in.keys - p.resident);
  CHECK_EQ(p.disk, uint64_t(0));
}

TEST(capplan_covering_caps_change_nothing) {
  DsShape k;
  k.in.vram_margin_bytes = 1536 * kMiB;
  k.in.ram_margin_bytes = 1 * kGiB;
  const ss_budget_out a = plan_of(k.in);
  k.in.vram_total_bytes = 96 * kGiB;
  k.in.ram_total_bytes = 200 * kGiB;
  ss_status st;
  const ss_budget_out b = plan_of(k.in, &st);
  CHECK_OK(st);
  CHECK_EQ(a.arena_slots, b.arena_slots);
  CHECK_EQ(a.resident, b.resident);
  CHECK_EQ(a.pinned, b.pinned);
  CHECK_EQ(a.disk, b.disk);
  CHECK_EQ(a.inclusive, b.inclusive);
  CHECK_EQ(b.device_capped, 0u);
  CHECK_EQ(b.host_capped, 0u);
  CHECK(b.device_planned_bytes + k.in.vram_margin_bytes <= k.in.vram_total_bytes);
  CHECK(b.host_planned_bytes + k.in.ram_margin_bytes <= k.in.ram_total_bytes);
}

TEST(capplan_whole_device_24_32) {
  DsShape k;
  k.in.vram_total_bytes = 24 * kGiB;
  k.in.ram_total_bytes = 32 * kGiB;
  k.in.vram_margin_bytes = 1536 * kMiB;
  k.in.ram_margin_bytes = 1 * kGiB;
  ss_status st;
  const ss_budget_out p = plan_of(k.in, &st);
  CHECK_OK(st);
  CHECK_EQ(p.device_capped, 1u);
  CHECK_EQ(p.host_capped, 1u);
  CHECK_EQ(p.device_fixed_bytes, k.device_fixed());
  const uint64_t room = 24 * kGiB - k.device_fixed() - 1536 * kMiB;
  CHECK_EQ(p.device_room_bytes, room);
  CHECK_EQ(p.arena_slots, room / kUnit);
  CHECK_EQ(p.resident, room / kUnit - 96);
  CHECK(p.device_planned_bytes + k.in.vram_margin_bytes <= k.in.vram_total_bytes);
  CHECK(p.device_planned_bytes + k.in.vram_margin_bytes + kUnit > k.in.vram_total_bytes);  // tight
  const uint64_t hroom = 32 * kGiB - 3 * kGiB - 1 * kGiB - k.in.load_staging_bytes;
  CHECK_EQ(p.host_cap_units, hroom / kUnit);
  CHECK_EQ(p.pinned, hroom / kUnit - 48 - 8);
  CHECK_EQ(p.three_tier, 1u);
  CHECK_EQ(p.staging, uint64_t(8));
  CHECK_EQ(placed(p), k.in.keys);
  CHECK(p.host_planned_bytes + k.in.ram_margin_bytes <= k.in.ram_total_bytes);
  CHECK(p.host_planned_bytes + k.in.ram_margin_bytes + kUnit > k.in.ram_total_bytes);
  CHECK(p.min_vram_total_bytes < k.in.vram_total_bytes);
  CHECK(p.min_ram_total_bytes < k.in.ram_total_bytes);
  std::printf("  24+32: arena %llu slots (%llu experts), pinned %llu, file %llu, minimum caps %.2f / %.2f GiB\n",
              (unsigned long long)p.arena_slots, (unsigned long long)p.resident, (unsigned long long)p.pinned,
              (unsigned long long)p.disk, double(p.min_vram_total_bytes) / kGiBd,
              double(p.min_ram_total_bytes) / kGiBd);
}

TEST(capplan_device_refusal_names_the_minimum) {
  DsShape k;
  k.in.vram_margin_bytes = 1536 * kMiB;
  k.in.vram_total_bytes = 12 * kGiB;  // below trunk + drafter
  ss_status st;
  ss_budget_out p = plan_of(k.in, &st);
  CHECK_EQ(int(st), int(SS_E_CAPACITY));
  const uint64_t min_cap = k.device_fixed() + 1536 * kMiB + (96ULL + 36ULL) * kUnit;
  CHECK_EQ(p.min_vram_total_bytes, min_cap);
  const std::string m = p.refuse;
  std::printf("  %s\n", m.c_str());
  CHECK(m.find("VRAM cap 12.00 GiB cannot hold the fixed parts") != std::string::npos);
  CHECK(m.find("trunk 10.29") != std::string::npos);
  CHECK(m.find("margin 1.50") != std::string::npos);
  CHECK(m.find("--vram-total-gib " + std::to_string(min_cap / kGiB + 1)) != std::string::npos);
  // exactly the minimum loads (36 residents), one byte less refuses
  k.in.vram_total_bytes = min_cap;
  p = plan_of(k.in, &st);
  CHECK_OK(st);
  CHECK_EQ(p.resident, uint64_t(36));
  CHECK_EQ(p.arena_slots, uint64_t(132));
  k.in.vram_total_bytes = min_cap - 1;
  p = plan_of(k.in, &st);
  CHECK_EQ(int(st), int(SS_E_CAPACITY));
}

TEST(capplan_host_refusal_names_the_minimum) {
  DsShape k;
  k.in.ram_margin_bytes = 1 * kGiB;
  k.in.ram_total_bytes = 4 * kGiB;  // the 3 GiB fixed + 1 GiB margin leave nothing
  ss_status st;
  ss_budget_out p = plan_of(k.in, &st);
  CHECK_EQ(int(st), int(SS_E_CAPACITY));
  const uint64_t min_cap = 4 * kGiB + k.in.load_staging_bytes + (48ULL + 8ULL + 1ULL) * kUnit;
  CHECK_EQ(p.min_ram_total_bytes, min_cap);
  std::printf("  %s\n", p.refuse);
  CHECK(std::string(p.refuse).find("RAM cap 4.00 GiB cannot hold the fixed parts") != std::string::npos);
  k.in.ram_total_bytes = min_cap;
  p = plan_of(k.in, &st);
  CHECK_OK(st);
  CHECK_EQ(p.pinned, uint64_t(1));
  CHECK_EQ(p.disk, k.in.keys - p.resident - 1);
  k.in.ram_total_bytes = min_cap - 1;
  p = plan_of(k.in, &st);
  CHECK_EQ(int(st), int(SS_E_CAPACITY));
  // without the file tier a RAM cap must hold every non-resident expert
  k.in.file_tier = 0;
  k.in.ram_total_bytes = 32 * kGiB;
  p = plan_of(k.in, &st);
  CHECK_EQ(int(st), int(SS_E_CAPACITY));
  CHECK(std::string(p.refuse).find("no file tier") != std::string::npos);
  CHECK_EQ(p.min_ram_total_bytes, 4 * kGiB + k.in.load_staging_bytes + (k.in.keys - p.resident + 48ULL) * kUnit);
  k.in.ram_total_bytes = p.min_ram_total_bytes;
  p = plan_of(k.in, &st);
  CHECK_OK(st);
  CHECK_EQ(p.disk, uint64_t(0));
}

TEST(capplan_device_refusal_wins_over_host) {
  DsShape k;
  k.in.vram_total_bytes = 1 * kGiB;
  k.in.ram_total_bytes = 1 * kGiB;
  ss_status st;
  const ss_budget_out p = plan_of(k.in, &st);
  CHECK_EQ(int(st), int(SS_E_CAPACITY));
  CHECK(std::string(p.refuse).rfind("VRAM cap", 0) == 0);
  CHECK(p.min_ram_total_bytes > 0);  // still reported
}

// ts_capplan's 20,000 random plans, on ss_budget: an accepted plan places every expert once, stays under both caps
// with their margins, keeps the request as a bound and the pinned tier under its ceiling; the minimum VRAM cap it
// reports loads (the device side) and one byte less is refused; a refusal names a cap below its minimum, and the
// minimum RAM cap clears the RAM refusal.
TEST(capplan_properties_random) {
  std::mt19937_64 rng(12345);
  int accepted = 0, refused_v = 0, refused_r = 0;
  for (int it = 0; it < 20000; ++it) {
    ss_budget_component dev[4], pin[2], page[2];
    ss_budget_in in;
    ss_budget_in_defaults(&in);
    in.keys = 1 + rng() % 20000;
    in.slot_bytes = 1 + rng() % (16 * kMiB);
    in.unit_bytes = in.slot_bytes + rng() % 4096;
    in.arena_spares = rng() % 128;
    in.arena_units = rng() % 4 ? rng() % (in.keys + 64) : kAuto;
    in.min_resident = rng() % 64;
    in.host_spares = uint32_t(rng() % 64);
    in.staging = uint32_t(rng() % 16);
    in.load_staging_bytes = rng() % kGiB;
    in.file_tier = rng() % 5 != 0;
    if (rng() % 3 == 0) in.ram_expert_bytes = (rng() % (in.keys + 200)) * in.unit_bytes;
    if (rng() % 4 == 0) in.host_avail_bytes = rng() % (200 * kGiB);
    if (rng() % 4 == 0) in.max_pinned_bytes = rng() % (200 * kGiB);
    const uint64_t inc_mode = rng() % 3;
    in.inclusive_bytes = inc_mode == 0 ? 0 : inc_mode == 1 ? kAuto : rng() % (32 * kGiB);
    in.device_context_bytes = rng() % kGiB;
    in.n_device_fixed = uint32_t(rng() % 5);
    for (uint32_t i = 0; i < in.n_device_fixed; ++i) dev[i] = {"d", rng() % (8 * kGiB)};
    in.n_pinned_fixed = uint32_t(rng() % 3);
    for (uint32_t i = 0; i < in.n_pinned_fixed; ++i) pin[i] = {"p", rng() % (4 * kGiB)};
    in.n_pageable_fixed = uint32_t(rng() % 3);
    for (uint32_t i = 0; i < in.n_pageable_fixed; ++i) page[i] = {"h", rng() % (4 * kGiB)};
    in.device_fixed = dev;
    in.pinned_fixed = pin;
    in.pageable_fixed = page;
    in.vram_margin_bytes = rng() % (2 * kGiB);
    in.ram_margin_bytes = rng() % (2 * kGiB);
    if (rng() % 4) in.vram_total_bytes = 1 + rng() % (200 * kGiB);
    if (rng() % 4) in.ram_total_bytes = 1 + rng() % (200 * kGiB);
    ss_status st;
    const ss_budget_out p = plan_of(in, &st);
    const uint64_t request = in.arena_units == kAuto ? in.keys : std::min(in.arena_units, in.keys);
    const std::string why = p.refuse;
    if (st == SS_OK) {
      ++accepted;
      CHECK_EQ(p.refused, 0u);
      CHECK(p.resident <= request);
      CHECK_EQ(p.arena_slots, p.resident + in.arena_spares);
      CHECK_EQ(placed(p), in.keys);
      if (!in.file_tier) CHECK_EQ(p.disk, uint64_t(0));
      if (in.file_tier) CHECK(p.pinned + p.spare_blocks + p.staging <= p.host_cap_units);
      if (in.vram_total_bytes) {
        CHECK(p.device_planned_bytes + in.vram_margin_bytes <= in.vram_total_bytes);
        CHECK(p.min_vram_total_bytes <= in.vram_total_bytes);
        // the minimum VRAM cap loads on the device side; one byte less is refused there
        ss_budget_in m = in;
        m.vram_total_bytes = p.min_vram_total_bytes;
        CHECK(std::string(plan_of(m).refuse).rfind("VRAM cap", 0) != 0);
        if (p.min_vram_total_bytes > 1) {
          m.vram_total_bytes = p.min_vram_total_bytes - 1;
          CHECK(std::string(plan_of(m).refuse).rfind("VRAM cap", 0) == 0);
        }
      }
      if (in.ram_total_bytes) {
        CHECK(p.host_planned_bytes + in.ram_margin_bytes <= in.ram_total_bytes);
        CHECK(p.min_ram_total_bytes <= in.ram_total_bytes);
      }
    } else {
      CHECK_EQ(int(st), int(SS_E_CAPACITY));
      CHECK_EQ(p.refused, 1u);
      CHECK(!why.empty());
      if (why.rfind("VRAM cap", 0) == 0) {
        ++refused_v;
        CHECK(in.vram_total_bytes && in.vram_total_bytes < p.min_vram_total_bytes);
      }
      if (why.find("cannot hold the fixed parts") != std::string::npos && why.rfind("RAM cap", 0) == 0) {
        ++refused_r;
        CHECK(in.ram_total_bytes && in.ram_total_bytes < p.min_ram_total_bytes);
        ss_budget_in m = in;
        m.ram_total_bytes = p.min_ram_total_bytes;  // the minimum it names clears this refusal
        const std::string again = plan_of(m).refuse;
        CHECK(again.rfind("RAM cap", 0) != 0 || again.find("cannot hold the fixed parts") == std::string::npos);
      }
    }
  }
  std::printf("  %d accepted, %d VRAM refusals, %d RAM refusals of 20000\n", accepted, refused_v, refused_r);
  CHECK(accepted > 2000 && refused_v > 500 && refused_r > 100);
}

TEST(ledger_add_sub_peak_and_components_at_the_peak) {
  ss_budget_ledger* l = nullptr;
  CHECK_OK(ss_budget_ledger_create(0, 0, &l));
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "trunk", 1000));
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "kv", 300));
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "arena", 500));
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "arena", 200));  // the same name accumulates
  CHECK_OK(ss_budget_ledger_sub(l, SS_BUDGET_DEVICE, "arena", 700));
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "scratch", 100));
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_PINNED, "blocks", 4000));
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_PAGEABLE, "host", 50));
  CHECK_EQ(int(ss_budget_ledger_sub(l, SS_BUDGET_DEVICE, "kv", 301)), int(SS_E_INVALID));  // more than it holds
  CHECK(std::string(ss_last_error()).find("kv") != std::string::npos);
  CHECK_EQ(int(ss_budget_ledger_sub(l, SS_BUDGET_DEVICE, "nothing", 1)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_budget_ledger_add(l, ss_budget_kind(7), "x", 1)), int(SS_E_INVALID));
  CHECK_OK(ss_budget_ledger_observe_device(l, 5000));
  CHECK_OK(ss_budget_ledger_observe_device(l, 4000));
  ss_budget_stats s;
  s.struct_size = sizeof s;
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  const ss_budget_kind_stats& d = s.kind[SS_BUDGET_DEVICE];
  CHECK_EQ(d.current, uint64_t(1400));
  CHECK_EQ(d.peak, uint64_t(2000));
  CHECK_EQ(d.n_top, 3u);  // at the peak: trunk 1000, arena 700, kv 300 (largest first)
  CHECK_EQ(std::string(d.top[0].name), std::string("trunk"));
  CHECK_EQ(d.top[0].bytes, uint64_t(1000));
  CHECK_EQ(std::string(d.top[1].name), std::string("arena"));
  CHECK_EQ(d.top[1].bytes, uint64_t(700));
  CHECK_EQ(std::string(d.top[2].name), std::string("kv"));
  CHECK_EQ(s.kind[SS_BUDGET_PINNED].current, uint64_t(4000));
  CHECK_EQ(s.host_current, uint64_t(4050));
  CHECK_EQ(s.host_peak, uint64_t(4050));
  CHECK_EQ(s.measured_device_last, uint64_t(4000));
  CHECK_EQ(s.measured_device_peak, uint64_t(5000));
  CHECK_EQ(s.observations, uint64_t(2));
  CHECK_EQ(s.refusals, 0u);
  // at most SS_BUDGET_TOP components at a peak, and names cut to fit
  const std::string long_name(100, 'n');
  for (int i = 0; i < 20; ++i)
    CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_PAGEABLE, (i == 19 ? long_name : "c" + std::to_string(i)).c_str(),
                                  uint64_t(100 + i)));
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  CHECK_EQ(s.kind[SS_BUDGET_PAGEABLE].n_top, uint32_t(SS_BUDGET_TOP));
  CHECK_EQ(std::strlen(s.kind[SS_BUDGET_PAGEABLE].top[0].name), size_t(SS_BUDGET_NAME - 1));
  CHECK_EQ(s.kind[SS_BUDGET_PAGEABLE].top[0].bytes, uint64_t(119));
  ss_budget_ledger_free(l);
}

// Measured samples against the caps (ported from ts_capplan's peak tracker): peaks with their stage, samples above a
// cap, a side without a measurement or without a cap.
TEST(ledger_measured_peaks_against_the_caps) {
  ss_budget_ledger* l = nullptr;
  CHECK_OK(ss_budget_ledger_create(24 * kGiB, 0, &l));
  CHECK_EQ(ss_budget_ledger_within_caps(l), 1);  // nothing measured
  CHECK_OK(ss_budget_ledger_observe(l, "load", 20 * kGiB, 30 * kGiB));
  CHECK_OK(ss_budget_ledger_observe(l, "warmup", 23 * kGiB, SS_BUDGET_NO_SAMPLE));
  CHECK_OK(ss_budget_ledger_observe(l, "request:a", 22 * kGiB, 31 * kGiB));
  ss_budget_stats s;
  s.struct_size = sizeof s;
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  CHECK_EQ(s.observations, uint64_t(3));
  CHECK_EQ(s.host_observations, uint64_t(2));
  CHECK_EQ(s.measured_device_peak, 23 * kGiB);
  CHECK_EQ(s.measured_device_last, 22 * kGiB);
  CHECK_EQ(std::string(s.measured_device_peak_at), std::string("warmup"));
  CHECK_EQ(s.measured_host_peak, 31 * kGiB);
  CHECK_EQ(s.measured_host_last, 31 * kGiB);
  CHECK_EQ(std::string(s.measured_host_peak_at), std::string("request:a"));
  CHECK_EQ(s.within_caps, 1u);  // no RAM cap
  CHECK_EQ(ss_budget_ledger_within_caps(l), 1);
  CHECK_EQ(s.kind[SS_BUDGET_DEVICE].current, uint64_t(0));  // samples charge nothing
  // a sample above the VRAM cap: counted, and the ledger is no longer within its caps (observe itself succeeds)
  CHECK_OK(ss_budget_ledger_observe(l, "request:b", 24 * kGiB + 1, 0));
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  CHECK_EQ(s.device_over, 1u);
  CHECK_EQ(s.within_caps, 0u);
  CHECK_EQ(ss_budget_ledger_within_caps(l), 0);
  CHECK_EQ(std::string(s.measured_device_peak_at), std::string("request:b"));
  // observe_device counts as a device sample (no stage)
  CHECK_OK(ss_budget_ledger_observe_device(l, 25 * kGiB));
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  CHECK_EQ(s.device_over, 2u);
  CHECK_EQ(std::string(s.measured_device_peak_at), std::string(""));
  ss_budget_ledger_free(l);
  // the RAM cap: a host sample above it; exactly the cap is within
  CHECK_OK(ss_budget_ledger_create(0, 32 * kGiB, &l));
  CHECK_OK(ss_budget_ledger_observe(l, "load", SS_BUDGET_NO_SAMPLE, 32 * kGiB));
  CHECK_EQ(ss_budget_ledger_within_caps(l), 1);
  CHECK_OK(ss_budget_ledger_observe(l, nullptr, SS_BUDGET_NO_SAMPLE, 32 * kGiB + 1));
  CHECK_EQ(ss_budget_ledger_within_caps(l), 0);
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  CHECK_EQ(s.host_over, 1u);
  CHECK_EQ(s.observations, uint64_t(0));
  ss_budget_ledger_free(l);
  // no caps: always within; NULL: 0 / SS_E_INVALID
  CHECK_OK(ss_budget_ledger_create(0, 0, &l));
  CHECK_OK(ss_budget_ledger_observe(l, "x", UINT64_MAX - 1, UINT64_MAX - 1));
  CHECK_EQ(ss_budget_ledger_within_caps(l), 1);
  ss_budget_ledger_free(l);
  CHECK_EQ(ss_budget_ledger_within_caps(nullptr), 0);
  CHECK_EQ(int(ss_budget_ledger_observe(nullptr, "x", 1, 1)), int(SS_E_INVALID));
  // an older caller's stats struct (before the appended fields): only its bytes are written
  CHECK_OK(ss_budget_ledger_create(1, 0, &l));
  CHECK_OK(ss_budget_ledger_observe(l, "x", 2, SS_BUDGET_NO_SAMPLE));
  struct {
    ss_budget_stats st;
    unsigned char tail[8];
  } o;
  std::memset(&o, 0x5A, sizeof o);
  const uint32_t old_size = uint32_t(offsetof(ss_budget_stats, measured_host_last));
  o.st.struct_size = old_size;
  CHECK_OK(ss_budget_ledger_stats(l, &o.st));
  CHECK_EQ(o.st.struct_size, old_size);
  CHECK_EQ(o.st.measured_device_peak, uint64_t(2));
  CHECK_EQ(reinterpret_cast<unsigned char*>(&o.st)[old_size], 0x5A);
  ss_budget_ledger_free(l);
}

TEST(ledger_cap_overrun_fails_with_the_name) {
  ss_budget_ledger* l = nullptr;
  CHECK_OK(ss_budget_ledger_create(1000, 500, &l));
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "trunk", 600));
  CHECK_EQ(int(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "arena", 401)), int(SS_E_CAPACITY));
  const std::string m = ss_last_error();
  CHECK(m.find("arena") != std::string::npos && m.find("VRAM") != std::string::npos);
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "arena", 400));  // exactly the cap
  CHECK_EQ(int(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "one more", 1)), int(SS_E_CAPACITY));
  CHECK_EQ(int(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "wrap", UINT64_MAX)), int(SS_E_CAPACITY));
  // the RAM cap holds pinned and pageable together
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_PINNED, "blocks", 300));
  CHECK_EQ(int(ss_budget_ledger_add(l, SS_BUDGET_PAGEABLE, "host weights", 201)), int(SS_E_CAPACITY));
  CHECK(std::string(ss_last_error()).find("host weights") != std::string::npos);
  CHECK_OK(ss_budget_ledger_add(l, SS_BUDGET_PAGEABLE, "host weights", 200));
  ss_budget_stats s;
  s.struct_size = sizeof s;
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  CHECK_EQ(s.refusals, 4u);
  CHECK_EQ(s.kind[SS_BUDGET_DEVICE].current, uint64_t(1000));  // refused adds charged nothing
  CHECK_EQ(s.kind[SS_BUDGET_DEVICE].cap, uint64_t(1000));
  CHECK_EQ(s.kind[SS_BUDGET_PAGEABLE].cap, uint64_t(500));
  CHECK_EQ(s.host_current, uint64_t(500));
  ss_budget_ledger_free(l);
}

TEST(ledger_concurrent_adds_keep_exact_totals) {
  constexpr int kThreads = 8, kOps = 20000;
  constexpr uint64_t kBytes = 3;
  ss_budget_ledger* l = nullptr;
  // the cap is exactly what every add together needs: none may be refused, one more must be
  CHECK_OK(ss_budget_ledger_create(uint64_t(kThreads) * kOps * kBytes, 0, &l));
  std::vector<std::thread> ts;
  std::vector<int> failed(kThreads, 0);
  for (int t = 0; t < kThreads; ++t)
    ts.emplace_back([&, t] {
      const std::string name = "t" + std::to_string(t % 3);  // shared names across threads
      for (int i = 0; i < kOps; ++i) {
        if (ss_budget_ledger_add(l, SS_BUDGET_DEVICE, name.c_str(), kBytes) != SS_OK) ++failed[size_t(t)];
        if (ss_budget_ledger_add(l, SS_BUDGET_PINNED, "pin", 1) != SS_OK) ++failed[size_t(t)];
        if (i % 2 && ss_budget_ledger_sub(l, SS_BUDGET_PINNED, "pin", 1) != SS_OK) ++failed[size_t(t)];
      }
    });
  for (auto& t : ts) t.join();
  for (int f : failed) CHECK_EQ(f, 0);
  ss_budget_stats s;
  s.struct_size = sizeof s;
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  CHECK_EQ(s.kind[SS_BUDGET_DEVICE].current, uint64_t(kThreads) * kOps * kBytes);
  CHECK_EQ(s.kind[SS_BUDGET_DEVICE].peak, uint64_t(kThreads) * kOps * kBytes);
  CHECK_EQ(s.kind[SS_BUDGET_PINNED].current, uint64_t(kThreads) * kOps / 2);
  uint64_t top_sum = 0;
  for (uint32_t i = 0; i < s.kind[SS_BUDGET_DEVICE].n_top; ++i) top_sum += s.kind[SS_BUDGET_DEVICE].top[i].bytes;
  CHECK_EQ(top_sum, uint64_t(kThreads) * kOps * kBytes);
  CHECK_EQ(int(ss_budget_ledger_add(l, SS_BUDGET_DEVICE, "t0", 1)), int(SS_E_CAPACITY));
  ss_budget_ledger_free(l);
}

namespace {

struct Run {
  ss_desc* d = nullptr;
  ss_plan* p = nullptr;
  ss_ctx* c = nullptr;
  ss_budget b;
  ss_options o;
  ~Run() {
    ss_close(c);
    ss_plan_free(p);
    ss_desc_free(d);
  }
};

// The runtime tests' placement: residents 0-3 of layers 1, 2 and 4, some pinned, the rest from the files.
bool prepare(Run& r, const sstest::Fixture& fx) {
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return false;
  ss_budget_defaults(&r.b);
  r.b.margin_gib = 1.0 / 1024;
  r.b.margin_fraction = 0;
  r.b.device_bytes = uint64_t(256) << 20;
  r.b.ring_slots = 6;
  r.b.host_pinned_bytes = 2 << 20;
  r.b.dynamic_slots_per_layer = 2;
  const std::string js = R"({"tiers":["","RRRRPPPPPPDDDDDD","RRRRPPDD","D","RRRRDDDDDDDDDDDD"]})";
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
  ss_options_defaults(&r.o);
  r.o.staging_bytes = 64 * 1024;
  return r.p != nullptr;
}

}  // namespace

TEST(host_context_charges_its_allocations_to_the_ledger) {
  sstest::Fixture fx("budget");
  ss_budget_ledger* l = nullptr;
  CHECK_OK(ss_budget_ledger_create(0, 0, &l));
  {
    Run r;
    if (!prepare(r, fx)) return;
    CHECK(r.o.budget == nullptr);  // the default: nothing charged
    r.o.budget = l;
    CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
    ss_budget_stats s;
    s.struct_size = sizeof s;
    CHECK_OK(ss_budget_ledger_stats(l, &s));
    CHECK(s.kind[SS_BUDGET_DEVICE].current > 0);  // the resident pool, the rings
    CHECK(s.kind[SS_BUDGET_PINNED].current > 0);  // the pinned units, the staging ring, the tables
    CHECK_EQ(std::string(s.kind[SS_BUDGET_DEVICE].top[0].name), std::string("tierstream device"));
    // a forward still works through the charged backend
    if (r.c) {
      CHECK_OK(ss_begin(r.c, nullptr, 0));
      const int32_t ids[] = {0, 5, 9, 15};
      CHECK_OK(ss_layer_routes(r.c, 1, ids, 4));
      ss_layer_view v;
      CHECK_OK(ss_acquire(r.c, 1, nullptr, &v));
      for (int u : ids) {
        const void* a = reinterpret_cast<const void*>(uintptr_t(v.table_host[u]));
        CHECK(a && std::memcmp(a, fx.expected[1][size_t(u)].data(), fx.expected[1][size_t(u)].size()) == 0);
      }
      CHECK_OK(ss_release(r.c, 1, nullptr));
      CHECK_OK(ss_end(r.c));
    }
  }
  ss_budget_stats s;
  s.struct_size = sizeof s;
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  CHECK_EQ(s.kind[SS_BUDGET_DEVICE].current, uint64_t(0));  // closing gave every charge back
  CHECK_EQ(s.kind[SS_BUDGET_PINNED].current, uint64_t(0));
  CHECK(s.kind[SS_BUDGET_DEVICE].peak > 0);
  const uint64_t device_peak = s.kind[SS_BUDGET_DEVICE].peak;
  ss_budget_ledger_free(l);

  // a VRAM cap one byte below what the context needs: ss_open refuses, naming the allocation; nothing stays charged
  CHECK_OK(ss_budget_ledger_create(device_peak - 1, 0, &l));
  {
    Run r;
    if (!prepare(r, fx)) return;
    r.o.budget = l;
    CHECK_EQ(int(ss_open(r.d, &r.b, r.p, &r.o, &r.c)), int(SS_E_CAPACITY));
    const std::string m = ss_last_error();
    std::printf("  %s\n", m.c_str());
    CHECK(m.find("tierstream device") != std::string::npos && m.find("VRAM cap") != std::string::npos);
  }
  CHECK_OK(ss_budget_ledger_stats(l, &s));
  CHECK_EQ(s.kind[SS_BUDGET_DEVICE].current, uint64_t(0));
  CHECK_EQ(s.kind[SS_BUDGET_PINNED].current, uint64_t(0));
  CHECK_EQ(s.refusals, 1u);
  ss_budget_ledger_free(l);
  // exactly the peak is enough
  CHECK_OK(ss_budget_ledger_create(device_peak, 0, &l));
  {
    Run r;
    if (!prepare(r, fx)) return;
    r.o.budget = l;
    CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  }
  ss_budget_ledger_free(l);
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }
