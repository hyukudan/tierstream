// Tier planner (docs/design.md §5). Same rule as hyperloom vramplan's plan() with the third tier, so a descriptor
// gives the same placement whether hyperloom plans it or the engine does through ss_plan_make.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "desc.hpp"
#include "json.hpp"
#include "strix_stream/ss.h"

namespace ss {

struct RingClass {
  uint64_t stride = 0;  // slot bytes (a unit stride)
  uint32_t slots = 0;
};

struct Plan {
  std::vector<uint8_t> tier;  // per global unit: ss_tier
  uint64_t cap = 0, margin = 0, fixed = 0, dynamic = 0, ring = 0, resident = 0, pinned = 0, disk = 0, peak = 0;
  uint64_t n_resident = 0, n_pinned = 0, n_disk = 0, n_pair = 0;
  std::vector<RingClass> rings;      // one stream ring per unit size that streams (empty when nothing streams)
  uint32_t dynamic_slots_per_layer = 0;
  std::vector<std::string> notes;
};

constexpr uint64_t kGiB = uint64_t(1) << 30;

uint64_t plan_margin(const ss_budget& b);
// benefit: per global unit (size total_units) or empty for the descriptor's.
Plan plan_make(const Desc& d, const ss_budget& b, const std::vector<double>& benefit);
// Tiers decided elsewhere (hyperloom vramplan, a test, an engine flag): same totals and checks as plan_make.
Plan plan_from_tiers(const Desc& d, const ss_budget& b, const std::vector<uint8_t>& tiers);
std::vector<uint8_t> tiers_from_json(const Desc& d, const Json& plan_json);
Json plan_to_json(const Desc& d, const Plan& p);

}  // namespace ss
