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

// Pinned host copies (tiers PINNED and RESIDENT_BACKED), packed in model order at 4 KiB strides into chunks of at most
// 1 GiB (one chunk holds a larger unit alone). The planner bills exactly what the context allocates.
struct PinnedLayout {
  std::vector<uint64_t> chunk_bytes;
  std::vector<int32_t> chunk_of;    // per global unit, -1 = no pinned copy
  std::vector<uint64_t> offset_of;  // per global unit, inside its chunk
  uint64_t total = 0;
};
PinnedLayout pinned_layout(const Desc& d, const std::vector<uint8_t>& tier);

// EXCLUSIVE tier mode: free pinned blocks of one host size (design.md §3.4).
struct SpareClass {
  uint64_t block_bytes = 0;  // host stride of the units sharing these blocks
  uint32_t asked = 0;        // host_spare_units, or the auto count
  uint32_t blocks = 0;       // what the host budget holds after every pinned unit (coverage first)
};
// Spare blocks (empty unless tier_mode is EXCLUSIVE), ascending block size, from the host bytes left after
// host_fixed and the pinned tier: they never displace a unit's home.
std::vector<SpareClass> spare_classes(const Desc& d, const ss_budget& b, uint64_t host_left);
uint64_t spare_bytes(const std::vector<SpareClass>& s);

struct Plan {
  std::vector<uint8_t> tier;  // per global unit: ss_tier
  ss_budget budget{};         // the budget the plan was made with (ss_open checks it)
  uint64_t cap = 0, margin = 0, fixed = 0, dynamic = 0, ring = 0, resident = 0, pinned = 0, disk = 0, peak = 0;
  uint64_t n_resident = 0, n_pinned = 0, n_disk = 0, n_pair = 0, n_backed = 0;
  std::vector<RingClass> rings;      // one stream ring per unit size that streams (empty when nothing streams)
  std::vector<SpareClass> spares;    // EXCLUSIVE: free pinned blocks (billed in `pinned`)
  uint32_t dynamic_slots_per_layer = 0;
  std::vector<std::string> notes;
};

constexpr uint64_t kGiB = uint64_t(1) << 30;

uint64_t plan_margin(const ss_budget& b);
bool same_budget(const ss_budget& a, const ss_budget& b);
// benefit: per global unit (size total_units) or empty for the descriptor's.
Plan plan_make(const Desc& d, const ss_budget& b, const std::vector<double>& benefit);
// Tiers decided elsewhere (hyperloom vramplan, a test, an engine flag): same totals and checks as plan_make.
Plan plan_from_tiers(const Desc& d, const ss_budget& b, const std::vector<uint8_t>& tiers);
// The planner's arithmetic for placements decided outside plan_make (recipes, docs/recipes.md §5.2).
uint64_t host_stride(uint64_t bytes);  // pinned host bytes of a unit (4 KiB stride)
// Host read-ahead buffer bytes: the largest host stride among the streamable, non-pair units whose tier is not PINNED
// (every unit that can be read from its file); 0 when there is none.
uint64_t ra_buffer_bytes(const Desc& d, const std::vector<uint8_t>& tier);
// cap - margin - fixed - workspace - dynamic-slot bound (throws SS_E_CAPACITY like plan_make when negative).
uint64_t plan_room(const Desc& d, const ss_budget& b);
// Device bytes of the stream rings for these streamed units (global indices).
uint64_t plan_ring_bytes(const Desc& d, const ss_budget& b, const std::vector<int64_t>& streamed);
// plan_make's pinned_backing step: RESIDENT -> RESIDENT_BACKED with the host budget left.
void plan_backing(const Desc& d, const ss_budget& b, const std::vector<double>& benefit, std::vector<uint8_t>& tier);
std::vector<uint8_t> tiers_from_json(const Desc& d, const Json& plan_json);
Json plan_to_json(const Desc& d, const Plan& p);

}  // namespace ss
