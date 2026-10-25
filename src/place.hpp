// Per-GPU placement planner (wave4/placement contract rev 4, §3): placement-mode routes only. Legacy routes keep
// recipe_plan / budget_for / tier_units unchanged and never come here.
//
// Inputs: the effective recipe + route, the binding of a probe report, the engine's components-v1 (and optionally
// the desc-v1 / usage-v1 identities it is checked against). Output: tierstream/placement-v1 (canonical JSON, integer
// bytes, placement_sha256 over the object without that key) and an explanation printed from the same numbers.
// Validation runs in phases V1 schema, V2 binding, V3 descriptor, V4 feasibility (§3.7); errors are aggregated within
// the failing phase in a stable order (component priority, device order) and a later phase does not run.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "components.hpp"
#include "desc.hpp"
#include "json.hpp"
#include "recipe.hpp"

namespace ts {

struct PlaceInput {
  const Recipe* recipe = nullptr;
  std::string effective_sha256, route;
  const Binding* binding = nullptr;
  const Components* components = nullptr;  // nullptr: PL_E_MISSING_DESCRIPTOR
  const Desc* desc = nullptr;              // the desc-v1 in use: desc_sha256 check, benefit rank
  const UsageDoc* usage = nullptr;         // usage-v1: the unit rank (usage_sha256)
  const ContextDoc* context = nullptr;     // context-v1: measured context bytes (context_sha256)
};

struct PlaceResult {
  std::vector<std::string> errors;  // "PL_E_<CODE> ..." lines; empty = placed
  std::string phase;                // V1..V4 of the errors
  bool computed = false;            // doc carries the S2-S4 numbers (always when placed; also after a host refusal)
  Json doc;                         // tierstream/placement-v1
  std::string explain;              // ts-recipe place --explain
};

PlaceResult place(const PlaceInput& in);

// The engine's sugar flags (contract §4.1), as one placement-mode recipe planned by place():
//   --anchor DEV  --gpu-budget-gib DEV=X[,DEV=X]...  --device-map COMP=DEV[,COMP=DEV]...
//   --host-pinned-gib X  --host-pageable-gib X  --host-available-floor-gib X  --margin-gib X  --margin-fraction X
// DEV: gpuN (N = index among the local host's devices of the backend in PCI order), uuid:<GPU-...> (CUDA only),
// pci:<dddd:bb:dd.f>, name:<substring> (exactly one match); COMP=DEV also takes host_pinned / host_pageable / file /
// auto. The default anchor is the device runtime ordinal 0 resolves to, recorded by physical identity. Sets are
// named gpuN. recipe_text (may be null) receives the recipe (JSON) the flags make.
PlaceResult place_from_flags(const Json& probe, const Components& c, const std::vector<std::string>& args,
                             std::string* recipe_text);

// margin_fraction -> parts per million, rounded half up on the decimal text of the value (exact, no binary float).
uint64_t frac_ppm(double fraction);

}  // namespace ts
