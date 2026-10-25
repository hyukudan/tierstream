// Adaptive memory plan, part 3: the reference tier solver (tierstream/adapt-solve-v1 -> tierstream/solver-v1).
//
// USER rule (2026-10-25): memory is one budget per GPU and per host, traded between the expert arena, the KV/state
// and the context length. For each operator choice (context, mode, batch) the solver
//   1. charges every fixed device line, the mode's lines and the state for that context (never extrapolated unless
//      the input says the state is a linear model, and then the row is tagged),
//   2. gives the rest of the device budget to the expert arena (active units), the pinned host budget (after its
//      auxiliary charges and spare blocks) to the next units of the usage ranking, and the tail to NVMe (FILE homes),
//   3. predicts tok/s with the integer arithmetic of placement contract rev 4 §3.9 (golden/solver_ref.py) from a
//      miss curve, the usage table's cold rate and measured unit costs; when a routing trace is given it also
//      simulates the exact split of misses between pinned and file homes for that row's (active, file) pair.
// Nothing is chosen silently: rows keep the input order and every number says where it came from.
#pragma once

#include <string>

#include "adapt_usage.hpp"
#include "json.hpp"

namespace ts {
namespace adapt {

// input: an adapt-solve-v1 document (paths inside it are resolved relative to `base_dir`). Returns solver-v1.
Json solve(const Json& input, const std::string& base_dir);

// The contract §3.9 prediction on its own (the same JSON as golden/solver_ref.py: curves, usage, costs, choices).
Json solve_ref(const Json& fixture);

}  // namespace adapt
}  // namespace ts
