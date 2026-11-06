// Budget solver (placement contract rev 7 §3.9): an operator trade-off report. Per choice (mode / max_rows,
// context, sequences) the route is planned by place() with the engine's components-v1 for exactly that
// configuration (S0-S5, S3e spill included) and its tok/s is predicted from measured inputs with integer arithmetic
// (reference golden/solver_ref.py, golden SV1). Rows keep the order of the choices; nothing is ranked or picked.
#pragma once

#include <string>

#include "components.hpp"
#include "json.hpp"
#include "recipe.hpp"

namespace ts {

// The prediction alone, on solver_ref.py's input shape: {curves: {"<lang>/R<r>": [[active_units, milli_misses]]},
// curves_applicability, usage: [coldest-first micro-selections per token], usage_lang: {lang: [the same units'
// micro-selections per token in that language, same order]}, costs: solver-costs-v1 (with `applicability`), choices: [{id, mode, R, ctx, lang, active_units, file_units, sequences, calib}]} -> {rows: [{id,
// tag, milli_misses_total, milli_misses_pinned, milli_misses_file, window_us, milli_tok_s}]}; an unpredicted row
// (sequences != 1, an incomplete or mismatching calibration, FILE units without the language's table, ...) has its
// numbers null. Rev 17: required keys compared only (drafter for spec rows, file_store for FILE rows), evaluation,
// probe_xfer vs costs.xfer_measured, lru for FILE rows, estimates per row, usage_lang_sha256 at the top.
Json solve_predict(const Json& inputs);

// cache_sha256 of a components-v1 (§3.9): sha256 of the canonical {elastic as written, config.cache_config}.
std::string cache_sha256_of(const Components& c);

// The full solver. inputs = {curves: tierstream/miss-curve-v1 ({schema, applicability, curves: {...}}) or the bare
// curve map + curves_applicability, costs: tierstream/solver-costs-v1, usage: the pooled tierstream/usage-v1 (optional;
// ranks the units and so chooses the cold set), usage_lang: {lang: usage-v1} (rev 16: the FILE rate of a row comes
// from its language's table over that cold set; usage-v1 documents embedded or named by a path), choices: [{id, mode:
// "plain"|"spec", R, ctx, lang, sequences?, components: <components-v1>}]}. The choice's calib is derived, never typed
// (a typed calib is refused): adapter, anchor_device / assist_device, xfer (sha256 of the tierstream/xfer-v1 profile
// from the probe rows' pci_path / link / via / numa_node, the probe's xfer_host block and the drafter link; "unknown"
// matches nothing), cache_sha256 ({elastic, config.cache_config} without dg_max_ids / banks / window_slots), build /
// content / arithmetic sha256, drafter (spec rows), file_store (FILE rows) / engram_store (the probe's), driver,
// platform, policy, prefetch, sequences, tier_mode, usage_sha256. Rev 17 gates as solver_ref.py: curves out of sample
// (`evaluation`), the choice's (else the probe's) probe_xfer within +-10 % of costs.xfer_measured, FILE rows need lru;
// rows carry `estimates` (costs.provenance). Each choice is planned on a copy of the effective recipe
// with routes.<route>.placement.<anchor>.max_context_tokens = ctx, routes.<route>.max_rows = R (1 for plain) and
// max_sequences when given; a choice refused only for pinned host while the engine lacks file_expert_tier is planned
// hypothetically with the capability and costs.file_staging_bytes of FILE staging (requires: [file_expert_tier]).
// Optional desc/context documents or paths on inputs (or a choice, taking precedence) pass through to place.
// An explicit pick returns the exact re-planned recipe-overrides-v1 bytes through overrides, refusing a material
// mismatch with PL_E_STALE; hypothetical rows cannot be picked. Default output is unchanged.
// Returns tierstream/solver-v1 (solver_sha256 over the object without that key).
Json solve(const RecipeStack& stack, const std::string& route, const Json& probe, const Json& inputs,
           const std::string& pick = "", std::string* overrides = nullptr);

// Compare the material plan (device/host lines, units and seed), ignoring hashes and line provenance that
// changes when an automatic quota becomes explicit. Refuses PL_E_STALE on any material difference.
void verify_picked_placement(const Json& expected, const Json& actual);

// The solver-v1 rows as a table (tok/s with three decimals).
std::string solve_table(const Json& solver);

}  // namespace ts
