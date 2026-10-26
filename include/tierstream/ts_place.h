/* tierstream per-GPU placement (placement-v1; docs/recipes.md §5.4, §8.5). Added after the strix-stream rename:
 * these functions exist only under their tierstream names (include/strix_stream/ss.h is frozen by the rename's
 * byte-identity gate). Included by include/tierstream/ts.h.
 *
 * Every char** output is NUL-terminated and freed with ss_free / ts_free (a length pointer may be NULL); on failure
 * nothing is written to any output. A refusal returns SS_E_RECIPE and ss_last_error() lists one "PL_E_<CODE> ..."
 * line per error of the failing validation phase (V1 schema, V2 binding, V3 descriptor, V4 feasibility). */
#ifndef TIERSTREAM_TS_PLACE_H
#define TIERSTREAM_TS_PLACE_H

#include "../strix_stream/ss.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Plans a placement-mode route of the effective recipe: binds the probe report (§8.2; placement devices carry
 * uuid / pci / arch / driver), checks and reads the engine's tierstream/components-v1, and returns
 * tierstream/placement-v1 (canonical JSON, placement_sha256) and, when explain is non-NULL, the explanation text.
 * Optional inputs (NULL = absent): desc, the desc-v1 in use (checked against components-v1 desc_sha256, ranks units
 * by its benefit without usage-v1; the S5 library compile, tiers_plan_id and desc_projection_sha256 need the
 * device-gated context and are not in this build); usage_json, tierstream/usage-v1 (the unit rank, usage_sha256); context_json,
 * tierstream/context-v1 (measured context bytes: S2 charges max(allowance, measured), context_sha256). */
SS_API ss_status ts_place(const ss_recipe* r, const char* route, const char* probe_json, size_t probe_len,
                          const char* components_json, size_t components_len, const ss_desc* desc,
                          const char* usage_json, size_t usage_len, const char* context_json, size_t context_len,
                          char** out_json, size_t* out_len, char** explain, size_t* explain_len);

/* The engine's sugar flags (argv, without the program name): --anchor DEV, --gpu-budget-gib DEV=X[,...],
 * --device-map COMP=DEV[,...], --host-pinned-gib X, --host-pageable-gib X, --host-available-floor-gib X,
 * --margin-gib X, --margin-fraction X. DEV = gpuN (PCI order) | uuid:<GPU-...> | pci:<dddd:bb:dd.f> |
 * name:<substring>. The same placement as the recipe these flags make (route "flags"). */
SS_API ss_status ts_place_from_flags(const char* probe_json, size_t probe_len, const char* components_json,
                                     size_t components_len, int argc, const char* const* argv, char** out_json,
                                     size_t* out_len, char** explain, size_t* explain_len);

/* Budget solver (§3.9): inputs_json = {curves (tierstream/miss-curve-v1), costs (tierstream/solver-costs-v1),
 * usage (tierstream/usage-v1, optional), usage_lang ({lang: usage-v1}, the FILE rate per language), choices: [{id, mode: "plain"|"spec", R, ctx, lang, sequences?,
 * components: <components-v1 for that configuration>}]}. Returns tierstream/solver-v1 (one row per choice, in order:
 * the placement or its refusal, the predicted tok/s and its tag) and, when table is non-NULL, the same as a table.
 * Optional desc and context documents (embedded or local file paths) may be given on inputs or each choice and
 * are passed to place unchanged. Nothing is ranked or picked by default. */
SS_API ss_status ts_solve(const ss_recipe* r, const char* route, const char* probe_json, size_t probe_len,
                          const char* inputs_json, size_t inputs_len, char** out_json, size_t* out_len, char** table,
                          size_t* table_len);

/* Explicit operator pick: same solver output as ts_solve, plus recipe-overrides-v1 text. The selected row must be
 * feasible without hypothetical capabilities. Re-plans the emitted overrides, comparing device/host allocations,
 * units and seed (ignoring hashes and quota provenance); a difference is PL_E_STALE. No files are written. */
SS_API ss_status ts_solve_pick(const ss_recipe* r, const char* route, const char* probe_json, size_t probe_len,
                              const char* inputs_json, size_t inputs_len, const char* row_id,
                              char** overrides, size_t* overrides_len, char** out_json, size_t* out_len,
                              char** table, size_t* table_len);

#ifdef __cplusplus
}
#endif
#endif /* TIERSTREAM_TS_PLACE_H */
