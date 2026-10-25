/* tierstream adaptive memory plan (C ABI, additive; nothing here changes the behaviour of ts.h / ss.h calls).
 *
 * USER rule (2026-10-25): memory is one budget per GPU and per host, traded between the expert arena, the KV/state
 * and the context length. The operator picks the context (or a speed vs context trade-off); the solver sizes the state
 * for it and gives the rest to the expert tiers VRAM -> pinned RAM -> NVMe, the least-used experts going to NVMe
 * first by a measured usage table.
 *
 *   usage table   tierstream/usage-v1: per (layer, unit) selection counts, micro-selections per token and a decayed
 *                 recency; deterministic ranking (use desc, recency desc, layer asc, unit asc). Built from recorded
 *                 routing traces (ts-adapt usage) or online (ts_usage_attach: the context feeds every ss_layer_routes
 *                 into the table; off unless attached).
 *   tiers         ts_usage_tiers: RESIDENT for the top `active`, PINNED for the next `host`, DISK (file home) for the
 *                 tail, ready for ts_plan_from_tiers (and for the device-gated mode's planner).
 *   solver        ts_adapt_solve: tierstream/adapt-solve-v1 -> tierstream/solver-v1 (one row per operator choice).
 *
 * Errors: ts_last_error() as for every other call. Strings returned are freed with ts_free. */
#ifndef TIERSTREAM_TS_ADAPT_H
#define TIERSTREAM_TS_ADAPT_H

#include "ts.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TS_ADAPT_ABI_VERSION 1

typedef struct ts_usage ts_usage;

/* Empty table for a model shape (units per layer; layers with 0 units are allowed and never ranked first).
 * half_life_tokens: recency halves every that many tokens (0 = no decay: recency = count x 10^6). */
TS_API ts_status ts_usage_new(const char* model, const uint32_t* units_per_layer, int n_layers,
                              uint32_t half_life_tokens, ts_usage** out);
/* Same shape as a descriptor (every layer's unit count). */
TS_API ts_status ts_usage_new_for(const ts_desc* d, uint32_t half_life_tokens, ts_usage** out);
TS_API ts_status ts_usage_from_json(const char* text, size_t len, ts_usage** out);
TS_API ts_status ts_usage_from_file(const char* path, ts_usage** out);
TS_API void ts_usage_free(ts_usage* u);
/* Canonical JSON (byte-identical for the same observations) and the sha256 of exactly those bytes. */
TS_API ts_status ts_usage_to_json(const ts_usage* u, char** out_json, size_t* out_len);
TS_API ts_status ts_usage_sha256(const ts_usage* u, char out_hex[65]);

/* Online update: the router's ids of one layer for the current token(s), then one tick per decoded token. */
TS_API ts_status ts_usage_observe(ts_usage* u, int layer, const int32_t* ids, int n);
TS_API ts_status ts_usage_tick(ts_usage* u, uint32_t rows);
/* Attach to a host-mode context (NULL detaches): every ss_layer_routes / ss_layer_routes_dev of a forward is observed
 * (ss_layer_all is not: dense layers carry no routing) and ss_end ticks once per row of the forward (the largest
 * rows_per_layer given to ss_begin, else 1). The table must have the descriptor's shape and outlive the attachment;
 * the context never frees it. The table has no lock: while attached it is used only on the engine thread (read it,
 * serialize it or merge into it after detaching, or between forwards on that thread). Attach / detach between
 * forwards only (SS_E_INVALID inside one). Device-gated contexts: the engine feeds ts_usage_observe / ts_usage_tick. */
TS_API ts_status ts_usage_attach(ts_ctx* c, ts_usage* u);

TS_API int64_t ts_usage_total_units(const ts_usage* u);
TS_API uint64_t ts_usage_tokens(const ts_usage* u);
/* Per global unit (layer-major, as ts_desc_unit_index): micro-selections per token, decayed recency, raw count. */
TS_API ts_status ts_usage_unit(const ts_usage* u, int64_t global_unit, uint64_t* use_micro, uint64_t* recency_micro,
                               uint64_t* count);
/* Global unit indices hottest first (n = ts_usage_total_units). */
TS_API ts_status ts_usage_rank(const ts_usage* u, int64_t* order, size_t n);
/* tiers[n] (n = total units): top `active` RESIDENT, next `host` PINNED, the rest DISK. */
TS_API ts_status ts_usage_tiers(const ts_usage* u, int64_t active, int64_t host, uint8_t* tiers, size_t n);
/* Adds b into a (same shape and half life). */
TS_API ts_status ts_usage_merge(ts_usage* a, const ts_usage* b);

/* Tier solver: input = tierstream/adapt-solve-v1 JSON (paths inside resolved against base_dir, may be NULL);
 * output = tierstream/solver-v1 canonical JSON. */
TS_API ts_status ts_adapt_solve(const char* input_json, size_t len, const char* base_dir, char** out_json,
                                size_t* out_len);

#ifdef __cplusplus
}
#endif

#endif /* TIERSTREAM_TS_ADAPT_H */
