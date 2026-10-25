# Adaptive memory plan

USER rule (2026-10-25): memory is one budget per GPU and per host, traded between the expert arena, the KV/state and
the context length. The operator picks the context (or a speed vs context trade-off); the planner sizes the state for
that context and gives the rest to the experts across tiers VRAM -> pinned RAM -> NVMe; a measured usage table ranks
the experts so the least used go to NVMe first; prefetch looks ahead to the next layer(s).

Everything here is additive (`TS_ADAPTIVE`, default ON): no existing call changes behaviour; the context's usage hook
is off until `ts_usage_attach`.

## Pieces

| piece | where | what |
|---|---|---|
| usage table | `src/adapt_usage.*`, `ts_usage_*` | `tierstream/usage-v1`: per (layer, unit) selection count, micro-selections per token (`rdiv(count x 10^6, tokens)`), decayed recency (each selection adds 10^6, halves every `half_life_tokens`; Q32 fixed point, factor stored in the file). Canonical JSON + sha256. Rank: use desc, recency desc, (layer, unit) asc (placement contract 3.1). |
| builder | `ts-adapt usage TRACE [--filter train-]` | from a recorded route dump (`# seq <name> rows N` + one `layer:ids` line per token) |
| online updater | `ts_usage_attach(ctx, table)` | every `ss_layer_routes[_dev]` observed, one tick per row at `ss_end` |
| tiers | `ts_usage_tiers`, `ts-adapt tiers` | top `active` RESIDENT, next `host` PINNED, tail DISK (file homes), for `ts_plan_from_tiers` |
| simulator | `src/adapt_sim.*`, `ts-adapt sim / xval / curve` | device-gated cache semantics (global arena, free-slot admission else bank, boundary LRU to the spare target, write-back latency, FILE victims discarded); counts misses by home; `tierstream/miss-curve-v1` |
| what-if solver | `src/adapt_solve.*`, `ts_adapt_solve`, `ts-adapt solve` | `tierstream/adapt-solve-v1` -> `tierstream/adapt-whatif-v1` (NOT `tierstream/solver-v1`: the operator solver `ts-recipe solve` is claudeadmin-placement's): per choice (mode, context, batch) charges the device lines + state for that context, arena = rest of the device budget, pinned = host budget after aux, spare blocks and (when there is a FILE tier) `file_staging_bytes`, NVMe = tail; tok/s by the placement contract 3.9 arithmetic (curve + cold-rate bound) and, given a trace, by simulating that exact (active, file) pair; rows with batch > 1 are unpredicted. The device-gated mode's spares and 2 x B bank slots are NOT added by the solver: give them as device or mode lines (the DS input does: `spares_96`, `banks_guard`) |
| contract arithmetic | `solve_ref`, `ts-adapt ref` | placement contract 3.9 exactly as `golden/solver_ref.py` (sha256 5722f6e2): sequences == 1, both applicability key sets complete and equal to the choice's `calib`, missing curve -> unpredicted; frozen goldens SV1 / SV1b in `tests/data/adapt`, byte for byte |
| bench | `ts-bench-adapt` (CUDA) | replays routes through a real descriptor with usage-ranked homes, compute stub per layer, next-layer lookahead (`ss_host_prefetch` for disk units), bitwise sample check |

The arithmetic of `solve_ref` equals `wave4/placement/golden/solver_ref.py` (frozen goldens in `tests/data/adapt`).

## Simulator scope

Modelled as the device-gated core does it: warm start from the ranking, demand misses take a free slot else are served
by the bank without admission, boundary eviction to the spare target, write-back latency in windows, FILE victims
discarded. Not modelled: prefetch / prediction, resurrection of a unit evicted in the same token, `keep`,
`evict_short`, the B-bank size limit, write-backs completing mid-token; a partial last window of a sequence counts as
a full window. Curves are therefore a miss model, not a timing model.

## Determinism

Counts, uses and recency are integers. The decay factor is `round(2^32 x 2^(-1/half_life))`, computed once with the
platform's `exp2` and **stored in the table** (`decay.factor_q32`); readers use the stored factor, so a table and its
ranking are byte-identical everywhere once written. Two machines building a table from the same trace agree unless
their `exp2` differs in the last bit at that half life (the factor is then visibly different in the file).

## Threads

The usage table has no lock. Attached to a context it is used on the engine thread only, between `ss_begin` and
`ss_end`; attach / detach between forwards (refused inside one); read, serialize or merge it after detaching or
between forwards on that thread.

## Exactness

Placement only moves bytes: every tier hands the engine the file's bytes. `test_adapt` checks the DISK tier against
the PINNED tier bit for bit on the CPU backend with lookahead read-ahead, and fail-closed reads (a unit whose hash
does not match is never served, later requests fail again, other units stay exact).

## Not here

The device-gated mode's FILE homes (`file-homes.md`, branch `file`, claudesoporte-wstream) and the operator-facing
`ts-recipe solve` / `ts_place` (claudeadmin-placement) consume the usage table and the solver; their lookahead stage
is proposed in `wave4/adaptive/file-prefetch.md`.
