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
| solver | `src/adapt_solve.*`, `ts_adapt_solve`, `ts-adapt solve` | `tierstream/adapt-solve-v1` -> `tierstream/solver-v1`: per choice (mode, context, batch) charges the device lines + state for that context, arena = rest of the device budget, pinned = host budget after aux + spare blocks, NVMe = tail; tok/s by the contract rev 5 3.9 integer arithmetic (curve + cold-rate bound) and, given a trace, by simulating that exact (active, file) pair |
| bench | `ts-bench-adapt` (CUDA) | replays routes through a real descriptor with usage-ranked homes, compute stub per layer, next-layer lookahead (`ss_host_prefetch` for disk units), bitwise sample check |

The arithmetic of `solve_ref` equals `wave4/placement/golden/solver_ref.py` (frozen goldens in `tests/data/adapt`).

## Exactness

Placement only moves bytes: every tier hands the engine the file's bytes. `test_adapt` checks the DISK tier against
the PINNED tier bit for bit on the CPU backend with lookahead read-ahead, and fail-closed reads (a unit whose hash
does not match is never served, later requests fail again, other units stay exact).

## Not here

The device-gated mode's FILE homes (`file-homes.md`, branch `file`, claudesoporte-wstream) and the operator-facing
`ts-recipe solve` / `ts_place` (claudeadmin-placement) consume the usage table and the solver; their lookahead stage
is proposed in `wave4/adaptive/file-prefetch.md`.
