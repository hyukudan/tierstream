# Example recipes (`strix-stream/recipe-v1`, calibrated)

Nine example recipes for the strix-stream recipe schema. They are written against `docs/recipes.md`, the final core
contract on branch recipes/v1, which is `wave4/recipes/core/schema-draft-5.md` plus the core's answers in the
coordination log:
- `routes.<id>.enabled`
- `h2d_gbytes_ps_min`
- uncached `prompt_tokens`
- `disk_available`
- `idle_preload_layers` and `max_copies_per_token`
- `host_free_commit_floor_gib`
- `tier_mode`

They cover today's lab and two hardware variants:
- **Today's lab:** two Strix Halo nodes on USB4, and a PC with an RTX PRO 6000 96 GB, 128 GB of RAM and an SN850X NVMe.
- **A 32 GB discrete card** in the same PC.
- **The same PC with ≥ 192 GB of host RAM.**

Each file covers one model on one hardware set. The files are in two folders:
- **`examples/`:** deployable today against the measured setups.
  - A route that still waits for something ships with `enabled = false`, and its file says what to wait for.
  - An override turns it on, for example `--set routes.prefill_long_rtx.enabled=true`.
- **`examples/pending/`:** the whole recipe waits for something: an engine run, a descriptor adapter, a card nobody
  has measured, or hardware the lab does not have. The blocker is in the header and in `x-status`.

All nine pass the core's `ss-recipe validate` (exit 0) and this folder's `validate_v1.py`.

**Citations** use `x-` keys, which the core keeps but never reads:
- `[x-evidence.<key>]`: `ref = "file:line"` in the lab spec tree, plus the `quote` found on that line and a `note`.
- `[x-cites]`: maps a settable path to one or more evidence keys.
- `[[x-predicted]]`: route, metric, value, `basis`, scope and evidence.

`basis` is one of:
- `measured`: a receipt of this configuration, with its scope stated;
- `derived`: arithmetic on measured inputs;
- `projected`: a model output for this configuration;
- `lower_bound` or `upper_bound`;
- `target`: an acceptance gate, not a model.

**What the checker establishes, and what it doesn't:**
- Every calibrated threshold field present in a file has a citation. That includes unit-group `layers` and `list`.
- Every quote is checked verbatim on its line.
- Not cited:
  - defaults;
  - `max_batch` / `batch_* = 1`, the one-prompt engine contract;
  - zero-valued "off" knobs.
- Whether a citation actually supports its field is review, not checking; Sol has done that review in rounds 1-4.

## Deployable (`examples/`)

| file | routes and placement | predictions (basis) |
|---|---|---|
| `deepseek-v4.1-flash.lab.toml` | **Pair:** TP prefill and decode. The long-context line: 33,024 positions, 2 sequences. **RTX PRO prefill:** prompts ≥ 2323 uncached tokens, hand-off to the pair; `enabled = false` until the quality receipt of its composed identity. | **Decode:** 33.9 tok/s at R1 (candidate cab238d6, 1-2 runs) [measured]. **Pair TTFT:** 2.20 s at 1K, 10.78 s at 8K, 39.55 s at 32K [measured]; 3.81 s at 2K [derived]. **RTX:** 4.53 s at 8K [derived, receipt-conditional]. |
| `deepseek-v4.1-flash.lab-hetero.toml` | **R1 decode:** the RTX owns exactly the measured W1 `placement-pro80-f0` set: 7,600 experts in 38 per-layer list groups, copied verbatim. The pair owns the rest. **Prefill:** on the pair. One decode configuration. | **Static placement:** 33.6 tok/s [measured], about the same as today's pair (33.9). **Refill mode:** 40.1 [measured]. It needs a fetch-from-peer tier that v1 lacks. |
| `minimax-m3.1-flash.lab.toml` | **Pair:** TP2, ≤ 9,216 context, ≤ 4 sequences. **RTX PRO prefill:** prompts ≥ 4K, with the t7 placement exactly: 32 device / 16 pinned / 12 NVMe blobs, in t7's list order. It is `enabled = false` until a ranked-order MiniMax descriptor exists; today's adapter is prefix-ordered, and the core refuses a spread list on it. Its disk tier is gated by `disk_available`, which fails closed and is set by the orchestrator from the disk duty rule. | **TTFT at 8K:** 16.1 s vs 45.0 s [measured, t7]. **TTFT at 1K:** 5.0 s on the pair [measured]. **Decode:** 20.4 tok/s; 36.79 at N4 (older binary) [measured]. **VRAM:** 89.07 GiB at 8K [measured, G6]. |
| `qwen3.8-flash-next.lab.toml` | **Replicas:** one per node, on the production line (8 slots × 12,288, T = 2048 prefill chunks), node workspace 7.2 GiB. **Prompts ≥ 1K:** the RTX PRO, fully resident, then hand-off. The block-schedule descriptor bills its workspaces as fixed bytes, so the RTX `workspace_gib` is 0. | **RTX TTFT:** 1.36 s at 1K, 2.94 s at 8K, vs 1.80 / 10.46 s on a node [measured]. **Two-replica serve:** C1 65.5, C8 163.5 tok/s wall aggregate [measured]. **VRAM:** 83.94 GiB [measured]. |

## Pending (`examples/pending/`)

| file | waits for | predictions (basis) |
|---|---|---|
| `minimax-m3.1-flash.morehostram.toml` | The ranked-order MiniMax descriptor, and a PC with ≥ 192 GB. | **TTFT:** 3.5 s at 1K, 12.3 s at 8K [derived from measured PRO compute; no live all-pinned run]. |
| `deepseek-v4.1-flash.lab-rtx-replica.toml` | An engine run on the PRO, and a replica lane in the orchestrator. Exclusive tiering is compile-refused by this core build. | **R1:** 55 tok/s [target = the spec's gate]. There is no model for this 8,000-slot budget. |
| `deepseek-v4.1-flash.card32.toml` | A 32 GB card measurement. Its route is also disabled until the DS quality receipt. | **At 32K:** 14.5 s vs 39.55 s on the pair. **At 16K:** 13.0 s [projected: the spec planner rerun at this room, `derived/`]. |
| `minimax-m3.1-flash.card32.toml` | A 32 GB card measurement, and the ranked MiniMax descriptor (its blob list needs it). | **At 8K:** 15.7 s vs 45 s [projected: planner rerun; fractional residency, PRO compute]. |
| `glm-5.3-flash.lab.toml` | The TP2 engine (P2). | **Decode:** ~24 tok/s [projected]; 41.5 [upper_bound]. **Peak:** 88.2 GiB/rank at 4 × 128K [projected]. |

## Thresholds and what stands behind them

- **DeepSeek 2323 tokens: a historical receipt, measured on uncached length.**
  - `prompt_tokens` in the request state is uncached length, by the core's definition.
  - The spec expects ~1.5K after the interleaved plan; re-measure the crossover when the route is enabled.
- **DeepSeek on a 32 GB card, 16K: the first modelled win.**
  - It comes from the planner rerun: 12.3 s compute + 0.7 s hand-off vs 20.4 s on the pair.
  - 8K loses: 12.1 s vs 10.8 s.
- **MiniMax 4K: main's route rule.**
  - It rests on two measurements: 1K loses on the RTX (7.4-7.9 s vs 5.9 s) and 8K wins (16.1 s vs 45 s).
  - 2K-4K is not measured.
- **MiniMax pinned count: 16, the t7 point.**
  - t7 derived it from free RAM at start.
  - In v1 the orchestrator or autotuner sets `host_pinned_gib` and `pinned`.
- **MiniMax context, 9,216:**
  - It is the measured 8K plus a 1K generation budget.
  - The VRAM for it is recomputed: ~89.74 GiB, which fits `device_gib` 93 after the default margin.
- **Qwen 1024: the router floor.**
  - The 549-token crossover is extrapolated below the measured 1K point.
  - The decode line caps context at 12,288. The 16K and 32K RTX receipts therefore exist but are not admissible here.
- **Qwen node workspace, 7.2 GiB: a conservative sum, not a receipt of the 8-slot line.**
  - The computed 8-slot state is 3.61 GiB.
  - Added to it is the whole measured engine reservation beyond the registered GGUF (~3.8 GB of GTT on
    single-request runs), counted as scratch.
  - The node minimum is 116 GiB of the ~117 GiB GTT, so that 103.4 GiB registered + 7.2 fits after the margin.

## Still outside v1

- **Hetero refill:** the measured 40.1 tok/s refills peer-owned experts into RTX slots. The `max_copies_per_token`
  knob exists, but a peer unit is never fetched, so there is no tier for it.
- **Exclusive tiering:** this core build refuses it at compile time.
- **One decode configuration per recipe:** hyperloom deploys only one. The pending replica file has two decode routes.
- **Descriptors come from the hyperloom adapters:**
  - `placement/ds41.py`, `minimax.py` and `qwen.py` supply fixed bytes, `host_fixed`, workspaces and benefit.
  - The DS RTX service plans its own cone residency, so its receipt is not this file's compiled plan.
- **Exactness:** a recipe has no exactness option. Every compiled placement is bit-identical by construction.

## Validate

```
D:\tmp\strix-stream-recipes\build\ss-recipe.exe validate FILE.toml   # the core (exit 0 for all nine)
python -I validate_v1.py "*.toml" "pending/*.toml"                  # structure + citations: all OK
python -I tests/make_tests.py                                        # regenerates tests/bad-*.toml (46 cases)
python -I validate_v1.py "tests/bad-*.toml"                          # each rejected by its intended rule
```

**What `validate_v1.py` checks:**
- Keys, types, ranges and references.
- Route rules and catch-alls; `max_batch` / `max_context_tokens`.
- Groups (overlaps, defaults, pinned without a host budget), links, and pins resolved in context against the §6.2
  whitelist.
- Citations.

**Against the core, on these 46 fixtures:**
- The core accepts exactly the 6 citation-only cases, since it does not read `x-` keys.
- It refuses every structural case, as `validate_v1.py` does. This is parity on the tested fixtures, not a proof
  of equivalence: the core is the authority.
- Descriptor-dependent compilation (tiering, capacity, rings) is `ss-recipe plan`.

**Other folders:**
- `derived/` holds the planner rerun behind the card32 numbers.
- `v0-draft/` holds the pre-core first draft and Sol-1's review of it.

**Shipping in strix-stream:**
- The `ref`s point into the lab's internal notes; one cited file name contains an internal name.
- Shipped copies keep `quote` and `note`, and point `ref` at the public receipts.
