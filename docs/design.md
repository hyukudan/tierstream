# strix-stream design (v1)

strix-stream is a model-generic weight-streaming library. An engine whose model is larger than its device keeps a
resident set of weight units in device memory. It streams the rest from a pinned host tier or from files read
unbuffered, and it reads every streamed or resident unit through one pointer table per layer. The library holds no
model or engine code. A model plugs in with a **descriptor** (§2) and a small **adapter** in the engine (§6).

Sources: the weight-streaming study (`wave4/wstream/study.md`), the RTX prefill spec §6 (`spec-rtx-prefill.md`, the
`ss_` names come from there unchanged), the DeepSeek and MiniMax streaming specs, and hyperloom vramplan's third
tier (the planner rule of §5).

## 1. Scope and non-goals

**In scope:**
- unit placement (resident / pinned / disk / pair);
- the resident set and its admission policy;
- prefetch (certain and speculative);
- the file → staging → device path;
- the H2D copy stream and its events;
- unit tables;
- stats.

**Not in scope:**
- kernels;
- format decoding;
- routing math (the engine runs its router, including the lookahead router);
- KV / state;
- inter-node transport. USB4 carries activations: a miss on the pair is computed by the peer, never moved as weights
  (study §0). The `pair` tier exists so a planner can say so; the library never fetches a pair unit.

**Exactness is not negotiable.** The library never changes bytes. Which slot holds a unit, and when it landed, reaches
the engine only through the table. Engine kernels must be slot-invariant (§6.2).

## 2. Descriptor `strix-stream/desc-v1`

The descriptor is a JSON manifest. Adapters or hyperloom emit it from the model files. The library reads only the
manifest, never GGUF or safetensors itself (hyperloom's `modelfiles.py` already has bounded header readers).

- Unknown keys are errors, except `x-*` keys, which are free-form adapter notes.
- All sizes are bytes, as exact integers (64-bit safe).
- `ss-plan DESC --desc` prints the canonical form: sorted keys, no whitespace.

### Top-level fields

| field | type | meaning |
|---|---|---|
| `schema` | `"strix-stream/desc-v1"` | required |
| `model`, `engine` | string | names, used in messages and receipts |
| `fixed_device` | `[{name, bytes}]` | always-device bytes not tied to a layer: embedding, head, global norms (vramplan `fixed`) |
| `host_fixed` | `[{name, bytes}]` | pinned host bytes in every placement, e.g. a PLE table (vramplan `host_fixed`) |
| `state_bytes_per_position` | u64 | KV/state per position, for planners. The library does not allocate state. |
| `workspace_bytes_per_row` | u64 | for planners. The engine passes its real workspace as `ss_budget.workspace_bytes`. |
| `unit_align` | u64, default 256 | device alignment of a unit in the pool and rings |
| `sources` | `[{name, path, align=4096}]` | files that hold unit bytes. `align` is the unbuffered read granularity. |
| `order` | `prefix` \| `ranked` | planner order rule. `prefix` = only a model-order prefix may be resident (`--resident N`, a layer cut). `ranked` = any subset, best benefit per byte first. |
| `tiers` | `pinned` \| `disk` \| `pinned\|disk` \| `pair` | where a non-resident unit may live (vramplan `Unit.host`) |
| `ring_slots` | u32, default 0 | slots per stream ring. 0 = 2 × the most units of any layer of that unit size. |
| `routes` | path | routing-frequency source (routes file) for hot ranking. Planners read it; the library does not. |
| `layers` | array, in order 0..n-1 | see below |
| `lookup_tables` | `[{name, source, offset, rows, row_bytes, placement}]` | row-addressed tables the engine reads itself (DS Engram, Qwen PLE). `placement` is `file`, `pinned` or `device`. Listed for planners and for the O_DIRECT rule of §4.4. |

### Layer fields

| field | meaning |
|---|---|
| `index` | must equal the position |
| `kind` | `dense`: only fixed bytes, no units. `moe`: routed units = experts. `blob`: exactly one unit = the whole layer's streamable bytes (MiniMax RTX blobs, layer-granular prefill). |
| `fixed_bytes` | always-device bytes of this layer: attention, norms, shared expert, router weight |
| `n_units`, `top_k`, `n_shared` | routed experts, experts per row, shared experts (informational; shared is inside `fixed_bytes`) |
| `unit_bytes` | engine bytes per unit = sum of `parts`. Computed if absent, checked if given. These are the bytes the engine stores, which can differ from file bytes (MiniMax repacks both TP halves): the adapter then points `parts` at a packed record file. |
| `parts` | `[{name, format, source, offset, stride, bytes}]`: unit u's part is `bytes` long at `offset + u × stride` in `source`. Parts lie back to back in the unit, in listed order. A GGUF stacked `*_exps` tensor is one part with `stride = bytes` per expert; a packed record file is one part. |
| `format` | `F32 F16 BF16 FP8_E4M3 FP8_E5M2 EXL3 MXFP4 NVFP4`, GGUF types (`Q2_K … Q8_0`, `IQ*`, `TQ*`), `mixed`, or `x-<custom>`. Informational: the library never decodes. |
| `router` | `{tensor, bias, format, score, rows, cols}`: the router the engine applies for lookahead (§3.3). `rows == n_units`. |
| `hash` | xxh64 per unit (16 lowercase hex digits) of the unit's engine bytes. Byte identity is checked against it (§4.5). |
| `benefit` | per unit, for the planner (e.g. measured route frequency). Default 1 for every unit. |
| `streamable` | default true. `false` = always resident; the units count as fixed. |

### Field mapping to vramplan's `Model` / `Unit`

This is what hyperloom's descriptor adapter implements, so that a new model needs a descriptor, not planner code.

| vramplan | descriptor |
|---|---|
| `fixed` | `fixed_device` + Σ `layers[].fixed_bytes` + non-streamable units + `workspace_bytes` |
| `units` | every (layer, unit) of streamable layers, `bytes` = unit stride, key `L<layer>.U<unit>` |
| `order` | `order` |
| `Unit.host` | `tiers` |
| `Unit.benefit` | `benefit`, or the routes file |
| `stream(streamed)` | Σ over unit sizes in the streamed set of `slots × size` (§5) |
| `host_fixed` | `host_fixed` |

### What the next big models need

These come from the published model configs as recalled. Check them against each `config.json` before writing a
real descriptor.

| model | layers | routed experts × top-k | shared | dense layers | descriptor notes |
|---|---|---|---|---|---|
| DeepSeek-V3 / R1 family | 61 | 256 × 8 | 1 | first 3 | `moe` layers with gate/up/down parts; the router is a sigmoid with a bias. MTP layer as an extra layer. |
| Kimi K2 | 61 | 384 × 8 | 1 | first 1 | same shape as DS-V3 with 384 experts. The FP8 block-scale tensors are extra parts. |
| Qwen3-235B-A22B (and Qwen3.x MoE) | 94 | 128 × 8 | 0 | none | `moe` everywhere. Softmax top-k router. |
| GLM-4.5 / 4.6 | 92 | 160 × 8 | 1 | first 3 | sigmoid router with a bias. |

Our current three (descriptor shapes):
- **DS V4.1 Flash**: 40 MoE layers, 384 × top-6, 9,953,280 B per expert from GGUF (three parts); Engram as two
  `lookup_tables`; the cone layers are given through `rows_per_layer`.
- **MiniMax M3.1 Flash**: 3 dense + 57 MoE layers, 128 × top-4.
  - Decode / expert streaming: 21.3 MB EXL3 experts.
  - RTX prefill: 60 `blob` units, 2,864,462,336 B per MoE layer (`prefix` order).
- **Qwen3.8-Flash-Next**: 48 layers, 512 × top-10; PLE as a `lookup_table` placed `pinned`.

## 3. Residency

### 3.1 Pool and rings

- **Resident pool.** Each layer with units has a persistent pool:

  ```
  capacity = planned residents + min(dynamic_slots_per_layer, n_units − planned)
  ```

  - Planned residents are loaded at `ss_open`.
  - With `pin_planned = 1` (prefill) they are never evicted.
  - With `pin_planned = 0` (decode) they are only the warm start of an adaptive set: our DS traces give LRU 95-98 %
    against 84-93 % for static placement at the same size (study §5).
- **Stream rings.** There is one ring per unit size, each with `slots × size` bytes. Every layer of that size shares
  the ring for its transient units: misses that are not admitted, whole-layer prefetch, and speculative reads that
  have no persistent slot.
  - A slot belongs to one layer from issue until that layer's `ss_release`.
  - A request that finds no free slot waits in a FIFO, which is drained at each release, in layer order.
  - `ss_acquire` of a layer that still has waiting units fails with `SS_E_CAPACITY`. It never hangs.
- **Size classes.** Rings are kept per unit size so that one blob never sizes the expert slots, and a DS ring stays at
  768 × 10 MB rather than 768 × the largest blob.

### 3.2 Admission

Admission runs per forward and per layer, on the distinct routed units. It is deterministic: every tie goes to the
lower id.

- **Hits** are protected: no miss of the same route can evict a hit.
- **LRU** (`SS_POLICY_LRU`): every miss is admitted. The victim is the least recently used unpinned resident that is
  not busy this forward. Hits refresh recency. When nothing is evictable, the miss is transient.
- **SCORED** (`SS_POLICY_SCORED`):
  - The unpinned residents outside the route and the misses compete for the rows they can hold (free slots plus those
    residents).
  - Rank: score desc, then last use desc, then id asc. The score is
    `0.7 × transition + 0.2 × window frequency + 0.1 × 1/(1+age)`, where:
    - `transition` = mean over the previous route's units p of P(u | p), from an n² count matrix;
    - window frequency counts the last `score_window` routes (default 16);
    - age is counted in routes.
  - Hits do not refresh recency; misses do.
  - Misses below the cut are transient. Residents below the cut are evicted only as needed.
  - It has to beat LRU on our traces (the hit-rate curves in wave4/vramplan/curves) before it becomes a default.
- **Seed** (`ss_seed`): load ranked units into free slots between forwards (usage profile, warm start). It never
  evicts.

### 3.3 Prefetch

- **Certain** (`SS_PREFETCH_CERTAIN`, `ss_prefetch_layer`) means the units will be used this forward:
  - layer-major prefill issuing layer L+k during layer L;
  - certain misses right after a barrier;
  - idle preload of the next request.
  - It is allowed between forwards (it opens the next forward) and is admitted like a demand.
- **Speculative** (`SS_PREFETCH_SPECULATIVE`) is the router lookahead:
  - The engine applies layer L+1's router to layer L's MoE input.
  - `ss_lookahead_select` turns the scores into a ranked union: per row top-k, union by (best rank, best score, id).
  - The first `lookahead_budget` (default 2) non-resident units are issued at lower priority. Workers take them only
    while no demand is queued and at most one demand is running.
  - A later route that includes one claims it (`spec_used`). Otherwise it is counted `spec_dropped`.
  - A speculative unit takes a persistent slot through admission (LRU or SCORED) when it can, else a ring slot. A
    speculative read that never started is cancelled at `ss_end`, and its slot is dropped.

### 3.4 Groups (hits first)

`ss_acquire_groups` returns at once:
- group 0 = units resident before this forward (no wait);
- groups 1..n = fetched units, ordered by route count desc then id, `group_units` per group.

The engine runs group 0, then calls `ss_group_wait(g)` before each later group. `ss_acquire` waits for everything.

## 4. Sources and the copy path

### 4.1 Pinned host tier

- Units the plan puts in the pinned tier are packed in model order into 1 GiB `cudaHostAlloc` chunks and filled at
  `ss_open`, read unbuffered from their files.
- A pinned → device copy is issued at once, on the engine thread, onto the copy stream: the copy stream waits for the
  slot's release event, the pieces go out (`copy_piece_bytes`, default 64 MiB), then the job event is recorded. There
  is no reader thread and no host wait.

### 4.2 Files: unbuffered reads + staging ring

- Files are opened with `O_DIRECT` on Linux and `FILE_FLAG_NO_BUFFERING` on Windows, falling back to buffered when
  the filesystem refuses.
- `reader_threads` (default 2) workers read each part's covering aligned range into one of `staging_buffers`
  (default 4 × 256 MiB) pinned buffers, in chunks of at most `staging_bytes − 2 × align`. On a discrete GPU, each
  chunk then goes H2D on the copy stream.
- A staging buffer is reused only after its last copy's event.
- Reads stop at end of file: an unbuffered read past EOF would fail on the unaligned offset.

### 4.3 Unified memory (Strix Halo, CPU backend)

- "Device" memory is host-mapped (`hipHostMalloc` mapped + coherent), so resident means mapped.
- A miss is a local NVMe read **in place**, straight into the slot, when the part's file offset, length and slot
  address are aligned. Otherwise it goes through the staging ring and a memcpy.
- There is no H2D and no pinned tier: plan with `host_pinned_bytes = 0`.
- Before writing a slot, the reader host-waits the slot's release event: the C-1 rule (the host must not write memory
  the GPU may still read).
- v1 `ss_acquire` host-waits the landed jobs. v2 gates the engine stream on a host-written mapped word
  (`hipStreamWaitValue64`) instead, so hits run while misses land.

### 4.4 Page cache

On unified memory the page cache **is** GPU memory. Never mmap a weight bank on Strix. Engines must also read
lookup tables (Engram) unbuffered (study §6.5).

### 4.5 Byte identity

- `verify_hash = 1` checks the xxh64 of every unit's bytes on the host side before they land: at the pinned fill,
  on the staging buffer, or in place.
- `ss_check_unit` (tests) copies the landed bytes back and compares them with a fresh read and with the descriptor
  hash.
- The CUDA test also checksums every routed unit **on the device through `table_dev`**. A missing stream wait shows as
  a wrong sum.

### 4.6 Events and hazards

| hazard | rule |
|---|---|
| a slot is overwritten while a kernel still reads it | Every write into a slot of layer L waits (on the copy stream, or on the host for unified backends) for L's release event. A ring slot waits for the release event of the layer that last used it. A slot is handed out only after its owner has recorded its release on the host side, so the wait is never on an unrecorded event. |
| the engine reads a unit before it lands | `ss_acquire` / `ss_group_wait` make the engine stream wait on each job's event, after the host has seen the job issued (discrete) or landed (unified). |
| the unit table is rewritten while in use | There are two device tables per layer, alternating per acquire. The host waits for the event of the previous H2D from that pinned mirror before rewriting it. The device copy is ordered on the engine stream. |
| jobs still in flight across forwards | `ss_end` drains the forward's jobs; speculative jobs that never started are cancelled. |

## 5. Planner (same rule as vramplan's third tier)

`ss_plan_make(desc, budget, benefit)` places units in four steps.

1. **Margin and room.**

   ```
   margin = max(margin_gib GiB, margin_fraction × cap)
   room   = cap − margin − fixed − dynamic
   fixed  = fixed_device + Σ layer.fixed_bytes + non-streamable units + workspace
   ```

2. **Residents.**
   - `prefix`: the longest prefix whose bytes plus the rings for the remaining units fit.
   - `ranked`: everything if it fits. Otherwise the rings for every unit size are reserved first, then units go in by
     best benefit per byte, ties in model order (greedy, exact for the equal-sized units of a layer).
3. **Non-resident units.**
   - `pinned` → pinned; refused if over the host budget (vramplan's refusal).
   - `disk` → disk.
   - `pair` → pair.
   - `pinned|disk` → pinned while the host budget lasts (after `host_fixed`), best benefit per byte first, else disk.
4. **Final check.** `peak = fixed + dynamic + rings(streamed) + resident ≤ cap − margin`.

`ss_plan_from_tiers` / `ss_plan_from_json` take tiers decided elsewhere and run the same totals and checks. This is
how an engine runs exactly the placement hyperloom planned (plan-v1 JSON: one `RPDX` letter per unit, per layer).

## 6. Engine contract

### 6.1 Calls

| call | when |
|---|---|
| `ss_desc_from_model(manifest)` / `ss_desc_from_json` | load |
| `ss_plan_make` or `ss_plan_from_json` | load (or once per request class: P bucket, cap) |
| `ss_open(desc, budget, plan, options)` | load: allocates the pool, rings, pinned tier and staging; fills residents and the pinned tier |
| `ss_begin(rows_per_layer)` | per forward. `rows_per_layer[L] == 0` marks a layer the forward skips (DS cone). |
| `ss_layer_routes(L, ids, n)` / `ss_layer_routes_dev(L, ids_dev, n, stream)` / `ss_layer_all(L)` | after router(L), or for a layer the forward needs whole |
| `ss_prefetch(L, ids, n, kind)` / `ss_prefetch_layer(L)` | lookahead / layer-major prefetch |
| `ss_acquire(L, stream, &view)` or `ss_acquire_groups` + `ss_group_wait` | before MoE(L). `view.table_dev[u]` is the device address of unit u. |
| `ss_release(L, stream)` | after the last kernel that reads L's units |
| `ss_end()` | end of forward: drains, stats, hook |
| `ss_stats_*`, `ss_set_stats_hook` | receipts (bytes pinned/disk, waits, hits, speculation) |

**Threading.** One engine thread drives a context. The library owns its reader threads. Every call returns an
`ss_status`, and `ss_last_error()` gives the message.

**Stream handles.** A `void*` stream is a `cudaStream_t` / `hipStream_t`. `nullptr` means the library's copy stream;
pass `cudaStreamLegacy` for the legacy default stream.

### 6.2 What the engine must guarantee (exact)

1. **Unit-table addressing.** Kernels read a unit only through `table_dev[u]`, never as base + u × stride.
2. **Order and slot invariance.** A unit's output bits depend neither on its slot, nor on which units share its
   launch, nor on the launch split or timing. The combine is a fixed-order step on the original ids. The gate is per
   engine: state bytes for every residency mix (all resident / interleaved / all streamed / routed) are equal to the
   all-resident run.
3. **Byte identity.** Streamed bytes equal resident bytes. The descriptor carries xxh64 per unit, checked at load and
   in test builds after every copy.
4. **The plan may depend on P and the budget; the bits may not.**

### 6.3 Per-engine status (from spec-rtx-prefill §6.4, spec-deepseek, spec-minimax)

| engine | uses | open work in the adapter |
|---|---|---|
| DS RTX prefill service | `ss_plan`/`ss_open` replace `ExpertStreamer` + `plan_streamed` + the host tier. `rows_per_layer` = cone. Routed fetch on cone layers. | none on the bits side (the MoE already runs range by range with the combine on ids) |
| DS RTX decode replica (wdsrtx) | dynamic slots (80 GiB LRU / SCORED), pinned cold tier, lookahead budget 2 | router L+1 on h_L in the engine; G0/G2 gates |
| MiniMax `mm-rtx-prefill` | `blob` units, `prefix` order, pinned + disk tiers (the NVMe ring that the live run needs) | `f32_moe` weight base becomes a table |
| Qwen RTX prefill | — | needs per-layer expert buffers behind a table (today it cudaMallocs every tensor) |
| Strix node replica (fallback) | HIP backend, `host_pinned_bytes = 0`, disk tier = local NVMe | v2 mapped-flag gate |

## 7. Stats

Per layer and in total, since open:
- `routed`, `hits`, `admitted`, `transient`, `prefetched`, `evicted`;
- `spec_issued`, `spec_used`, `spec_dropped`;
- `bytes_pinned`, `bytes_disk`;
- `wait_ns`: host time inside acquire / group_wait.

They balance: `hits + admitted + transient + prefetched + spec_used == routed`.

`ss_stats_to_json` gives `strix-stream/stats-v1`. The hook fires at every `ss_end` with the forward's numbers, which is
where a TTFT receipt takes its bytes and waits.

## 8. Backends

| backend | status | notes |
|---|---|---|
| CPU | done, tested | host memory, unified; the reference for the in-place path |
| CUDA | done, CPU-built; GPU test `ss-test-cuda` (one short PC-BIG) | VRAM pool and rings, `cudaHostAlloc` tier and staging, one non-blocking copy stream: one DMA engine reaches 31-32.5 GB/s and a second stream adds nothing (spec-rtx-prefill §2.1) |
| HIP | stub: compiles only with `-DSS_HIP=ON` on a ROCm host; not built or run yet | unified: slots are mapped host memory, misses are read in place from the local NVMe, host waits in v1 |

## 9. Versioning

- `SS_ABI_VERSION` = 1. Structs are passed by pointer and filled with `*_defaults`.
- A field added at the end is a minor change. Any other change bumps the ABI.
- The descriptor schema is versioned separately (`desc-v1`). Plan and stats JSON carry their own schema ids.

## 10. Planned (not in v1)

- **Part-granular tickets.** gate/up land before down, so their GEMV starts about a third earlier.
- **Coalesced reads.** Read adjacent units of the same part in one request (GGUF stacked tensors: consecutive ids
  are contiguous).
- **Mapped-flag gates** (`cuStreamWaitValue32` / `hipStreamWaitValue64`) instead of host waits for file jobs.
- **Per-layer row splits** from miss curves (dynamic slots per layer instead of a uniform count).
- **A rate-tally file across runs.** Measured to generalize poorly in the study, so only as an opt-in.
