# tierstream design (v1)

tierstream is a model-generic weight-streaming library. An engine whose model is larger than its device keeps a
resident set of weight units in device memory. It streams the rest from a pinned host tier or from files read
unbuffered, and it reads every streamed or resident unit through one pointer table per layer. The library holds no
model or engine code. A model plugs in with a **descriptor** (§2) and a small **adapter** in the engine (§6).

Sources: the weight-streaming study, the RTX prefill spec §6 (`spec-rtx-prefill.md`, the
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
- `ts-plan DESC --desc` prints the canonical form: sorted keys, no whitespace.

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
  - An evicted planner resident reloads from its file unless it has a pinned copy: tier `RESIDENT_BACKED` (letter
    `B`). With `ss_budget.pinned_backing = 1` the planner gives residents pinned copies from the host budget left after
    the pinned tier, lowest benefit per byte first. The DS decode replica ("80 GiB LRU + the rest pinned") needs this,
    or every eviction becomes an NVMe read.
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
  - It has to beat LRU on our traces (measured hit-rate curves) before it becomes a default.
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
  - By default a speculative unit lands **only in a ring slot** and never evicts a resident on a guess.
    `spec_persistent = 1` sends it through admission into a persistent slot instead. Turn that on only once a trace
    shows the extra evictions pay.
  - A speculative read that never started is cancelled at `ss_end`; a persistent slot it held is dropped.

### 3.4 Groups (hits first)

`ss_acquire_groups` returns at once:
- group 0 = units resident before this forward (no wait);
- groups 1..n = fetched units, ordered by route count desc then id, `group_units` per group.

The engine runs group 0, then calls `ss_group_wait(g)` before each later group. `ss_acquire` waits for everything.

### 3.5 Tier modes: inclusive and exclusive

`ss_budget.tier_mode` says whether a unit may be held twice.

- **Inclusive** (default, v1 behaviour): a unit promoted from the pinned tier into a device slot keeps its pinned
  copy. A resident evicted by the dynamic set is reloaded from its file next time, unless it has a `B` copy.
- **Exclusive**: at every forward boundary each unit is in exactly one place: a device slot, a pinned block, or only
  its file. Device + host bytes then cover `units` instead of `units + pinned copies of the slots`; for the DS RTX
  decode replica, 80 GiB of slots + ~63 GiB of blocks hold all 142.4 GiB of experts and no file is read after open.

Exclusive mechanics:
- **Blocks.** The pinned tier is a pool of blocks per host size (units with equal 4 KiB-strided bytes share blocks): the
  plan's pinned units, plus `host_spare_units` spare blocks per size (0 = min(64, units of that size)). Coverage
  comes first: spares take only the host bytes left after `host_fixed` and every pinned unit, so asking for spares
  never moves a unit to disk (the plan JSON gives `asked` and `blocks` per size). Blocks sit in ≤ 1 GiB chunks; a
  block larger than 1 GiB has a chunk of its own.
- **Promotion.** A miss admitted into a slot is copied from its block (or its file). At `ss_end` its block goes to
  the free list; it is reusable from the next forward.
- **Eviction.** Admission caps its evictions at the free blocks of the victims' size. Each victim is copied
  device → host from its slot into a free block on a second copy stream (the write-back stream, the opposite PCIe
  direction), and the admitted unit's copy into that slot waits for the write-back. A miss past the cap is served
  from the stream ring and its victim stays (`deferred`): no unit ever falls back to its file. With no spare and full
  slots the cache cannot adapt (every miss from the ring) but stays covered.
- **Discard** (`ss_options.exclusive_discard = 1`, default 0). A demand admission is not capped: the victims past
  the free blocks are discarded instead of deferred (no D2H, no block; exact, since weights are immutable and every
  unit's bytes are in its file) and the misses take their slots, so the cache keeps adapting with the pinned budget
  full. A speculative admission keeps the cap (a guess never discards). A discarded unit is file-homed from then on
  (fetched from its file; host read-ahead takes a unit only when its size fits the read-ahead buffers, which hold
  every size that has resident or disk units, the only sizes a discard can happen in). Blocks keep their units;
  `discards` counts the victims (not write-backs, no bytes). Set it after `ss_recipe_plan`, which rebuilds the
  options.
- **Speculation.** A speculative admission (`spec_persistent`) leaves half of the spares to demands.
- **Refused:** `pinned_backing` / tier `B` (the second copy), `fill_residents = 0` (a resident's slot is its only
  copy), the HIP backend (on unified memory a slot and a block are the same memory), descriptors whose tiers do not
  allow pinned; `exclusive_discard` with a descriptor whose tiers are `pinned` (its units must not live on disk).

Exclusive ordering (CUDA; the CPU backend copies synchronously):

| hazard | rule |
|---|---|
| a write-back reads a slot before its unit landed | every victim became resident in an earlier forward (or at open / seed, which synchronize); the write-back stream waits once per forward for `blocks_ev`, recorded on the copy stream at the previous `ss_end` after all of that forward's copies |
| a freed block is overwritten while an H2D still reads it | blocks freed in forward F are taken only from F + 1, after the same `blocks_ev` wait |
| the admitted unit overwrites the victim before the write-back read it | its copy into the slot waits for the write-back event (as well as the slot's release event) |
| a written-back unit is read again in the same forward (a certain prefetch evicted it, then the routes name it) | its copy out of the block waits for that write-back event |
| a block written in F is read in a later forward | `ss_end` makes the copy stream wait for the write-back stream's mark before recording `blocks_ev` |
| events reused | write-back events return to the pool at `ss_end`, after every wait on them is enqueued (a CUDA wait captures the record at call time); no wait cycle: every wait points at an earlier record |

### 3.6 Host read-ahead

Without it, a file-backed unit starts reading only once a device slot is free. With a 2-slot ring of 2.86 GB layer
blobs (MiniMax RTX prefill), that put every disk read on the critical path: ~0.45 s of acquire wait per disk layer at
8K. `ss_options.readahead_units = R` (default 0 = off) adds R pinned host buffers, each the size of the largest unit
that can be read from its file.

- **Requests.**
  - `ss_host_prefetch(_layer)(L)` asks for a host-only read of layer L's file-backed units, with no device slot. The
    engine calls it for L+1..L+H while layer L runs; recipe engine knob `host_prefetch_layers` sets H.
  - Units that wait for a ring slot are requested too.
  - Requests are idempotent per forward. A unit that is resident, in flight, computed by a peer or pinned is skipped.
- **Reads.** A reader reads the unit into a free buffer, with the hash checked as on every path, at a priority below
  demands. Unattached read-ahead reads use at most max(1, readers - 1) readers.
- **Landing.** When the unit gets a device slot (its ring slot, or a persistent slot when a certain fetch admits it),
  a copy job takes it from the buffer: H2D on the copy stream after the slot's release event and, in the exclusive
  mode, after the write-back of the victim that held a persistent slot; memcpy on unified backends. If the read has
  not finished, the copy is queued the moment it does, and a still-queued read is promoted to a demand. A buffer is
  reused only after its last copy's event.
- **Not used:** a speculative fetch abandons (persistent) or ignores (ring) the request and takes the ordinary
  path. A failed read falls back to an ordinary file read; it fails the forward only when a landing was
  already waiting on it.
- **Shutdown.** `ss_end` and `ss_close` sweep and drain every request: unused reads, waiting requests, parked and
  queued landings.
- **Stats.** `ss_stats_to_json` has `"readahead"`: requested, read, landed, unused, failed, fallback, bytes. File
  bytes count once in `bytes_disk`.
- **Cost.** R x buffer bytes of pinned host memory, billed by recipes in `resources.readahead_bytes`. The adapter
  must take them out of the host budget before placement.

### 3.7 Device-gated write-back scheduling

In device-gated decode (README, "Device-gated decode") each boundary evicts least recently used experts into free
pinned blocks. With `dg_wb_defer = 1` (the default) the boundary still chooses the victims, reserves their blocks and unpublishes
them in the table upload, but copies nothing. Each victim waits in a FIFO of records: unsubmitted, partly submitted
(only the oldest, with a part cursor), or submitted with its completion event recorded after its last part. Only
submitted records are ever queried for completion. The service thread submits parts after a layer's ready signal and
prefetches, on the demand stream by default (stream order keeps them behind the ready), under a per-layer byte credit
that may go into debt for a part larger than the budget. It submits nothing after the last gated layer, stops as soon
as the next layer has exported (checked before every part), and holds back while the bytes in flight reach a cap. A
demand without a free slot waits for the oldest write-back in flight, or with none finishes the oldest record, until
a persistent slot is free, before the usual bank fallback. A boundary counts deferred victims as in flight in its
spare-target arithmetic, carries up to the larger of its own and the previous boundary's victim count into the next
token, and submits the older excess after its table upload. A victim routed or predicted again
before any part was submitted is cancelled: its record goes, its block returns unread, and it is published again on
the demand stream as any resurrection. A partly submitted one gets its event after the parts already sent and is
discarded, so its block returns once those copies finish and the remaining parts are never sent. Exactness rests on
the existing rules: a victim's slot is not written until its write-back's event completes, the device never reads an
unpublished slot, and a block becomes a unit's home only after every part landed. Close drops unsubmitted records
without new device work and frees memory only after the submitted copies are drained.

### 3.8 Device-gated scored eviction

With `policy = SS_POLICY_SCORED` in device-gated decode only the boundary's choice of victims changes. Demand misses,
banks, prefetch, resurrection, write-backs (deferred or not), the table protocol and the device kernels are those of
LRU, so results are bit-for-bit identical and only residency (and so timing) differs.

`SS_POLICY_SCORED_AUTO` shares that scorer and learns on every cycle. `ss_dg_end_rows(c, R)` closes the completed
window (all verify rows, including rejected rows): AUTO selects scored victims only for `R > 1`, and walks the
unchanged LRU tail for R1. `ss_end` is one row. Between tokens, `ss_dg_set_policy(c, policy)` drains write-backs,
switches LRU / SCORED / AUTO, and resets learning to the open profile (or zero) and demand epochs while retaining
placement and recency. Open with SCORED or AUTO to provision tables for later switches; an LRU-only open refuses a
switch to scoring. `ss_dg_reset_open` restores the open policy as well as its learning and placement.

- **Eligibility** is LRU's exactly: a device home, no write-back in flight, not in the keep list. Units of the current
  token are eligible. Coming units are never victims: at the boundary each one has been admitted (then it is a device
  home) or reverted to its block.
- **One acceptance loop** serves both policies. It takes candidates from a policy order (LRU: the recency list from
  its tail; scored: the smallest eligible scores) and accepts an eligible one while `free + pending + accepted < target`
  and a free block exists; at the first victim without a block nothing more is accepted. `pending` counts the
  deferred write-back records and the in-flight ones that were not discarded. Only the accepted list feeds the
  deferred records, the boundary's copies and the carry bound. The pressure test override then walks the recency
  list from its head through the same loop and counts accepted victims.
- **The score**, per gated layer g, in integers. `T_g[r][e]` are uint32 counts that saturate at 2^32 - 1 and are
  never aged; each row sum is an exact uint64 that moves only when one of its cells does; `cnt_g[e]` counts the last
  16 unions of g that contain e. With `cur` the token's unique union of g's routed ids (every exported row included),
  `N = sum_{r in cur} T[r][e]`, `D = sum_{r in cur} rowsum[r]`:
  `P = ((|cur| * N) << 16) / max(1, D)`, `F = (cnt[e] << 16) / 16`, `Rec = (1 << 16) / (1 + epoch - demand_epoch[e])`,
  `S = w_t * P + w_f * F + w_r * Rec`. All divisions are unsigned floor divisions; with at most 65536 units per layer
  and 64 ids per export every intermediate stays below 2^60. The order everywhere is (S, demand epoch, recency
  stamp), ascending: a total order, because stamps are unique.
- **State and its lifetime.** The epoch starts at 0 and each token adds 1 (the first token is epoch 1). A demand use
  sets a unit's demand epoch; seeding, prefetch and resurrection never do. Every push to the recency list takes the
  next stamp of one uint64 counter, checked before each phase that can push (a token, a seed, a window hint): past a
  limit 2^40 below the maximum the phase fails before changing anything (the service's failure poisons the context),
  so the counter never wraps. The history (counts, rings, previous unions, epochs) lives as long as the context:
  `ss_stats_reset` keeps it and `ss_dg_reset_open` restores the open state (the profile's counts or an empty table,
  epoch 0, the open stamps). Toggling write-back deferral between tokens does not touch it.
- **Where the work runs.** The service's `serve_use` saves the unique union, updates the transition counts and ring,
  and stamps routed uses every cycle. After the next layer's prefetches and the write-back slice it tags the layer
  with the epoch before publishing terminal ownership. Ranking runs at the boundary, after admissions and collection
  establish final eligibility. A max heap retains only the needed smallest candidates in the same public total order;
  an F + Rec lower bound skips transition reads when a candidate cannot enter the heap. AUTO R1 and boundaries needing
  no victims do no ranking. A layer's union, epochs and stamps are final once it is served, so this produces exactly
  the original sorted-list / merge victims. That original implementation remains an internal CPU differential
  reference. `score_ns` includes boundary ranking; `score_candidates` counts eligible keys considered (including
  lower-bound rejections). Service `score_last_ns` now measures only tagging, not boundary ranking.
- **Fallback.** A failed or stopped token, a timeout or a device error poisons the context as before; the boundary
  never runs. If the handoff succeeded but some list is not tagged with this token's epoch, that boundary feeds the
  LRU order into the same loop and counts `score_fallbacks` (an internal-consistency guard that should stay at 0).
- **Profile.** The optional `dg_score_profile` holds starting counts: magic `TSSCPRF1`, version 1, the gated layer
  count and units, a 32-byte identity, a provenance string (at most 4096 bytes, reported in the stats JSON), the
  counts row-major per layer, then a SHA-256 of everything before it. The identity is the SHA-256 of the layer count
  and, per gated layer, its model layer index, units, unit bytes and the descriptor's per-unit xxh64 list, so a
  descriptor without unit hashes cannot load one. `ss_open` checks the exact size, magic, version, units, identity and
  checksum, rebuilds the row sums and loads the counts before the service starts. Any mismatch fails the open. The
  counts are kept for `ss_dg_reset_open`, so a profile doubles the tables' host memory (4 x units^2 bytes per layer,
  pageable).
- **Admission.** The tables are ordinary pageable host memory, not pinned, so the pinned host admission
  (`host_pinned_bytes`) does not bill them; `ss_open` only checks that their size fits the address space (with at most
  65536 units per layer). A recipe's host estimate for `policy = scored` already includes one transition table per
  layer; the profile's kept copy is not in it. On DeepSeek (40 layers x 384 units) that is 22.5 MiB, or 45 MiB with a
  profile.
- **Policy values.** Device-gated `ss_open` accepts LRU and SCORED and refuses any other policy value with
  `SS_E_INVALID`. Before this mode existed, SCORED itself was refused with `SS_E_UNSUPPORTED`.
- **Gates.** A brute-force oracle recomputes every boundary of the randomized core runs from the same state
  (deferred, partial and recorded write-backs, resurrections and collections after ranking, keep lists, changing
  targets, shortage, pressure) and must match exactly; injected faults (a pending victim, an ignored keep list, a stale
  list, ranking from overwritten scratch) must be caught. A replay drives the real core through a route trace and
  matches a port of the reference simulator token by token (misses and victims, both policies). On a DeepSeek decode
  route trace (8,576 residents, 96 spares, 49,430 tokens, five folds) both read 8.140 misses per token for LRU and 7.652
  for scored with an empty starting table. Profile-seeded counts reach 7.18 in the simulator.

### 3.9 Device-gated kept-resident layers and unexported layers

Layer-wide prefill (a grouped MoE kernel over every expert of one layer at once) runs through the same device-gated
cache as decode: one copy of each expert, arena slot <- pinned block <- file. Per layer g, between tokens:
`ss_dg_make_resident(c, g, ids, n)` (every unit of g), then one step that exports no other layer
(`rows_per_layer[h] = 0` for h != g) and reads g's units through `ss_dg_slot`, then `ss_dg_keep(c, g, NULL, 0)`.
Several layers may be held at once when the arena has room. Normal decode tokens follow.

- **The call.** Engine thread, between tokens, under the state rules of `ss_dg_set_policy` (the device and the service
  handed the core back). It validates g and the ids (distinct, in range), then refuses with `SS_E_CAPACITY` before any
  action when n + the distinct kept units outside the ids (every layer's keep, g's current one included: it stays
  protected until the call succeeds) + the spare target exceed the persistent slots (residents
  + billed spares; bank slots and the guard slot are never kept), when the free slots, the slots write-backs in flight
  will free and the evictable units cannot take the missing units, or (two-tier) when there is neither a free slot nor
  a free pinned block to begin with. Then it drains every read, copy and write-back (and collects them). Units of g on
  the device stay where they are (a prior prefetch still coming is admitted), the others take free slots, else evict
  device units no layer keeps, in the policy's order: the recency tail under LRU, the boundary's bounded score order
  (score, demand epoch, stamp) under SCORED and SCORED_AUTO. A victim goes down one tier with the boundary's rules: a
  clean copy back to its block; a write-back into a free block (synchronous, before its slot is reused), a block of a
  copy already made from a pinned home once the demand stream is synced, three-tier a pinned unit dropped to its file
  for that block, else the victim itself dropped to its file. The units are copied in from their pinned blocks, or read
  from their files into the staging buffers (up to `dg_disk_queue` reads in flight) and copied from there; new units are
  the most recent; no clean copy is kept. The table is uploaded before the call returns. Placement only: every slot
  receives its unit's file bytes, so outputs are those of any other placement.
- **Failures.** Each transfer is a transaction: a unit's home changes only after its copy and its entry were
  accepted (else its slot goes back to the free list and its mirror entry to -1; its source keeps its home and block),
  a source block is freed and a written-back victim's slot reused only after the stream that reads it was synced, a
  victim leaves the device only after its write-back was accepted. A file read failure (`SS_E_IO`, the device fine)
  re-establishes the cache: the reads in flight are ended, what was enqueued is synced, the table uploaded; every keep
  is unchanged (g's current set was protected, so its units are where they were) and the units moved so far stay. Any
  device failure (a copy, a publication, an event record, a synchronize, the upload) poisons the core and the context
  (`SS_E_BACKEND`); the host records stay one consistent ownership, but what the device holds is unknown.
- **The kept set.** g's keep becomes exactly the ids; `ss_dg_keep` for g (n = 0 releases) or a later make_resident
  replaces it, `ss_dg_reset_open` drops every keep. While held, g is never exported (its rows are ignored): readers
  resolve its kept ids through `ss_dg_slot` / `ss_dg_slot_if_ready`, any other id is absent (the guard slot, and the
  token fails). A kept unit on the device keeps its slot: inside a token entries only change from -1 to a slot, and no
  boundary evicts or reverts a kept unit (a kept unused prefetch is admitted rather than reverted).
- **Unexported layers.** `rows_per_layer[h] == 0` for a gated layer (or a held set) means the step has no
  `ss_dg_enqueue_export` / `predict` / `wait` for h. The service waits for and serves only the exported layers, in
  order; a prediction an export promised for an unexported next layer is ignored; the "device ran ahead" checks and
  the write-back slices look at the next exported layer. The core's token has no demand, use, recency or score history
  for h (its units age as unrouted ones: the token's epoch still advances), and the boundary ranks h from its
  unchanged history, so every score list is the token's and no fallback is counted. h's export flag keeps the sequence
  of the last step that exported it; the service checks a later export against that value and `ss_begin` checks every
  flag against it, so a stray export of an unannounced layer is a desync (the context is poisoned).
- **Counters.** The call's copies count only in `ss_dg_stats.resident_units`, `resident_evicted`, `resident_bytes` and
  `resident_ns` (also in the stats JSON); misses, fetches, prefetches, evictions, write-backs and file reads are
  decode's. Work it completes at its drain (write-backs and reads already in flight) is counted as before.
- **Gates (CPU).** The randomized core runs (every LRU, scored, deferred, window-prefetch and three-tier shape) add
  make_resident episodes, held layers read in every layer of every step, and randomly unexported layers; the device
  model checks every read's bytes and that no copy lands in a slot under a read, the oracles check every boundary's
  victims, held units never move, refusals change nothing. Deterministic tests check a full layer, several layers and
  the capacity refusals, the victims against the LRU order and the scored oracle, the three-tier write-back / drop
  paths and a failed read, clean-copy victims, and that an unexported layer's counters and score history stay put.
  Device failures are injected at every H2D, D2H, entry write, event record, event / stream synchronize and upload of
  the call in turn: each must fire (observed), poison the core and leave consistent ownership. A failed replacement
  (a read failing after evictions) is followed at once by a step reading the old set unexported. The context's
  bookkeeping (`DgSteps`: keeps, held sets, the exported layers, the flag values) runs with the service and the core
  in the GLM sequence on 4 x 288 units (per-layer make_resident, a step exporting nothing, release, decode; two whole
  layers held for 40 decode steps). The same sequence through the real context and `ss_dg_slot` is the GPU test
  `ts-test-dg-cuda --only dg_cuda_make_resident_prefill`.

### 3.10 Device-gated placement rounds and no-admission misses

Placement in the device-gated cache otherwise changes only through demand: a miss is admitted, the boundary evicts
by LRU / SCORED / AUTO. A placement round moves units toward a target computed from their export history instead,
between tokens, within a budget the engine chooses.

- **Counts.** Every gated unit has a decayed count `s` (zero at open). `serve_use` counts, for every exported layer,
  each unique routed id once per step (the ids the SCORED tables see; host only, no device work; an unexported or held
  layer counts nothing). A round first applies `s = decay x s + hits` to every unit and clears the hits.
  `ss_dg_round_seed_counts` sets `s` directly (finite, >= 0; every entry checked before any is set), e.g. from a
  shipped routing profile. `ss_dg_reset_open` zeroes counts and hits.
- **Target.** The rank by `s` (ties by key): the top `persistent slots - spare target - kept units` on the device
  (kept = every layer's `ss_dg_keep` / make_resident keys, distinct; prefetches still coming also hold slots), the
  next in the pinned tier, the rest in their files.
- **The call.** `ss_dg_round(c, params, result)`: engine thread, between tokens, the state rules of
  `ss_dg_make_resident`. Params: `budget_bytes` (file reads + H2D + write-backs; 0 = no limit), `max_units`
  (promotions, default 96; 0 = only the counts decay), `hysteresis` (>= 1, default 1.25), `decay` (0..1, default
  0.85), `flags` (0). It drains every read, copy and write-back first (a device failure there poisons), then:
  - (a) pinned -> device, hottest first: into a free slot while the device holds less than its share and more than
    the spare target is free; else in place of the coldest movable device unit (ties: least recent) when the
    candidate's `s > hysteresis x` its `s`. The demoted unit goes back to its block when it holds a clean copy (no
    write); else it is written back into a free pinned block on the write-back stream (`dg_wb_stream = 1`) or the
    demand stream, synchronously, its bytes charged; three-tier with no free block, the coldest pinned unit is
    dropped to its file to free one. The promoted unit's block is freed once its copy was synced (no clean copy kept).
  - (b) three-tier: file -> pinned, hottest first, in place of the coldest pinned unit (ties: least recently homed)
    when `s > hysteresis x` its `s`: that unit is dropped to its file (no copy), and the candidate is read straight
    into its block by the reader pool, up to `dg_disk_queue` reads in flight.
  - A unit with `s = 0` is never promoted. Kept or held units, units with a write-back in flight and coming
    prefetches never move. A move whose bytes do not fit the remaining budget, or `max_units` promotions, ends its
    phase. Decay alone never churns: equal decay keeps every ratio.
- **Transactions and failures.** Each move is synchronous: a unit's home changes only after its copy and its entry
  were accepted and the stream that copied it was synced; a demoted unit leaves its slot only after its write-back was
  synced; a source block or slot is freed only then. A file read failure (`SS_E_IO`) ends the reads in flight (their
  blocks freed, their units still in their files), uploads the table and returns with the moves made so far; not
  poisoned. Any device failure (a copy, a write-back, an entry, an event record, a synchronize, the upload) poisons
  the core and the context, the host records one consistent ownership. The table is uploaded (and synced) before the
  call returns.
- **Cancellation.** `ss_dg_round_cancel` (any thread: one atomic store) makes the round in progress stop at its next
  unit boundary. A copy made but not committed when the flag is seen (a promotion's H2D synced, a file read ended) is
  dropped: its slot or block is freed, its entry goes back to -1, its source keeps its home. The round returns
  `SS_OK` with `cancelled = 1`. A cancellation with no round running has no effect (each round clears the flag when
  it starts).
- **Result and counters.** `promoted_to_device`, `promoted_to_pinned`, `demoted_from_device`, `dropped_to_file`,
  `bytes_read`, `bytes_h2d`, `bytes_written_back`, `ns`, `cancelled` (also filled after a failure), and the lifetime
  counters `ss_dg_stats.round_count / round_units / round_bytes / round_ns / round_cancelled` (also in the stats JSON).
  A round's copies, reads and write-backs count nowhere else.
- **No-admission misses** (`ss_options.dg_no_admit = 1`, default 0). A demand miss takes a bank slot (pinned block ->
  bank, or file -> staging buffer -> bank, the existing bank fallback) and is released at the boundary: its home
  (block or file) is unchanged and no resident is evicted. No prefetch (prediction, window prior, l+1) is issued and
  the boundary selects no victims, so placement changes only through rounds, `ss_dg_make_resident` and `ss_seed`. A
  bank holds `dg_max_ids` slots and a layer has at most that many distinct misses, so both banks and the guard (the
  existing ring rule) always suffice; `ss_open` names that count when the ring is short (`SS_E_CAPACITY`), and the
  core refuses a configuration with fewer bank slots than `max_ids`.
- **Gates (CPU).** Seeded rankings check the exact moves, hysteresis (9 against 10 does not move), decay without
  exports (no churn), `max_units`, `budget_bytes`, kept units, a device below its share, clean-copy demotions, and the
  three-tier write-back / drop / read paths with synchronous and overlapped reads. A randomized oracle runs a cache
  with rounds (random counts, limits, keeps; deferred and immediate write-backs on both streams; clean copies;
  three-tier) against one without: every token reads the same bytes. Device failures are injected at every H2D, D2H,
  entry, record, event / stream synchronize and upload of a round in turn (each observed, poisons, consistent
  ownership); file read failures at the first and later reads (`SS_E_IO`, consistent, the next round completes).
  Cancellation from another thread at a unit boundary, during a copy and during the file reads. `dg_no_admit`: misses
  served with the exact bytes through bank slots, homes unchanged after every boundary, the refusal, and with rounds
  the hot set converging onto the device (no misses on hot-only tokens).

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

- `verify_hash = 1` checks the xxh64 of every unit's bytes on the host side before its job is reported done: at the pinned fill,
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
| the unit table is rewritten while in use | There are two tables per layer, alternating per acquire. `ss_release` records the table's event on the engine stream, and the next rewrite of that version host-waits it. This covers the CUDA H2D from the pinned mirror and the unified backends, whose kernels read the mapped mirror itself. |
| jobs still in flight across forwards | `ss_end` waits until every job of the forward has been *issued* (discrete) or has *landed* (unified), and cancels speculative jobs that never started. On a discrete GPU an issued copy may still be in flight, so for every layer that got persistent copies `ss_end` records a per-layer `ready` event on the copy stream. The next `ss_acquire` of that layer makes the engine stream wait for it once, before any hit is read. |
| a failed read leaves a unit marked resident | `ss_end` drops every persistent unit whose job did not finish; `ss_seed` drops the seeds that failed and returns the first error. A failed unit is read again at its next use. |

### 4.4 Several drives: multi-location files

- A model file (the **primary**) may have **stripe copies** of some of its chunks in other directories, normally on
  other NVMe drives. `ss_options.weight_dirs` (recipe `weight_dirs`, engines `--weight-dirs`) lists them, `;`
  separated; the primary's own directory and duplicates are ignored, so with no other directory nothing changes.
- A stripe directory holds `<primary basename>.stripe.json` (`tierstream/stripe-v1`) and the data file it names.
  The manifest gives the chunk size (a power of two, 16 MiB by default) on a grid over the primary's offsets, the
  placed chunks as runs `[first chunk, count, first slot]`, the xxh64 of every placed chunk's bytes, the primary's
  SHA-256, and a digest of its own canonical text. Placed chunks are stored whole and back to back (the last one
  zero-padded), so every read of the data file is aligned. Validation is strict (sorted non-overlapping runs, dense
  slots, sizes, names, limits checked before allocation) and any failure disables that location only.
- `ts-stripe make PRIMARY RANGES.json OUT_DIR` builds a stripe: one sequential unbuffered pass over the primary
  (SHA-256 + copy), unbuffered writes, a full read-back check, an immutable versioned data file, then an atomic
  manifest switch. A ranges file names byte ranges with a policy: `all` (every chunk on the stripe) or `interleave`
  (odd chunks on the stripe, so that whole-layer streams and random record reads split about evenly over two
  drives; a single small unit usually lies in one chunk). `verify`, `gc` and `show` maintain stripes.
- **Session.** The registry lookup key is (canonical primary, directories, direct flag). Reuse requires matching
  `drive_readers` (1..32, default 4), unchanged file identities and manifests, matching caller digest when supplied,
  and verification at least as strict as requested. A directory with no manifest stays reusable while it stays absent.
  Reader pools are shared process-wide by (volume identity, reader count), including across shard files on that volume.
  Each call waits for its own jobs on success and error; a shared pool never makes another session's jobs its responsibility.
  Reads and queries may run concurrently on a live handle with non-overlapping destination buffers. Close must not race
  any call on that handle; closing a separate handle to the session is safe.
- **Writer contract.** Windows opens every serving file with `FILE_SHARE_READ` only, excluding writers. Linux has no
  share-read-only exclusion: stripes are published mode 0444, but owners and privileged processes can still write.
  Identity, including Linux ctime, is checked at reuse and after verification. Keep files immutable while sessions live;
  use `weight_verify=always` on Linux where stripe directories are not exclusively owned. Partial reads still rely on
  open-time verification, even with `always`. Stripe directories must be as trusted as the primary: xxh64 detects
  accidental corruption, not deliberate tampering.
- **Bound opens (resolve, then bind).** Every file a session or a descriptor source reads (the primary, each stripe
  directory's manifest and data file, each `desc.sources` file) is first resolved once to its real final path
  (Windows: opened normally and `GetFinalPathNameByHandleW`, which also expands 8.3 short names; Linux: `realpath`),
  so symlinked or junctioned model and stripe directories keep working: a link that exists then is followed once.
  That real path is then opened through held handles of its directory and of every ancestor from the volume root,
  and a directory or file found to be a link during that walk (one swapped in after the resolve) is refused (a stripe
  location is disabled, a primary or source fails to open). On Windows a link is a name-surrogate reparse point
  (symlink, junction, mount point); other reparse points (OneDrive and other cloud files, dedup, compression) are the
  entry's own data and are accepted. Windows holds each directory with `FILE_FLAG_OPEN_REPARSE_POINT` and without
  `FILE_SHARE_DELETE`: while a session (or a context's source) is open, the model directory and every ancestor up to
  the volume root cannot be renamed or deleted, and opening fails while another process holds one of them with
  DELETE access (e.g. a rename or delete in progress). It also checks that the file's final path is its held
  directory's plus its name. Linux walks `openat(O_PATH | O_DIRECTORY | O_NOFOLLOW)` from `/` and opens the file
  `O_NOFOLLOW` relative to the held directory (renames cannot be denied there). The manifest stays open for the
  session's lifetime and is read through its handle. Resolving follows whatever the path names at open time, so a caller binds to what it authenticated by
  identity: `ss_mfile_location_info` reports, per location, the manifest's digest field and the identities (volume +
  file index, or dev + inode, and size) of the held manifest, data file and primary; `ss_options.source_identities`
  makes `ss_open` require an identity per descriptor source (`SS_E_DESC` naming the source otherwise). The loaders'
  per-worker handles are checked against the identity of the context's open source.
- **Reads.** Offset, length and buffer are 4 KiB aligned (a read may run past EOF to the aligned end; that tail is
  zero-filled on every route). A read is split at chunk boundaries into one piece per chunk. Windows issues concurrent
  4 MiB parts within each piece; Linux reads those parts sequentially. With a stripe location, reads of 8 MiB and up
  always go to the shared volume pools; a smaller single-piece read runs on the caller's thread and other multi-piece
  reads go to the pools, even if all pieces use one location.
- **Which reads use a stripe.** The primary holds every byte, so a stripe pays only when two drives work on the same
  request. Requests of at least 8 MiB (whole-layer expert streams, startup loads) are split; smaller requests (one
  expert's parts, a decode step's 24-48 Engram pages) and scattered batches under 64 reads stay on the primary,
  exactly as with one drive: a split small request waits for the slower drive. Measured: splitting a
  24-row Engram job made it slower on a quiet primary (0.29 -> 0.47 ms), so it is not done.
- **Adaptive shares.** In a split request each chunk with a stripe copy goes to whichever of its two homes has the
  least of (bytes this request already gave it + the chunk) / its share. Shares follow each location's measured
  throughput (its bytes of a request over the time until its last piece finished, EWMA). Every 16th large request
  reads the primary alone, so a current single-drive baseline is always known; while split requests measure slower
  than that baseline, splitting is off for the next 64 large requests. This bounds a regression rather than excluding
  it: no controller can know the single-drive time of the request it splits.
- **Verification.** Every byte served from a stripe belongs to a chunk whose xxh64 matched the manifest, checked
  through the serving handle at open (`weight_verify = always`) or at the first open of that data-file version
  (`cached`: the passing-chunk bitmap is kept by file identity; the identity is volume, file index, size, last-write
  and creation time - not the NTFS change time, which metadata-only events move); with `always` also again on every read covering a
  whole chunk (that hash sits on the read path: measured, it cost a large prompt-prefill weight stream about a quarter of its
  throughput, so the default `cached` relies on the open-time check and the platform writer contract).
- **Admission cost.** Without a caller-supplied digest or a primary identity-cache hit, first open hashes the entire
  striped primary and verifies the entire stripe; hundreds of GiB can take minutes. `primary_sha256` is trusted without
  recomputing it and must describe the primary's current bytes. Cached verification results avoid later stripe scans.
  The C API always validates the reader count and verification mode, even without stripes, and refuses test-only OFF.
- **Fallback.** A read error, short read or hash mismatch on a stripe re-reads that piece from the primary in the
  same job and demotes the chunk to the primary for good; three errors disable the location. A missing drive,
  manifest or data file, or a manifest bound to other primary bytes, disables that location at open. Engines see
  the primary's bytes in every case.
- Counters (`ss_mfile_get_counters`): bytes, reads and busy time per location, fallback bytes, demoted chunks,
  verification failures and time, disabled and unbuffered locations, the effective verification mode.

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
3. **Non-resident units** (host bytes are counted as the pinned tier allocates them: 4 KiB strides packed into
   chunks of at most 1 GiB, exactly the layout the context allocates):
   - `pinned` → pinned; refused if over the host budget (vramplan's refusal).
   - `disk` → disk.
   - `pair` → pair.
   - `pinned|disk` → pinned while the host budget lasts (after `host_fixed`), best benefit per byte first, else disk.
   - With `pinned_backing`, what is left gives residents pinned copies (`B`), the lowest benefit per byte first.
4. **Final check.** `peak = fixed + dynamic + rings(streamed) + resident ≤ cap − margin`, and pinned
   (`host_fixed` + pinned copies) ≤ the host budget. Both are refused, as in vramplan.
   - Dynamic slots are billed exactly as allocated, `min(dynamic, n_units − residents)` per layer. Room is first
     reserved with the upper bound, so the real peak is never above the plan's.
   - Non-streamable units are billed at the pool stride they occupy.

**Decode budgets set `ring_slots`.** The default (2 × the most units of a layer of that size) suits layer-major
prefill. For DS it means 768 expert slots, about 7.6 GB, far more than a decode window needs (top-6 × verify rows ×
2 layers in flight).

**Parity.** `tools/vramplan_parity.py` builds the vramplan `Model` from a descriptor (the §2 mapping) and compares
every unit's tier and the totals on DS- and MiniMax-shaped descriptors across caps and host budgets: 13 of 13 equal.
Intentional differences are per-size rings, dynamic slots, `B` tiers and 4 KiB host strides.

`ss_plan_from_tiers` / `ss_plan_from_json` take tiers decided elsewhere and run the same totals and checks. This is
how an engine runs exactly the placement hyperloom planned (plan-v1 JSON: one `RPDX` letter per unit, per layer).
Pair units (`X`) are never fetched, so they take no stream ring: `rings(streamed)` counts only pinned and disk units.
This matters when tiers mix `X` with resident units on a descriptor whose `tiers` is not `pair` (a heterogeneous
split, where the device keeps hot units and the peer computes the rest).

**Recipes.** A recipe ([recipes.md](recipes.md)) compiles to these same inputs: an `ss_budget`, `ss_options` and
tiers that go through `ss_plan_from_tiers`. A recipe placement with one unit group `{rank = "frequency", resident =
"fill", pinned = "fill", rest = "disk"}` gives exactly `ss_plan_make`'s plan.

### 5.1 Whole-device budgets

An engine's budget is the whole device, as if the card had N GiB (`--vram-total-gib N`), and the whole host, pinned
included (`--ram-total-gib M`). `ss_budget_plan(in, out)` (sized structs, pure integer arithmetic on bytes,
saturating; no device call) is the one implementation every engine uses for it:

- **Inputs.** The caps (0 = uncapped); the measured device context; named fixed components for the device (trunk /
  dense weights, KV for the requested context, workspaces, drafter), for pinned host buffers and for pageable host
  memory; the load staging; the expert geometry (keys, device slot bytes, host stride, arena spares, host spare
  blocks, file-tier staging, inclusive request, whether the file tier is allowed); and the uncapped knobs of the
  engines' three-tier placement (explicit arena and RAM expert budgets, host memory available now, its margin, the
  pinned ceiling).
- **Device.** `arena = ⌊(cap − context − Σ device components − VRAM margin) / slot⌋ − spares`, never more than an
  explicit arena request (uncapped: the request, or every expert). The margins (`vram_margin_bytes`,
  `ram_margin_bytes`, default 0) keep room under each cap for what is allocated after the plan (graph capture, warmup,
  per-request buffers, allocator slack); the minimum caps include them.
- **Host.** The pinned room is the smallest of (available − margin − load staging), (pinned ceiling − engine pinned −
  load staging) and, with a RAM cap, (cap − pinned − pageable components − load staging − RAM margin). Experts that
  do not fit their pinned share go to the file tier (refused without it). Inclusive clean copies only take RAM left
  over.
- **Refusals.** A cap that cannot hold `context + device components + VRAM margin + (spares + min_resident) slots`
  (device) or `host components + RAM margin + spare blocks + staging + one block` (host; every non-resident expert
  without a file tier) is refused with `SS_E_CAPACITY` and a message naming each component and the smallest cap that
  would do (`min_vram_total_bytes`, `min_ram_total_bytes`, the flag value rounded up to whole GiB).
- **Uncapped = today.** With both caps 0 the outputs (counts per tier, inclusive, spare blocks, staging, room, the
  host need and the pinned peak) are those of DeepSeek's `rtx::tier_plan`; `tests/test_budget.cpp` carries that
  function unchanged as an oracle and compares its cases and a 16,200-case grid.
- **Receipt.** `ss_budget_plan_json(in, out, buf, len)` writes the plan as JSON members (no braces): caps, margins,
  the fixed components by name, the expert geometry and requests, the placement, planned bytes, `device_room_bytes`,
  `host_cap_units` (the pinned ceiling used), `device_capped` / `host_capped` (the cap, not the request, sized that
  tier), and the minimum caps in bytes, in GiB rounded up to 0.01 and as the whole-GiB flag value.

The ledger (`ss_budget_ledger`) tracks what is really allocated against the same caps: `add` / `sub` per kind
(DEVICE, PINNED, PAGEABLE; PINNED and PAGEABLE share the RAM cap) and component name, thread-safe, an add over the
cap refused with `SS_E_CAPACITY` naming the component; the stats give current and peak per kind, the 16 largest
components at that kind's peak, and the peak of externally measured device use (`observe_device`: cudaMemGetInfo or
NVML samples), so the receipt shows the ledger's peak beside the measured one. `observe(stage, device, host)` records one measured
sample per side (the engine's cudaMemGetInfo / NVML device use, its process commit with pinned memory; either side
may be absent): the last value, the peak with the stage it happened at, and the samples above each cap;
`within_caps` is 1 when every measured peak stayed at or below its cap (the receipt's "peak <= cap"). Passed as `ss_options.budget`, the
context charges its own allocations before making them: in device-gated mode the arena, slot table, controls, mapped
area, pinned blocks, disk staging, staging, bounce (pageable, one per file reader), table staging, diagnostics and
(until the open ends) the load ring; in host mode every allocation of its backend. A cap overrun fails `ss_open` naming the allocation; closing
releases the charges.

### 5.2 Device profiles

The caps of §5.1 can come from the machine instead of flags. A device profile (`strix/device-profile-v1`, canonical
JSON) describes ONE machine as the engine sees it: host RAM (total, available now, unified with the GPU or not), CPU
cores, NVMe drives, every GPU (api `hip` / `cuda`, name, arch such as `gfx1151` / `sm_120`, compute units, wave, LDS,
memory / GTT / visible bytes, memory and host-to-device bandwidth, PCIe link), the inter-host links (kind, count,
measured Gbps), the margins (host floor, device reserve), class limits, evidence ids, the merged override paths and
notices.

- **Detection** (`ss_devprof_detect`, < 1 s, never fails): Linux `/proc/meminfo`, `/proc/cpuinfo`, the KFD topology
  (arch from `gfx_target_version`, CUs from `simd_count / simd_per_cu`, wave, LDS), the amdgpu DRM `mem_info_*`
  totals (the APU's memory = GTT + carve-out), Thunderbolt / USB4 XDomain peers and `/sys/class/nvme`; then the HIP
  runtime (flat `hipDeviceGetAttribute` / `hipMemGetInfo`, loaded at run time) and on NVIDIA the CUDA driver API +
  NVML (loaded at run time; no context, no allocation, no probe: name, UUID, SMs, compute capability, total memory,
  PCIe gen / width, driver). The H2D rate is the engine's measured one from its per-GPU cache file
  (`strix-gpu-<uuid>.json`); NVMe rates come from an override (or a recipe's evidence) until measured. tierstream builds
  and runs without either runtime.
- **Class** (`profiles/<class>.json`, the compiled table written out; CTest `devprof` compares them byte for byte):
  picked by api + arch + name substring + memory range (`strix-halo-395`: gfx1151 with 96-128 GiB of GTT +
  carve-out, the size it was qualified on; any other gfx1151 is `generic-hip`); gives the margins (today: host floor 8 GiB = the recipes'
  `host_free_ram_floor_gib`, device reserve 2 GiB = `ss_budget.margin_gib`), a measured memory bandwidth with its
  evidence, `unified`, and **limits** engines read instead of literals (`ss_devprof_limit`): `strix-halo-395`
  max_piece_rows 16384, uma_ceiling_bytes 120 GiB, persist_blocks 40; `rtx-pro-6000-blackwell` long_vram_cap 38 GiB,
  vram_reserve 1 GiB, replica host margin 6 GiB, max_pinned 68 GiB, service commit floor 20 GiB. An unknown machine
  (a new gfx, another card) gets `generic-hip` / `generic-cuda` / `generic-cpu`: detected values and today's margins,
  no limits (the engine keeps its own rule) and a notice - never another arch's or size's qualified numbers. What a
  class fills is marked `class:<name>` in `evidence`; an override that changes `class` or device 0's api / arch / name
  / memory takes all of it back (except values the override sets itself) and picks the class again.
- **Override**: `--device-profile FILE` (`ss_devprof_detect_opts.override_path`) or `STRIX_DEVICE_PROFILE`, a full or
  partial profile merged over the detected one (objects by key, `devices` / `links` / `host.nvme` by index, `limits`
  and `evidence` by key; the text's own `overrides` / `notices` / `sources` are ignored, so a saved `ts-devprof` output
  works as a full override; atomic; every merged path recorded as `FILE:path` in `overrides`). Detection drops GPUs
  the HIP runtime hides (visibility variables) and joins runtime and sysfs devices by PCI address, never by order.
  `STRIX_DEVICE_PROFILE_ROOT` (or `opts.root`) reads `/proc` and `/sys` under another root (tests, replay).
- **Receipts**: `ss_devprof_hash` (xxh64 of the canonical text) names the profile; `ss_devprof_probe_report` gives
  the recipes' probe report (§8.2 of recipes.md, plus `class` and `arch`).

Planner hooks (all pure arithmetic):

- `ss_devprof_budget(p, device, in)` writes only the caps still at their defaults (explicit flags win): discrete
  `vram_total_bytes = memory - device_reserve`, `ram_total_bytes = available - host_floor`; unified (Strix Halo and
  successors) one pool `min(available - host_floor, memory - device_reserve)` as `ram_total_bytes` with
  `in.unified = 1`. A profile whose margins eat a known size is refused (`SS_E_CAPACITY`), never turned into 0
  (= uncapped); an unknown size stays uncapped. A caller whose `ss_budget_in` predates the `unified` field gets
  `SS_E_INVALID` for a unified profile (two independent caps of the pool could each take all of it).
- `ss_budget_in.unified = 1` (`SS_BUDGET_UNIFIED`): one joint placement - the most residents `r` (up to the arena
  request, at least `min_resident`) such that the device part (context, components, VRAM margin, `r` + spare slots) and
  the host part (components, load staging, RAM margin and the smallest host placement for the `keys - r`
  non-residents: spare blocks, plus the file tier's staging and one block, or every non-resident without a file tier)
  fit the pool together (closed form: the sum is linear in `r` below `keys`). The pinned tier and the clean copies then
  get what the device leaves; the final sum is checked against the pool. Both minimum caps are the smallest such sum
  (a plan at exactly that pool succeeds). With no cap, or `unified = 0`, plans and their JSON are unchanged.
- `ss_devprof_topology(profiles, n, in, out)`: single box when this box's budget holds the fixed parts + the smallest
  expert tier at the context, else the fewest boxes whose budgets do, provided a link with a measured rate exists
  (`gbps_measured > 0`: discovering USB4 peers does not qualify a pair - the engine's link measurement or an override
  supplies it); the layers split by exact byte budget share (equal boxes: `ceil(layers / 2)` on box 0);
  `pinned_world` keeps a qualified recipe's choice.
- `ss_devprof_piece_rows(p, in, &rows)`: the prefill piece from the budget left after the plan, at most the largest
  multiple of `row_align` within the caller's bounds (on the class a rule was tuned on its bound binds, so its output
  is today's); an unknown device size gives that maximum, a known size the margins exhaust is refused.

`ts-devprof` prints the effective profile (`--budget`: the caps it gives device 0; `--probe-report HOST`;
`--expect-class NAME` for node checks).

## 6. Engine contract

### 6.1 Calls

| call | when |
|---|---|
| `ss_desc_from_model(manifest)` / `ss_desc_from_json` | load |
| `ss_plan_make` or `ss_plan_from_json` | load (or once per request class: P bucket, cap) |
| `ss_open(desc, budget, plan, options)` | load: allocates the pool, rings, pinned tier and staging; fills residents and the pinned tier |
| `ss_begin(rows_per_layer)` | per forward. `rows_per_layer[L] == 0` marks a layer the forward skips (DS cone; device-gated: a gated layer not exported, §3.9). |
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

**One engine stream per forward (enforced).** Every stream argument of one forward must be the same stream:
`ss_layer_routes_dev`, `ss_acquire*`, `ss_group_wait` and `ss_release`. Otherwise the call returns `SS_E_INVALID`. An
engine that runs a layer on several streams joins them into that stream (event + wait) before `ss_release`. A new
forward may use another stream.

**Lifetimes.** An `ss_layer_view` (tables and groups) is valid from `ss_acquire*` of L until `ss_release` of L.
`table_dev` changes between forwards.

**Graphs.** The `ss_*` calls are host calls that issue copies and record events, so they cannot be captured into a
CUDA/HIP graph or run from a graph host node. An engine captures only the work between `ss_acquire` and
`ss_release`, and passes the table through a pointer the graph reads at run time.

**Fixed placement.** A context's placement (plan, pool, rings, pinned tier) is fixed for its lifetime. Another P
bucket or cap is another context; `ss_open` refuses a plan made with a different budget. Resizing in place is not in
v1.

**Versioned structs.** `ss_budget` and `ss_options` start with `struct_size`. `*_defaults` sets it, and the library
reads only that many bytes and defaults the rest. Fields are appended only.

### 6.2 What the engine must guarantee (exact)

1. **Unit-table addressing.** Kernels read a unit only through `table_dev[u]`, never as base + u × stride.
2. **Order and slot invariance.** A unit's output bits depend neither on its slot, nor on which units share its
   launch, nor on the launch split or timing. The combine is a fixed-order step on the original ids. The gate is per
   engine: state bytes for every residency mix (all resident / interleaved / all streamed / routed) are equal to the
   all-resident run.
3. **Byte identity.** Streamed bytes equal resident bytes. The descriptor carries xxh64 per unit, checked at load and
   in test builds after every copy.
4. **The plan may depend on P and the budget; the bits may not.**
5. **Weight slots are read-only.** Kernels never write the bytes a table entry points to. The exclusive tier mode
   (§3.5) writes a slot back into a pinned block as the unit's only copy.

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

Exclusive tier mode, since open (`ss_stats_exclusive`, and `"exclusive"` in the JSON): `writebacks`,
`writeback_bytes`, `deferred` (misses left in the ring for lack of a free block), `promoted` (blocks freed by
promotions), `free_blocks`, `min_free_blocks` (at forward starts). Too many deferred misses means too few spares.

`ss_stats_to_json` gives `strix-stream/stats-v1`. The hook fires at every `ss_end` with the forward's numbers, which is
where a TTFT receipt takes its bytes and waits.

### 7.1 Snapshot (any thread)

The stats calls above belong to the engine thread. A status endpoint on another thread reads `ss_stats_snapshot`
(the totals as `ss_layer_stats` plus `ss_snapshot_info`: publication sequence, forwards, read-ahead and exclusive
counters, flags saying which apply) or `ss_stats_snapshot_json` (the same `strix-stream/stats-v1` document
`ss_stats_to_json` gave at that moment, byte for byte, per-layer counters included).

- Armed on demand: until the first snapshot call, the engine thread does one relaxed load of a false flag per
  `ss_end` and per `ss_stats_reset`, nothing else. The first call arms and returns `SS_E_INVALID` while nothing is
  published; from then on the engine thread publishes at every `ss_end` that folds a forward (also one that reports a
  read failure) and every `ss_stats_reset` between forwards.
- A publication is a copy of the per-layer counters, the read-ahead counters, the exclusive counters and the forward
  count under a small mutex, into storage sized at open: no allocation, no formatting, O(layers) per forward.
  Readers copy it under the same mutex (`ss_stats_snapshot` also sums the per-layer counters there, O(layers)) and
  format outside it; backend, sources, read-ahead buffers and block count are fixed at open.
- Every publication is the live state at a forward boundary. Changes made between boundaries (inside a forward,
  `ss_seed` promotions, read-ahead requested between forwards, a reset during a forward) appear at the next one; an
  `ss_end` that fails before folding, and anything after the context is poisoned, publish nothing, so the last
  publication stays readable. During the stats hook the previous publication is still shown.
- Readers may run concurrently with every call except `ss_close`, which they must not overlap or follow.
- `ss_snapshot_info` is versioned by size like `ss_exclusive_stats` (inline `ss_stats_snapshot` passes `sizeof`);
  fields are only appended.

## 8. Backends

| backend | status | notes |
|---|---|---|
| CPU | done, tested | host memory, unified; the reference for the in-place path |
| CUDA | built and GPU-tested, `ts-test-cuda` | VRAM pool and rings, `cudaHostAlloc` tier and staging, one non-blocking copy stream for H2D: one DMA engine reaches 31-32.5 GB/s and a second H2D stream adds nothing (spec-rtx-prefill §2.1); exclusive mode adds a write-back (D2H) stream |
| HIP | built and GPU-tested on Strix Halo (gfx1151), `-DTS_HIP=ON` | unified: slots are mapped host memory, misses are read in place from local NVMe; exclusive mode is refused |

CUDA device-gated decode is merged and opt-in; its GPU tests and the DeepSeek full-model bit-for-bit and
speed-parity gates pass. It uses a global slot arena and boundary eviction to a spare-slot target, separate from
the host-controlled per-layer pools described in §3. See [the engine contract in the README](../README.md#device-gated-decode-graph-captured-moe-decode).
The older 31-32.5 GB/s transfer measurements above are not a fixed platform limit; see
[the current PCIe conditions](../README.md#when-it-pays-and-when-it-does-not).

Deferred, bounded write-backs (`wbsched`) are merged and on by default through `ss_options_defaults`: pending victims
carry across token boundaries and background submissions are bounded to limit copy-engine queueing.
`ss_dg_set_sched` switches deferral and ready flushing between tokens, including per-request choices by the engine;
outputs are bitwise identical. See [results and conditions](../README.md#status).

The pipelined startup loader is merged and on by default; `DG_LOAD_SERIAL=1` restores serial loading.
In pipeline mode the pinned blocks are allocated on a helper thread, chunk by chunk, while the loader already reads
the device-resident units; a host unit is read once its chunk exists, and the helper is joined right after the loader
(on by default; `DG_LOAD_OVERLAP_ALLOC=0` restores the serial allocation order). The `dg_load` lines report
`pinned_alloc_s` (total allocation time), `pinned_alloc_wait_s` (time readers waited for blocks) and `resident_done_s`.
`ss_dg_reset_open` restores the resolved open-time cache state. `ss_dg_residency` snapshots device-resident units in
recency order for a later load's `dg_seed`. Measured startup, reset and seed results are in the
[README](../README.md#status). Window prefetch (`winpf`) is merged, exact and off by default; evaluation is closed
with no repeatable gain in matched warm comparisons.

## 9. Versioning

- `SS_ABI_VERSION` = 1. Structs are passed by pointer and filled with `*_defaults`.
- A field added at the end is a minor change. Any other change bumps the ABI.
- Calls are only ever added: `ss_stats_snapshot_sized` / `ss_stats_snapshot_json` (`SS_HAVE_STATS_SNAPSHOT`) came
  after the first v1 release; a caller that uses them needs a library that exports them.
- The descriptor schema is versioned separately (`desc-v1`). Plan and stats JSON carry their own schema ids.

## 10. Planned (not in v1)

- **Part-granular tickets.** gate/up land before down, so their GEMV starts about a third earlier.
- **Coalesced reads.** Read adjacent units of the same part in one request (GGUF stacked tensors: consecutive ids
  are contiguous).
- **Mapped-flag gates** (`cuStreamWaitValue32` / `hipStreamWaitValue64`) instead of host waits for file jobs.
- **Per-layer row splits** from miss curves (dynamic slots per layer instead of a uniform count).
- **A rate-tally file across runs.** Measured to generalize poorly in the study, so only as an opt-in.
- **Adaptive memory planning** (`adaptive`, in development): usage tables rank VRAM / pinned / NVMe homes after
  reserving context state. The measured replay and its limits are in the [README](../README.md#adaptive-memory-and-nvme-measurements-in-development).
- **Persistent loaded worker and graph pre-capture/warm-up** are in development in the engine (replica).
  The cache reset-to-open, runtime scheduling and residency APIs are already merged in tierstream.
