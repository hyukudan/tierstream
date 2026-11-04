# tierstream

[English](README.md) · [Español](README.es.md)

*Formerly strix-stream.* All-in-one weight streaming across memory tiers, for MoE and dense models, on CUDA and HIP.

A weight cache for inference engines whose model does not fit in GPU memory. It keeps the weights that matter now
on the GPU, keeps more in page-locked host RAM, and reads the rest from NVMe. It moves weight *units* (an expert, or
a whole layer's blob) between these tiers while the engine runs. Engines read every unit through an address table,
so where a unit lives never changes a single output bit.

- C ABI: `include/tierstream/ts.h` (prefix `ts_`). The strix-stream names (`include/strix_stream/ss.h`, `ss_*`,
  the `strix_stream` CMake target and library file, `ss-recipe` / `ss-plan`, `SS_*` options and `SS_TEST_DIR`)
  keep working unchanged for one release (see Names below).
- Design and engine contract: [docs/design.md](docs/design.md). Recipes: [docs/recipes.md](docs/recipes.md).
- Backends: CPU (tests), CUDA (RTX PRO 6000, 4090), HIP for Strix Halo (unified memory; see Status).

## How it works

### Three tiers

```
     engine kernels ── read unit u of layer L at table[L][u] ──┐
                                                               v
  ┌─ 1. GPU memory (VRAM) ──────────────────────────────────────────────────────────┐
  │  resident slots (the plan's units) │ dynamic slots (adaptive) │ ring (temporary)│
  └─────────────────────────────────────────────────────────────────────────────────┘
        ^ promote: copy RAM -> VRAM          │ evict
        │                                    v   inclusive: drop it (RAM or the file still has it)
        │                                        exclusive: write it back VRAM -> RAM into a free block
  ┌─ 2. pinned host RAM ────────────────────────────────────────────────────────────┐
  │  unit blocks │ staging buffers │ read-ahead buffers (optional)                  │
  └─────────────────────────────────────────────────────────────────────────────────┘
        ^ on a miss: read the file into a staging buffer, then copy it to VRAM
        │ ahead of time (opt.): read-ahead fills its buffers before the GPU slot is free
  ┌─ 3. model files on NVMe ────────────────────────────────────────────────────────┐
  │  GGUF / safetensors / packed records; unbuffered (direct I/O) by default,       │
  │  buffered when the file system refuses it                                       │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

- **Ring.** A small set of temporary GPU slots, used by misses that do not get a persistent slot and by prefetches.
- **Inclusive vs exclusive RAM.** In inclusive mode (the default), a unit can be in RAM and in VRAM at once. In
  exclusive mode, RAM holds the units that are not in a persistent GPU slot; during a forward a unit can briefly be in
  both (below).
- **Unified memory (Strix Halo).** Tiers 1 and 2 are the same memory, and there is no separate pinned tier. A
  resident unit is mapped host memory, and a miss is read from NVMe into its slot. Buffered reads land directly in
  the slot. With direct I/O, each part's file offset, length and destination address must be aligned; otherwise the
  read goes through a staging buffer and a memcpy.

A **descriptor** lists the model's units: their files, offsets, sizes and parts (for example an expert's gate, up and
down matrices). A **plan** assigns each unit a starting tier from a memory budget:
- First it reserves the descriptor's fixed bytes, the engine's workspace, the margin, the dynamic slots and the ring.
- Then it fills the rest of VRAM:
  - with `order = ranked`, the units with the highest benefit per byte (by default every unit is worth the same; a
    measured routing frequency makes it smarter);
  - with `order = prefix`, the longest model-order prefix of units that fits (units are walked layer by layer, so
    with one blob per layer this is a prefix of whole layers).
- Then pinned RAM; the rest stays on disk.

`ts-plan` prints a plan; `ss_plan_from_tiers` takes your own.

### One forward, from the engine's side

`ss_begin` opens a forward and `ss_end` closes it. For each layer in between:
1. The engine says which units it needs: `ss_layer_routes` (the router's expert ids), or `ss_layer_all` for a dense
   layer.
2. `ss_acquire` gives it the layer's table once those units are on the GPU.
3. `ss_release` says the kernels are done with them.

Units already resident need no weight transfer. The others are *misses*, copied from pinned RAM or read from disk
into a free slot. `ss_acquire_groups` lets the engine start on the resident units while the misses are still coming
("hits first").

### Device-gated decode (graph-captured MoE decode)

In normal mode, the host handles each layer's routes and supplies its address table. Device-gated decode
(`forward_mode = SS_FORWARD_DEVICE_GATED`) lets the engine capture a whole token in a CUDA graph: a recorded sequence
of GPU operations. The GPU exports routes and waits for missing weights. A host service thread still processes every
layer, but a layer whose weights are already available needs no host reply.

```text
 engine's CUDA graph                         tierstream service thread (host)
 -------------------                         --------------------------------
 begin
 router for layer g
 export(g, ids) ---- mapped route record ----> validate ids; serve any misses
       |                                      demand copies and slot-table writes
       |                                      run on the demand copy stream
       |
 wait(g) <---------- ready flag -------------- written on that same stream,
       |                                      after the required copies
       |  No misses: wait returns without a host reply.
       v
 experts read slot = ss_dg_slot(&view, g, id)
 part address = part base + slot x part stride

 optional engine prediction for layer g+1:
 predict(g, ids) --- mapped prediction record -> prefetch on a separate copy stream,
                                                within slot and prefetch limits
 ... next layer ...

 end -------------- completion word ---------> ss_end waits for device and service;
                                                maintains the cache and folds stats
```

- **On the device:** `slot_of` maps each expert id to a slot. Each weight part has its own array. Kernels read it at
  `part base + slot x part stride`.
- **On the host:** the service reads a record for every layer from mapped host memory, which both CPU and GPU can
  access. It validates ids and updates usage counts and recency. For misses, it arranges copies from pinned RAM and
  publishes slot-table entries. The demand copy stream writes the ready flag after the required copies and table
  updates. The engine can also supply predicted ids for the next layer. The service prefetches them within its slot
  reserve and per-layer and per-token limits.
- **Failure handling:** a wait timeout or an invalid routed id sets a persistent abort flag. Every weight lookup must
  use `ss_dg_slot` (`ts_dg_slot` in `tierstream/ts.h`); after abort, it returns a zero-filled guard slot. `ss_end`
  returns an error and poisons the context. The engine must commit its KV, carry, and speculative state only after a
  successful `ss_end`. On failure, discard the token's outputs and, once outstanding work can no longer write model
  state, restore the last committed state or terminate the session. Reopening the cache alone does not repair state
  written by the failed step. Every layer's weight readers must finish before the next layer's export. The end kernel
  must follow all readers and joined branches.
  - Each `ss_begin` authorizes exactly one gated step. The step's begin kernel advances the device sequence and checks
    it against the host authorization; a mismatch aborts the step, so lookups return the guard slot. A stray step
    still queued when `ss_begin` runs can consume that authorization: `ss_end` may succeed for that token, with the
    desync reported at a later `ss_begin` or `ss_end`. Never launch extra gated steps outside the pair, and
    stream-synchronize any prior warm-up before starting a pair.
  - After the layer's wait, `ss_dg_slot` returns the guard slot for an unexpected absent table entry. If the view
    includes a non-null `absent_err` field, the lookup sets that word; `ss_end` then fails the token and poisons the
    context. Older, smaller views return the guard without reporting the absent entry.
  - Before an eager gated step, load every kernel that may first launch while a wait kernel is spinning. CUDA lazy
    loading can otherwise block that launch until the wait deadline. Set `CUDA_MODULE_LOADING=EAGER` before CUDA
    initialization, or call `cudaFuncGetAttributes` on each such kernel before launching the step. Any full gated
    warm-up token must use its own `ss_begin`/`ss_end` pair and finish before the next pair. Captured graphs load their
    kernels during instantiation.
- **Budgets and statistics:** it uses the existing planner, then checks the gated layout at open. Every weight unit
  must fit in GPU memory or pinned RAM; decode does not fetch weights from disk. The checks include device tables and
  controls, mapped records, staging buffers, and enough spare host blocks. Existing statistics APIs remain available.
  Counters update after successful `ss_end` calls; snapshots expose the last publication. Statistics reset is allowed
  only between tokens.
- **Warm start and limits:** `dg_seed` supplies `(gated ordinal, unit index)` entries in priority order, most recent
  first. At open, resident slots are filled from distinct seed entries, then planned residents, then remaining units in
  key order, up to the persistent-slot count minus the spare target. Out-of-range seed entries are rejected; keep the
  array valid through `ss_open`.
  - `dg_spare_slots` sets one global spare-slot count; zero uses the plan's per-layer counts summed across gated
    layers. `dg_spare_target` is the boundary's free-slot target, at most the billed spare count; its default is that
    full count. Between tokens, `ss_dg_set_budgets` changes the target for the next boundary and prefetch limits for
    the next token. The default prefetch reserve follows the target; an explicit reserve stays fixed. Boundary
    write-backs are limited by available free pinned RAM blocks, so remaining victims stay resident and the target may
    not be reached.
- **Write-back scheduling (on by default):** a boundary that copies its evicted experts back to RAM right away makes
  the next token's first demand copies queue behind them on a GPU whose one copy engine serves both directions. With
  `dg_wb_defer = 1` (the default; 0 restores the immediate copies) the boundary only chooses the victims and
  unpublishes them. The service thread then copies
  them back part by part during the next token, right after a layer's ready signal and prefetches, when the copy
  engine is otherwise idle. It never copies after the last gated layer, and stops as soon as the device has
  exported the next layer. `dg_wb_layer_bytes` sets the budget per layer (default: one part).
  `dg_wb_max_outstanding` caps the bytes in flight (default: two experts). `dg_wb_stream` picks the write-back stream
  (default: the copies overlap the next layer's demand copies) or the demand stream (0: they queue ahead of them). A
  victim that is routed or predicted again before any of its bytes were copied
  costs no copy. A demand that finds no free slot waits for a write-back already in flight, or else finishes the
  oldest deferred one. Unfinished write-backs carry into the next token, up to about one boundary's worth of victims,
  and the next boundary copies the older excess. Results are bit-for-bit identical. `ss_stats_dg` reports the deferred, cancelled, forced and
  boundary-flushed counts and bytes.
- **Between requests:** `ss_dg_set_sched` switches write-back deferral and ready flushing between tokens, so an
  engine can choose scheduling per request; outputs remain bitwise identical. `ss_dg_reset_open` restores the
  resolved open-time cache state. `ss_dg_residency` snapshots resident units in recency order for a later load's
  `dg_seed`. These APIs and the pipelined startup loader are merged in [PR #1](https://github.com/hyukudan/tierstream/pull/1).
  Write-back deferral is on by default through `ss_options_defaults` since
  [PR #2](https://github.com/hyukudan/tierstream/pull/2); a zero-initialized options struct still selects immediate copies.
- **Layer-wide prefill (kept-resident layers):** `ss_dg_make_resident(c, g, ids, n)` makes units of gated layer g
  resident and kept in the same cache decode uses (one copy of each expert: arena slot, pinned RAM, file), between
  tokens and synchronously, evicting unkept experts by the policy with the usual write-back rules; `SS_E_CAPACITY`
  with nothing changed when the kept units of every layer plus the spare target do not fit the persistent slots. A
  held layer is never exported: one step with `rows_per_layer[h] = 0` for every other layer reads g's experts through
  `ss_dg_slot`, then `ss_dg_keep(c, g, NULL, 0)` releases it. A gated layer with 0 rows is simply not exported in that
  step (no export, prediction or wait for it); the service skips it and its counters and score history stay as they
  were. Its copies count only in `ss_dg_stats.resident_*`. See docs/design.md §3.9.
- **Placement rounds and no-admission misses:** `ss_dg_round(c, &params, &result)`, between tokens, moves experts
  toward the rank of a decayed export count (`s = decay x s + exports since the last round`, counted on the host): the
  hottest pinned experts replace the coldest GPU ones and, three-tier, the hottest file experts replace the coldest
  pinned ones, only when the newcomer's count is above `hysteresis` (default 1.25) x the replaced one's, at most
  `max_units` (default 96) promotions and `budget_bytes` bytes per round. Kept and held experts never move. Each move
  is transactional like `ss_dg_make_resident` (a read failure is `SS_E_IO` with the cache consistent, a device failure
  poisons); `ss_dg_round_cancel` (any thread) stops a round at the next expert, and `ss_dg_round_seed_counts` starts
  the counts from a routing profile. With `dg_no_admit = 1` a miss is served from a bank slot and released at the
  boundary without evicting anything or changing the expert's home, so placement changes only through rounds (and
  `ss_dg_make_resident` / `ss_seed`). Results are bit-for-bit identical; `ss_dg_stats.round_*` counts the rounds. See
  docs/design.md §3.10.
- **Scored eviction (opt-in):** with `policy = SS_POLICY_SCORED` the boundary still keeps the same number of slots
  free, but it picks the experts least likely to be needed soon instead of the least recently used ones. Each expert
  gets an integer score from three parts: how often, in this context, the experts routed in the previous token were
  followed by this one (given the experts routed in the current token), how often it appeared in the layer's last 16
  tokens, and how recently it was routed. The weights default to 700 / 200 / 100 out of 1000
  (`dg_score_w_trans`, `dg_score_w_freq`, `dg_score_w_recency`). The service thread ranks each layer right after
  serving it, so the boundary only merges the ranked lists. The counts start empty and learn online, or start from a
  profile of recorded routes (`dg_score_profile`, bound to the model by its per-unit hashes and refused on any
  mismatch). The history lives as long as the context; `ss_dg_reset_open` returns it to its open state. Results are
  bit-for-bit identical; only which experts stay on the GPU changes. On the DeepSeek route trace used for planning
  (8,576 resident experts and 96 spares) it cuts misses from 8.14 to 7.65 per token with online learning only.
  `ss_stats_dg` reports the victims taken in score order and the ranking time.

It is opt-in and CUDA only, for decode. It requires exclusive placement and LRU or scored eviction, with
`pin_planned = 0`.
Before an engine makes it the default, it must pass the bit-for-bit correctness and performance checks against that
engine's own cache, then receive maintainer approval.

### Promotion and eviction, in both directions

- **Promotion (RAM -> VRAM, disk -> VRAM).** A miss is copied into a slot.
  - With `pin_planned = true` (prefill), the planned residents never move; only the dynamic slots adapt.
  - With `pin_planned = false` (decode), the planned residents are only a warm start, and the resident set follows
    what the router actually picks.
- **Eviction (VRAM -> RAM, or nowhere).** When a slot is needed, the cache picks a victim:
  - **LRU** (default) evicts the least recently used unit that this forward is not using. On our DeepSeek decode
    traces it hit 95-98 % of lookups, against 84-93 % for a fixed placement of the same size (design.md §3.1).
  - **SCORED** ranks units by a mix of routing transitions (which experts follow which), recent frequency and age. It
    is opt-in until it beats LRU on real traces.
- **Inclusive vs exclusive** (`tier_mode`) decides what an eviction costs.
  - **Inclusive** (default): a unit promoted from pinned RAM keeps its RAM copy, so eviction just drops the VRAM copy.
    It is simple, but the same unit can occupy VRAM and RAM at once.
  - **Exclusive**: at every forward boundary, each unit lives in exactly one of three places: a VRAM slot, a RAM
    block, or only its file. (During a transfer it is briefly in two; the files always stay as backing storage.)
    - **Promotion.** When a miss is promoted into a persistent slot, its RAM block is freed at `ss_end`. A miss
      served from the ring keeps its block.
    - **Eviction.** The victim is written back from VRAM into a free RAM block of its size: a device-to-host copy on
      its own stream. On a GPU with one copy engine (the RTX PRO 6000 here reports one), that copy shares the
      engine with the misses' copies, so it is not free. With no free block, the miss is served from the ring and the
      victim stays. With full slots and no spare blocks, the resident set stops adapting, but nothing falls back to
      disk.
    - **Capacity.** VRAM + RAM can then hold the whole model when, after the fixed bytes, workspace, ring and
      alignment, their unit capacity covers it. Design example (design.md §3.5): for the DeepSeek V4.1 RTX decode
      replica, 80 GiB of VRAM slots plus ~63 GiB of RAM blocks would hold all 142.4 GiB of experts, with no file read
      after open. The shipped pending recipe's actual budget still leaves 5.8 GiB on NVMe.

**Clean eviction for read-only units (`dg_clean_evict`, opt-in).** Expert weights never change, so their file always
holds the same bytes. With `dg_clean_evict = 1` (or `ss_dg_set_clean_evict` between tokens), a unit leaving the device
when no pinned block is free goes straight back to its file: no device-to-host copy, and no pinned unit is pushed out
to make room. This stops the "evict, write back, re-read" loop when a model's working set is larger than VRAM + pinned
RAM (for example a whole-layer prefill on a small card). Counted in `ss_dg_stats.clean_drops` / `clean_drop_bytes`.

**Unused prefetches reverted (`dg_pf_revert_unused`, opt-in).** A prefetch its token did not use is normally admitted
at the boundary like any other copy, so the boundary writes another device unit back to keep the spare slots free. With
`dg_pf_revert_unused = 1` (or `ss_dg_set_pf_revert_unused` between tokens), such a unit goes back to its pinned block
or its file instead (no copy), and the boundary evicts nothing for it. Used and kept prefetches are admitted as before.
Counted in `ss_dg_stats.pf_reverted` / `pf_revert_bytes`.

**Bank-served file reads kept in RAM (`dg_bank_keep`, opt-in).** When a layer misses a unit and no device slot is free,
the unit is served through a short-lived bank slot. A unit that lived in its file went back to its file afterwards, so
the next use read it from disk again, even when it was hot. With `dg_bank_keep = 1` the read goes into a pinned RAM
block instead of a staging buffer (a free one, else the least recently homed RAM unit's), and the unit stays in RAM as
the most recently homed one. The same bytes, hash check and copy; no extra memory. Counted in `ss_dg_stats.bank_kept`.
No effect without `dg_disk_tier` or with `dg_no_admit`. `dg_bank_keep = 2` inserts the kept unit in the middle of the
RAM recency order instead of at its head, so a unit read once no longer pushes out the units used most recently (a unit
used again moves to the head as usual).

**Predictions stay out of the demand's way.** A plain prediction for a layer the device already reached is skipped
(`pred_skipped_ahead`); a layer's ready waits only for the prefetch reads of units it actually uses (a wrong file
prefetch keeps reading in the background; a later read that needs its queue place or staging buffer, or the token's
end, still waits for it); a full read queue ends demand reads before prefetch reads; and with `dg_pf_strict_admit = 1`
a prefetch only uses free slots above the spare target, so demand misses always find theirs (`pf_strict_refused`).
These three admission changes are measured before they become the default: until then `DG_PF_LEGACY` defaults to 1
(the previous behaviour, exactly) and `DG_PF_LEGACY=0` selects them.

**Read queue for your drives (`SS_DG_DISK_QUEUE_AUTO`, opt-in).** Setting `dg_disk_queue = SS_DG_DISK_QUEUE_AUTO`
picks the three-tier reads in flight from the read locations the gated sources are read from: `clamp(3 x L, 4, 8)`,
L = the source file plus each `weight_dirs` directory holding its stripe copy (normally one per drive: one = 4, two =
6, three or more = 8). The planner bills the same readers; the result is in `ss_dg_stats.disk_queue_resolved` /
`disk_locations`. Details: [design §3.14](docs/design.md).

Off: leave `dg_disk_queue` at 0 (one synchronous read, the default) or set any value from 1 to 64. Measured with DeepSeek V4.1
Flash on one RTX PRO 6000 limited to 24 GiB of VRAM and 32 GiB of host RAM, one drive, output identical in every arm:
5.45 tok/s at queue 1 -> 6.55 tok/s at queue 4 (+20 %), +26 % with speculative decoding; queue 16 = queue 4 (one
drive saturates at 4).

**Read depth that follows the drive (`dg_disk_queue_adapt`, opt-in, with AUTO).** More reads in flight usually help,
but not always: in a drive's slow phase the same reads at queue 4 took 22 ms each against 5-8 ms one at a time, and
decode was 32 % slower than at queue 1, while an hour later on the same drive queue 4 was 8 % faster. With
`dg_disk_queue_adapt = 1` the AUTO readers stay allocated, but the pool runs demand reads either all at once or one at
a time, whichever it measures faster: it compares the bytes per second of groups of at least 32 reads at both depths
(U, 1, 1, U), keeps the higher depth only when both comparisons show it at least 3 % faster, checks again every 512
reads (4096, then 8192, at depth 1) and at once when the rate moves by more than 25 %. The reads, their order and their
bytes do not change, only how many run together. Not used with landing blocks or file prefetches. Reported in
`ss_dg_stats.disk_depth_*` and in the stats JSON (`three_tier.disk_depth`, with the last 16 decisions). Details:
[design §3.17](docs/design.md).

**Reads in flight on one drive.** With `dg_disk_queue` above 1 and no stripe copies, the three-tier reads of a source
now go through an overlapped file handle, so the readers really are in flight together on the drive (before, Windows
ran them one at a time on the source's synchronous handle, and a deeper queue only added waiting). The same bytes are
read; `STRIX_FILE_SYNC_READS=1` keeps the previous single synchronous handle. Measured with GLM-5.3-Flash on one RTX
PRO 6000 limited to 24 GiB of VRAM and 32 GiB of host RAM, one drive, five alternating arms, output identical in every
arm: queue 8 with overlapped reads +7.6 % (1K prompt) and +13.8 % (8K) cold decode against queue 1, and faster in 10
of 12 arm-and-cell pairs; the two slower warm cells come from one arm in which the drive read the same bytes at about
half its earlier rate. The same queue on the synchronous handle was 7.5-35 % slower than queue 1. At queue 1 reads are
issued one at a time as before.

**Two-stage file prefetch (`dg_landing_blocks`, opt-in).** In a three-tier open a predicted file unit normally needs
a free GPU slot, which is rare when VRAM is small. With `dg_landing_blocks = L` (and `dg_disk_queue > 1`), L pinned
landing blocks are set aside: a predicted file unit is read into one while earlier layers compute, taking no GPU slot
and causing no eviction or write-back. Its demand then costs one host-to-device copy instead of a file read. Landing
reads queue behind demand reads (at most `dg_landing_max_queued` running), and a landing its layer did not use returns
to its file right after that layer (`dg_landing_ttl = 0`, the default; N >= 1 keeps it N tokens), so a few layers'
worth of blocks serve every layer. Off by default; the planner charges the blocks. Counted in `ss_dg_stats.landing_*`.

Landing reads yield to demand reads: they run in 4 MiB chunks (`dg_landing_chunk`) and hand their reader to a waiting
demand read between chunks, one reader always stays free for demand reads, and a landing its layer no longer needs
stops at its next chunk. Measured with GLM-5.3-Flash on one RTX PRO 6000 limited to 24 GiB of VRAM and 32 GiB of host
RAM, model striped over three NVMe drives, 16 landing blocks, 8 landing reads, output identical in every arm: decode
+19.6 % / +22.6 % (1K prompt cold / warm) and +39.6 % / +32.5 % (8K prompt) against no landing; the same landing with
whole, non-yielding reads (`dg_landing_chunk = 0xFFFFFFFF`, the earlier scheduler) measured about equal to no landing,
and slower at the same depth before. Repeats of one setting varied by 15-55 %, so the sizes are not precise; all four
cells were faster.

**`ss_dg_make_resident` reads in flight.** Up to `min(dg_disk_queue, dg_disk_staging)` file reads run together and are
finished as they complete; the copied units still become the most recent in the order given, so placement does not
depend on read timing. Its own reads, bytes, wait time, host-to-device and write-back bytes are in `ss_dg_stats.mkres_*`.

**Layer tokens for layer-major prompt blocks (`ss_dg_begin_layer`, opt-in).** An engine that runs a prompt block one
layer at a time can open a token that serves only gated layer g, in up to 32 parts: `ss_dg_begin_layer(c, g, parts)`,
then on the device `ss_dg_enqueue_export_part` / `ss_dg_enqueue_wait_part` per part and `ss_dg_enqueue_end_layer` after
the layer's readers. A unit fetched by one part covers the later parts (one copy per unit per layer token). When a
demand finds no free slot, a device unit of another gated layer moves off the device the boundary's way (honouring
`dg_clean_evict`); units kept by `ss_dg_keep` or held by `ss_dg_make_resident` are never moved. A full export inside a
layer token fails it as a sequence desync. Ordinary tokens (`ss_begin`) are unchanged. Placement only: the same bytes.
Counted in `ss_dg_stats.lm_*`.

**Held layers overlap other layers' loads.** A layer held by `ss_dg_make_resident` may run its kernels outside
`ss_begin` / `ss_end`. `ss_dg_make_resident` / `ss_dg_keep` of other layers may overlap them: they never evict a held
unit, sync only the cache's own streams, and, while a layer is held, upload only the changed layers' table entries, so
a fully held layer is never rewritten. Release a held layer only after its kernels completed; its slots may be reused
by the next call.

### Fetching ahead

- **Certain prefetch** (`ss_prefetch_layer`): layer-by-layer prefill knows it needs layer L+1 next, so that layer is
  copied while layer L computes (recipe engine knob `prefetch_layers`).
- **Speculative lookahead** (`ss_prefetch` with `SS_PREFETCH_SPECULATIVE`):
  - For MoE decode, the engine runs layer L+1's router on layer L's input, and `ss_lookahead_select` turns the scores
    into the few experts most likely next (`lookahead_units`, default 2).
  - They are read at low priority, only while no demand is waiting.
  - By default they land in the ring and never evict a resident on a guess (`spec_persistent = true` lets them in).
- **Host read-ahead** (`readahead_units`; the engine calls `ss_host_prefetch_layer`, recipe engine knob
  `host_prefetch_layers`):
  - Disk is the slowest tier, and a disk read normally cannot start until a GPU slot is free.
  - Read-ahead reads the next layers' disk units into pinned RAM buffers early, so when the slot frees up only the
    RAM -> VRAM copy is left.
  - Before it existed, MiniMax RTX prefill at 8K waited ~0.45 s per disk layer (design.md §3.6).
- **Seed** (`ss_seed`): between requests, load a usage profile's favourite units into free slots.

Engine knobs (`prefetch_layers`, `host_prefetch_layers`, …) only take effect in an engine that implements them.

### Several drives (multi-disk streaming)

A model streamed from one NVMe drive is limited by that drive. tierstream can read the same model from several
drives at once, without a second full copy:
- `ts-stripe make MODEL.gguf RANGES.json STRIPE_DIR` copies part of the model to another drive as a
  **stripe**: whole chunks (16 MiB) on a fixed grid, listed in a manifest with the model's SHA-256 and the xxh64 of
  every chunk. A ranges file says which bytes to place; `interleave` puts every other chunk of a range on the stripe,
  so a whole-layer expert stream or a batch of random Engram rows is read about half from each drive. For DeepSeek
  V4.1 Flash Q2 the routed experts and both Engram tables interleaved = 166 GiB on the second drive.
- Readers pass the stripe directories (`weight_dirs`, engines `--weight-dirs MODEL_DIR;STRIPE_DIR`).
  Sessions share **one reader pool per volume and reader count** across files (`drive_readers`, 1..32, default 4); unbuffered reads
  stay unbuffered. Large requests (8 MiB and up) are split between the drives with **adaptive shares** that follow
  each drive's measured speed (a slow drive gets little or nothing, and every 16th large request reads the original
  alone as the live baseline); small and latency-bound reads stay on the original drive, as with one drive.
- **Verification and trust.** A stripe chunk is served only after its xxh64 matched the manifest (`weight_verify`:
  `always` checks every chunk at open and re-checks whole-chunk reads inline, `cached` checks once per data-file
  version). Read errors, short reads and mismatches fall back to the original file; missing or stale stripes disable
  that location. Stripe directories must be as trusted as the primary: xxh64 detects accidental corruption, not
  deliberate tampering. A supplied `primary_sha256` is trusted without checking and must describe the current primary.
  Windows holds serving files with `FILE_SHARE_READ` only, excluding writers. Linux has no share-read-only exclusion:
  stripe data is published mode 0444, but its owner or a privileged process can still change it. Identity, including
  Linux ctime, is checked at reuse and after verification. Keep primary and stripe files immutable while sessions live.
  Use `weight_verify=always` on Linux where stripe directories are not exclusively owned; partial reads still rely on
  open-time verification. Byte identity assumes these trust and immutability requirements.
- **First open.** Without an admitted `primary_sha256` or an identity cache hit, opening a striped primary hashes the
  entire primary and scans the stripe. Hundreds of GiB can take minutes; cached subsequent opens avoid these scans.
  The `ss_mfile` C API permits concurrent reads and queries with separate destination buffers. Close must not race any
  call using that handle; separate handles to a shared session can be closed independently.
- With no stripe directory (the default) nothing changes: the original single-file code path runs.

**Measured** (DeepSeek V4.1 Flash Q2 on the RTX PRO 6000 during prompt prefill, exact profile, 32 GiB pinned host tier;
original on a PCIe 4.0 NVMe, stripe on a second PCIe 4.0 NVMe; the same tokens and states in every arm):

| prompt | one drive | two drives | |
|---:|---:|---:|---:|
| 8K | 13.2 s | 10.6 s | -19 % |
| 32K | 19.2 s | 17.4 s | -9 % |
| 64K | 29.1 s | 27.8 s | -5 % |
| 128K | 49.9 s | 48.4 s | -3 % |

The expert stream itself goes from 11.0 to 8.45 s per request (-23 %, 29 % of the bytes from the second drive) and the
service starts 17 % faster; longer prompts gain less because they are compute bound. A third, slower drive (a
DRAM-less system disk) added nothing: the shares give it little. Decode reads single experts and Engram rows; with the
previous routing (`STRIX_MFILE_ROUTE_LEGACY=1`) those stay on the original drive, so decode speed does not change.

**Balanced reads across stripe copies (default for striped models).** With `weight_dirs` stripe copies, a read of
1-8 MiB (one expert's parts during decode) whose bytes all have a copy on one stripe drive now goes to whichever of
the primary file and that copy has fewer bytes in flight. Before, such reads went to the primary unless other large
reads were already queued, so the primary drive carried most of the decode traffic. Reads of 8 MiB and more are still
split across drives as before. Output is identical: the same verified bytes, only the drive that serves them changes.
The rule assumes stripe drives about as fast as the primary; a stripe drive that measures under half the primary's
speed on large reads gets none of these reads.

Off: `STRIX_MFILE_ROUTE_LEGACY=1` restores the previous routing exactly. Measured on GLM-5.3-Flash on one RTX PRO 6000
limited to 24 GiB of VRAM and 32 GiB of host RAM, model striped over three NVMe drives, two runs per arm in both
orders, output identical in every arm: decode +5.5 % / +8.5 % with a 1K prompt (cold / warm) and +17.2 % / +0.3 % with
an 8K prompt, +7.7 % on average, never slower; the primary drive's share of decode reads goes from 40-46 % to
28-37 %. Repeats of the same configuration differ by up to about 19 %, so the size of the gain is not precise.

### Prepared packs (one pack per model, every layout)

Engines convert some weights at load time (a tensor-parallel rank keeps its half of each expert's rows, a split
quantized block, a repacked layout). A **pack** stores those bytes once, in each engine's final device format, so a
load is a plain read straight into place. One pack per model serves every layout:
- an expert-parallel rank reads the whole experts of its group;
- a tensor-parallel rank reads its row range of each expert in one read (or its own per-rank record when a format
  cannot be split by rows);
- the device-gated cache reads units by key: `ss_pack_dg_descriptor` turns the engine's descriptor into one whose
  expert parts point into the pack.

A pack is a directory: `pack.json` (model, the SHA-256 of every source file, how each format was produced, the
SHA-256 of every data file and of the index), `pack.index` (one record per tensor or expert part: key, format,
offset, bytes, rows, xxh64) and `pack-NNNN.data` (records 4 KiB aligned, the parts of one expert next to each other).
Pin the SHA-256 of `pack.json` and it pins everything else; records are checked against their xxh64 the first time
they are read (`cached`, default), on every read (`always`) or not at all after `ts-pack verify --full` (`off`). A pack
built from other source files than the ones the app loads is refused. Pack data files are ordinary files for
`ts-stripe`, so a pack can be spread over several drives like a model file.

```sh
# The engine describes where its device bytes come from (tierstream/pack-layout-v1: byte ranges of the model file,
# optionally row selections and byte rearrangements inside rows); the tool prints the sizes first.
ts-pack build LAYOUT.json /models/x.pack --dry-run
ts-pack build LAYOUT.json /models/x.pack            # prints the pin: the SHA-256 of pack.json
ts-pack verify /models/x.pack --pin SHA256 --full   # every file's SHA-256 and every record's xxh64
ts-pack show /models/x.pack                         # sizes per format, files, sources
```

An engine whose device format is not a byte rearrangement of the model file writes the pack itself through the writer
API (`ss_pack_writer_begin` / `ss_pack_writer_add` / `ss_pack_writer_finish`: the pack becomes visible only when it is
complete). Engines open a pack with `ss_pack_open` and read with `ss_pack_find` + `ss_pack_readv`; without a matching
pack (or with the engine's switch off) they keep their original loader. Format and API: [docs/pack.md](docs/pack.md).

### What never changes

Bytes are copied, never transformed, and engine kernels must not depend on which slot a unit sits in. So every
placement, policy and prefetch setting gives bit-identical outputs. With `verify_hash` enabled and descriptor hashes
supplied, unit bytes loaded from files are checked before they are accepted, and a mismatch is an error.

### One cache for every engine

Every engine in this lab (DeepSeek, MiniMax, Qwen, GLM, on the RTX card and on the Strix Halo nodes) reads its
streamed weights through tierstream instead of keeping its own expert cache. The residency, prefetch, prediction and
write-back logic lives here once:
- engines describe their units in a descriptor;
- recipes choose the budgets;
- the engine-side code shrinks to an adapter that names the units it needs and reads them through the tables.

### Whole-device budgets

`--vram-total-gib 24` means "run as if this were a 24 GB card": one cap for everything on the device, one for the
host (`--ram-total-gib`, pinned included). `ss_budget_plan` subtracts the measured CUDA context and the engine's
named fixed parts (trunk, KV for the requested context, workspaces, drafter, pinned and pageable host buffers, load
staging); the expert arena gets what is left of the device, the pinned tier what is left of the host, the file tier
the rest. When the fixed parts alone do not fit, it refuses at startup naming them and the smallest cap that would
work. With no cap it gives exactly today's three-tier placement. A device-gated engine also gives it the library's own
host memory (`ss_budget_in_dg_library` fills it from the descriptor and options: the mapped control area, one bounce
buffer per file reader with `dg_disk_queue`, the staging buffers), charged as a fixed host part and reported in
`ss_budget_out.lib_*` with the same arithmetic `ss_open` admits it with. `ss_budget_ledger` (passed as
`ss_options.budget`) charges every allocation tierstream makes against the same caps and keeps the peak per kind with
the components at that peak, next to the peak the engine measured (cudaMemGetInfo / NVML). Details:
[design §5.1](docs/design.md).

### Running on your hardware (device profiles)

At startup tierstream can describe the machine it runs on as a device profile (`strix/device-profile-v1`): host RAM,
whether the GPU shares it (Strix Halo and its successors: one memory pool), each GPU's arch, compute units and memory,
the USB4 links to a second box, the NVMe drives. It reads Linux sysfs (KFD topology, amdgpu memory totals,
Thunderbolt peers), the HIP runtime, or on NVIDIA the CUDA driver and NVML (read-only queries, no allocation). A
device class (`profiles/*.json`: `strix-halo-395`, `rtx-pro-6000-blackwell`, `rtx-4090`, generic) adds the safety
margins and the measured numbers detection cannot know; a machine with no class (a new gfx, another card) runs on its
detected values and today's margins, with a notice, and is never given another arch's tuned values.

What the profile drives, when no explicit flag is given:
- **GPU / RAM / disk split**: `ss_devprof_budget` sets the whole-device caps (discrete card: VRAM - reserve and
  available RAM - floor; unified: one pool shared by the expert arena and the pinned tier); `ss_budget_plan` then
  places experts on the device, in pinned RAM and on disk as before.
- **One box or two**: `ss_devprof_topology` runs single-box when the model's fixed parts and a minimal expert tier fit
  this box's budget at the requested context, else across the boxes with a measured link, splitting the layers by
  budget share. A qualified recipe can pin its choice.
- **Prefill piece size**: `ss_devprof_piece_rows` sizes the piece from the memory left after the plan, within the
  engine's own bounds.

```
ts-devprof                                  # the effective profile (JSON) of this machine
ts-devprof --budget                         # the caps it gives the planner
ts-devprof --device-profile my-box.json     # override: a full or partial profile merged over the detected one
STRIX_DEVICE_PROFILE=my-box.json <engine>   # the same override for any engine that reads profiles
```

Explicit `--vram-total-gib` / `--ram-total-gib` (or a recipe's numbers) always win over the profile. A partial
override is enough to correct one value, e.g. `{"margins": {"host_floor_bytes": 17179869184}}` keeps 16 GiB free on
the host. Every receipt can carry the profile's hash (`ss_devprof_hash`) and the merged override paths, so a result
says which machine numbers drove it. Details: [design §5.2](docs/design.md).

## When it pays (and when it does not)

The question is how many weight bytes each token must move across PCIe, compared with what the GPU reads from its
own memory, which is much faster.
- On an RTX PRO 6000 (96 GB, Windows WDDM), pinned RAM ↔ VRAM microbenchmarks measured about 22 GB/s
  with a busy PC and 34-40 GB/s with a quieter PC: 40 % and 62-73 % of the lab's 55 GB/s practical PCIe reference,
  respectively (not the theoretical link rate). The unchanged comparison benchmark shows the same effect, so the
  load-dependent loss is not specific to tierstream. Host-memory contention is suspected but not isolated.
  The device exposes one copy engine; queued write-backs can delay demand copies across streams.
- NVMe is slower still.

| workload | streaming | why |
|---|---|---|
| **MoE decode** (DeepSeek, MiniMax, Qwen MoE, GLM) | can pay, with a hot set | A token uses only the top-k experts of each layer (6 of 384 for DeepSeek V4.1). When routing has a hot set, as on our DeepSeek traces (95-98 % LRU hits), most lookups need no transfer and a miss costs one expert copy. |
| **Prefill and large batches** (dense or MoE) | can pay | Many tokens reuse every weight byte, so the copy of layer L+1 can hide behind layer L's compute, as long as that compute takes longer than the copy. |
| **Models that do not fit at all** | makes them run | The alternative is not running, or splitting across machines. That is a capacity gain, not a speed gain. Exclusive mode removes disk reads only when VRAM + RAM cover the units. |
| **Dense decode** (one or a few tokens per step) | does **not** pay | Every token reads every weight. Whatever is not resident crosses PCIe on every token, far slower than VRAM, so the streamed fraction dominates the step time. Use a smaller quantization, more GPUs, or a unified-memory machine instead. |
| **MoE decode with near-uniform routing** | pays less | With no hot set, the miss rate approaches the fraction of experts that are not resident. |
| **KV cache, adapters** (later) | not yet | The same tiers fit other large, unevenly used byte ranges. Units today are weights only. |

### Adaptive memory and NVMe measurements (in development)

The `adaptive` branch treats expert memory and KV/state as one device budget: reserve state for the selected
context, then place experts across VRAM, pinned RAM and NVMe. A `tierstream/usage-v1` table ranks experts by measured
use, decayed recency and stable unit order. It includes a usage updater, miss-curve simulator and what-if solver;
these are not yet shipped. The generic file tier already exists; device-gated NVMe support is separate development.

Measured on an RTX PRO 6000 (96 GB, Windows WDDM): replay of held-out code decode routes from DeepSeek V4.1 Flash,
using real GGUF experts (IQ2_XXS gate/up, Q2_K down) and a compute stub sized to a 6.18 ms all-hit floor.
The context labels below are **memory budgets**, not full-model context-length benchmarks. Sampled expert bytes
matched bit for bit (924 checks, no failures); this does not qualify full-model logits or generated tokens.

| Context budget / state allocation | Experts: VRAM / pinned / NVMe | ms/token | Throughput as % of stub ceiling | Pinned / NVMe misses per token |
|---|---:|---:|---:|---:|
| 8K, maximum expert capacity | 8,594 / 6,766 / 0 | 9.06 | 68.2 % | 8.39 / 0 |
| 128K, lean state allocation | 7,713 / 7,215 / 432 | 10.43 | 59.2 % | 11.71 / 0.036 |
| 128K, current state buffers | 5,113 / 7,233 / 3,014 | 32.7 | 18.9 % | 78.5 / 4.5 |

Host read-ahead did not help this decode replay: persistent-slot admissions re-read the file instead of consuming
ready read-ahead buffers. PR #2 fixes that host-mode path and adds opt-in `exclusive_discard` when no free host block
is available. A decode prefetch speedup has not been measured with the fix.

Retained host-copy pools did not justify their memory cost in CPU trace simulation. Lossless compression also gave
no useful gain: 298 sampled experts across all 40 DeepSeek V4.1 Flash MoE layers, with the same IQ2_XXS/Q2_K bytes,
compressed only 1.008× with zstd levels 1-19 and round-tripped bit for bit. Neither approach is being adopted.

## Recipe knobs at a glance

A recipe (below) compiles to the library's budget and options. The knobs that shape streaming:

| knob | default | what it does |
|---|---|---|
| `device_gib` | the device set's `memory_min_gib` | the VRAM budget |
| `host_pinned_gib` | 0 | the pinned-RAM budget |
| `workspace_gib` | derived from the descriptor and the route's bounds (refused if it cannot be derived) | the engine's own VRAM |
| `units`: `resident` / `pinned` / `rest` | `"fill"` / 0 / `"disk"` | how many units of each group start in VRAM, in RAM, and on disk |
| `pin_planned` | true | true: planned residents never move (prefill). false: they are a warm start for LRU / SCORED (decode). |
| `dynamic_slots_per_layer` | 0 | extra VRAM slots per layer for the adaptive set |
| `ring_slots` | 0 (auto) | temporary VRAM slots for misses that are not admitted, and for prefetches |
| `policy` | `lru` | `lru` or `scored` eviction |
| `tier_mode`, `host_spare_units` | `inclusive`, 0 (auto) | inclusive or exclusive; free RAM blocks requested for write-backs |
| `lookahead_units`, `spec_persistent` | 2, false | speculative experts per layer; whether a guess may take a persistent slot |
| `readahead_units`, `host_prefetch_layers` | 0, 0 | host read-ahead buffers and distance |
| `prefetch_layers` | 1 | certain whole-layer prefetch distance |
| `reader_threads`, `staging_mib`, `direct_io` | 2, 256, true | the disk reader |
| `weight_dirs`, `drive_readers`, `weight_verify` | none, 4, cached | stripe copies on other drives (several drives) |

The full list, with units and checks, is in [docs/recipes.md](docs/recipes.md).

## Build

```
# Linux / WSL
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build   # -DTS_CUDA=ON, -DTS_HIP=ON
# Windows (MSVC + Ninja), CPU or CUDA
tools\build-win.bat          &  tools\build-win.bat cuda
```

The tests write small synthetic model files under `TS_TEST_DIR` (or `SS_TEST_DIR`), or the system temp directory if it is not set. The
CUDA test `ts-test-cuda` needs a GPU. It checksums every streamed unit on the device and probes pinned and NVMe
bandwidth with about 256 MiB each.

`ts-plan DESC.json --device-gib X --host-gib Y` validates a descriptor and prints its tier plan, in the same format
hyperloom's planner reads.

To rebuild, rerun the gates and reproduce the measured DeepSeek results, see [docs/reproduce.md](docs/reproduce.md).

## Recipes

A recipe is a small, human-editable TOML (or JSON) file that says how one model runs on one set of hardware, for each
kind of request. The full schema, with every field, unit and default, is in [docs/recipes.md](docs/recipes.md).

A recipe has four parts:

- **match**: the model and the hardware it is for: device sets, memory, host RAM, disk and PCIe bandwidth, and the
  links between sets.
- **routes**: ordered rules that pick which devices serve a request, from its phase (prefill or decode), prompt
  length and batch.
- **placement**: for each route and device, the memory budget, the streaming knobs, and which weight units are
  resident on the device, which come from pinned host memory, which are read from NVMe, and which another device
  computes.
- **limits and fallbacks**: what happens when a route cannot run.

A short example (a model that prefills long prompts on a discrete card and decodes on a two-node pair):

```toml
schema = "strix-stream/recipe-v1"
name = "example-lab"

[match]
model = "example-moe"

[[match.devices]]
id = "card"
kind = "cuda"
memory_min_gib = 90.0

[[match.devices]]
id = "pair"                       # two unified-memory nodes run as one set
kind = "hip"
count = 2
memory_min_gib = 100.0
internal_link = { kind = "usb4" }

[[match.links]]
id = "card-pair"
between = ["card", "pair"]

[[routes]]
id = "prefill_long"               # long prompts: prefill on the card, hand the state to the pair
phase = "prefill"
devices = ["card"]
execute = "single"
handoff_route = "decode"
links = ["card-pair"]
fallback = "prefill_pair"         # card busy, link down or low RAM: the pair prefills
when = { prompt_tokens_min = 4096 }
x-evidence = "the measured crossover goes here"

[routes.placement.card]
device_gib = 90.0
host_pinned_gib = 48.0
workspace_gib = 4.0
max_context_tokens = 32768
max_batch = 1

[[routes.placement.card.units]]   # 60 layer blobs: 32 resident, 16 pinned, 12 from NVMe
id = "layers"
class = "blob"
resident = 32
pinned = 16
rest = "disk"

[[routes]]
id = "prefill_pair"
phase = "prefill"
devices = ["pair"]
execute = "tp"
handoff_route = "decode"

[routes.placement.pair]
workspace_gib = 2.0
max_context_tokens = 32768
max_batch = 4

[[routes]]
id = "decode"
phase = "decode"
devices = ["pair"]
execute = "tp"

[routes.placement.pair]
workspace_gib = 2.0
max_context_tokens = 32768
max_batch = 8
```

**Routes and fallbacks.** Routes are tried in file order and the first whose `when` holds is taken, if it can be
admitted: its devices are available, its links are up, the request fits its context and batch bounds, and every host
it uses keeps its free-RAM floor after the route's compiled memory. Otherwise its `fallback` is tried, then the next
routes. Each phase ends with a catch-all route. A running request never changes route.

**Placement.** Unit groups rank units (model order, a fixed list, or measured frequency) and fill the tiers in order:
integer counts first, then `"fill"` (as many as the device or host budget holds, by the same rule as the planner),
then the rest. The result is an ordinary `strix-stream/plan-v1` placement, so an engine opens it like any other plan;
a placement that does not fit is refused, never shrunk. Heterogeneous routes split the units of one model between
devices, with exactly one owner per unit.

**Autotuner and overrides.** Three layers make the effective recipe: the recipe, an autotuner profile, and human
overrides (files or `--set path=value`), highest last. A profile is bound to the recipe's bytes, the engine, the
model, the devices, their drivers and the descriptors, which must all still match, and to the probed bandwidths,
which may drift by up to `max_drift` (default 10 %). Otherwise the profile is refused and the recipe runs untuned. A recipe can pin fields the autotuner may not change. `ts-recipe explain` shows, for
every field, its value, which layer set it and what it replaced.

**Exactness is not an option.** Every placement a recipe can express produces bit-identical outputs; a recipe that
asks for "exact" anything is refused.

```
ts-recipe validate RECIPE                 # strict check, prints the effective recipe
ts-recipe explain  RECIPE --set routes.prefill_long.when.prompt_tokens_min=3072
ts-recipe match    RECIPE PROBE.json      # does this hardware fit? which devices go where
ts-recipe select   RECIPE STATE.json      # which route admits this request, and why not the others
ts-recipe plan     RECIPE --route R --device-set S --desc DESC.json   # budget, options and plan-v1
```

The same calls are in the C ABI (`ss_recipe_*`).

**Example recipes** are in [recipes/examples/](recipes/examples/README.md), calibrated on the lab's measured setups
(two Strix Halo nodes over USB4 and an RTX PRO 6000 (96 GB, Windows WDDM)):

- validated by `ts-recipe` (admission on a given machine may still refuse them): DeepSeek V4.1 Flash (pair, and the
  measured RTX + pair hetero placement), MiniMax M3.1 Flash, Qwen3.8 Flash Next;
- `pending/`: recipes that wait for an engine run, a descriptor or hardware: a DeepSeek RTX decode replica, DeepSeek
  and MiniMax on a 32 GB card, MiniMax with >= 192 GB of host RAM, GLM 5.3 Flash.

Every number in them carries its evidence in `x-` keys (`x-evidence`, `x-cites`, `x-predicted` with a basis:
measured, derived, projected, bound or target). CTest validates each file with `ts-recipe` and checks that the
structural negative fixtures in `recipes/examples/tests/` are refused; `validate_v1.py` adds the citation checks
(it needs the lab notes the `ref`s point into).

## Names

strix-stream was renamed tierstream. For this release the change is purely additive:
- **C ABI.** `ts.h` adds the `ts_*` names on top of the unchanged `ss.h`. The types are the same types and each
  `ts_*` function calls its `ss_*` function, so old and new code link against one library.
- **Build and tools.** The CMake target is `tierstream` (alias `strix_stream`). The library file is built under both
  names. The tools are `ts-recipe` / `ts-plan`, and `ss-recipe` / `ss-plan` are built too, each printing its own
  name. The options are `TS_CUDA` / `TS_HIP` / `TS_TESTS`, and an `SS_*` value still works.
- **Formats.** Every parser accepts `tierstream/<format>` and `strix-stream/<format>`. Every output still writes
  `strix-stream/<format>` in this release, so no recipe hash, plan or stats document changes. Recipe files are
  unchanged.

The old names are removed in a later release, once every consumer has moved.

## Status

v1. The CPU tests pass on MSVC and on GCC with ASan, UBSan and TSan. The CUDA backend is built and its GPU test runs
in one short window. Exclusive mode and host read-ahead are opt-in.

The HIP backend for Strix Halo (unified memory) is built with the nodes' ROCm toolchain (`-DTS_HIP=ON`) and its GPU
test passes on a Strix Halo node (gfx1151). `ts-test-hip` is the CUDA GPU test compiled for HIP. Decode, prefill and
read-ahead reads all check out on the device, and a read failure is reported and recovered. The test also measures
landing throughput into GPU-mapped host memory: about 29 GB/s from the pinned tier and about 6 GB/s from an unbuffered
file. Exclusive mode requires a discrete device in this implementation and is refused by the unified-memory HIP
backend.

Device-gated decode is opt-in and CUDA only. Its GPU tests cover graph-captured decode, prediction, memory pressure,
protocol failures, deferred write-backs, pipelined loading, reset, seeding and scored eviction. The single-copy-engine
exception and required gates are documented in [docs/reproduce.md](docs/reproduce.md). The full-model DeepSeek bit-for-bit
comparison against the replica's own cache passed, and so did speed parity against that cache (no slower chat,
A/B/A); tierstream is now that replica's only cache.

Measured with DeepSeek V4.1 Flash on one RTX PRO 6000 (96 GB), Windows WDDM, device-gated mode,
Q2 GGUF experts (IQ2_XXS gate/up, Q2_K down; dense Q8_0), exclusive VRAM + pinned-host cache.

| Merged feature | Measured comparison | Exactness / conditions |
|---|---|---|
| Pipelined loading (on by default) | Full load: 108.7 → 61.8 s; expert fill: 83.1 → 38.0 s; 2.25 → 6.73 GB/s | Serial vs pipelined with 8 readers; disk-bound; bitwise-identical placement and outputs. `DG_LOAD_SERIAL=1` restores serial loading. |
| Reset to open state | About 3.4 s to restore the fresh-load cache state | Requests after reset were bitwise identical to a fresh process. |
| Residency snapshot / seed | Fresh-process step 0: 332.6 → 156.0 ms; mean steps 0–15: 56.05 → 44.67 ms/step | Fresh load seeded from a prior request's resident set vs baseline fresh load; bitwise-identical outputs. |

Window prefetch (`winpf`) is merged, exact and off by default. Evaluation is closed: matched warm comparisons showed
no repeatable gain. Persistent loaded workers and graph pre-capture/warm-up are in development in the engine
(replica); tierstream supplies the cache reset, scheduling and residency APIs.

## Credits

The design of this library was inspired by [mlx-stream](https://github.com/davidtai/mlx-stream) by David Tai, a
DeepSeek-V4.1-Flash plugin for mlx-serve that streams experts from SSD. Its approach to MoE weight streaming
(scored residency, next-layer router lookahead, unbuffered expert reads into device slots, hits-first gating) is
the starting point for the mechanisms here; thanks to its author for publishing it.

No third-party source code is included: every file in this repository is new code written for this project (see
NOTICE). If a ported file is ever added, it keeps its original license notice in the file and in NOTICE.

## License

MIT, see [LICENSE](LICENSE).
