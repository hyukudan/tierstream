# tierstream

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

It is opt-in and CUDA only, for decode. It requires exclusive placement and LRU eviction, with `pin_planned = 0`.
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
ready read-ahead buffers. That path needs repair before claiming a decode prefetch gain.

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

Device-gated decode is opt-in and CUDA only. Its GPU unit test passes on the RTX PRO 6000 (18 cases: graph-captured
decode with prediction, statistics, memory pressure, global spares with fewer spare blocks than spare slots, a stopped
service, an out-of-range id, sequence desyncs, an absent entry, open-time refusals and seeding, test-hook gating,
budgets, DMA timing, and two protocol mutations that the test must catch). The full-model DeepSeek bit-for-bit
comparison against the replica's own cache and the speed-parity gate both passed. Device-gated decode is merged;
the DeepSeek replica now uses tierstream as its only expert cache. This qualification does not cover unmerged features.

As of 25 October 2026 evening:

| Work | State | Scope / qualification |
|---|---|---|
| Device-gated decode | Merged, opt-in | CUDA decode; full-model bit-for-bit and speed-parity gates passed. |
| Deferred, bounded write-backs (`wbsched`) | In development | Carries pending victims across token boundaries and budgets background submissions so they do not flood the shared copy engine. Preserves exact bytes and slot ownership; GPU tests passed on the RTX PRO 6000. Full-model performance qualification is in progress. |
| Priced window prefetch (`winpf`) | In development, in qualification | Deduplicates multi-row predictions, prices candidates by observed usefulness and copy cost, and bounds speculative traffic behind demand. No accepted full-model speed gain yet. |
| Adaptive budgets, usage tables and NVMe placement (`adaptive`) | In development | Trace-replay measurements above; not an end-to-end engine result. |
| Pipelined cold-start loader (`coldload`) | In development, in progress | Bounded overlap of reads, staging and device copies; no measured saving yet. |

Cold-start loading is still serial. Recent DeepSeek V4.1 Flash RTX runs with the same GGUF expert formats took
92-104 s to initialize, including about 79 s in the serial expert-fill phase in one measured run. Loading precedes
prompt execution; that run then continued from an imported 2,048-token state. These are internal load timers across
different builds/configurations, not a controlled cold/warm-disk comparison or request TTFT. Expert loading includes
allocation, file reads and device copies; their individual times have not been isolated.

## Credits

The design of this library was inspired by [mlx-stream](https://github.com/davidtai/mlx-stream) by David Tai, a
DeepSeek-V4.1-Flash plugin for mlx-serve that streams experts from SSD. Its approach to MoE weight streaming
(scored residency, next-layer router lookahead, unbuffered expert reads into device slots, hits-first gating) is
the starting point for the mechanisms here; thanks to its author for publishing it.

No third-party source code is included: every file in this repository is new code written for this project (see
NOTICE). If a ported file is ever added, it keeps its original license notice in the file and in NOTICE.

## License

MIT, see [LICENSE](LICENSE).
