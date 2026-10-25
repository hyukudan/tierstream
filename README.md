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
  returns an error and poisons the context. The engine must discard that token's outputs. Every layer's weight readers
  must finish before the next layer's export. The end kernel must follow all readers and joined branches.
- **Budgets and statistics:** it uses the existing planner, then checks the gated layout at open. Every weight unit
  must fit in GPU memory or pinned RAM; decode does not fetch weights from disk. The checks include device tables and
  controls, mapped records, staging buffers, and enough spare host blocks. Existing statistics APIs remain available.
  Counters update after successful `ss_end` calls; snapshots expose the last publication. Statistics reset is allowed
  only between tokens.

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
      its own stream, the opposite PCIe direction. With no free block, the miss is served from the ring and the
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
- Pinned RAM to VRAM measured 31-32.5 GB/s on the RTX PRO 6000 with one copy engine (design.md §8).
- NVMe is slower still.

| workload | streaming | why |
|---|---|---|
| **MoE decode** (DeepSeek, MiniMax, Qwen MoE, GLM) | can pay, with a hot set | A token uses only the top-k experts of each layer (6 of 384 for DeepSeek V4.1). When routing has a hot set, as on our DeepSeek traces (95-98 % LRU hits), most lookups need no transfer and a miss costs one expert copy. |
| **Prefill and large batches** (dense or MoE) | can pay | Many tokens reuse every weight byte, so the copy of layer L+1 can hide behind layer L's compute, as long as that compute takes longer than the copy. |
| **Models that do not fit at all** | makes them run | The alternative is not running, or splitting across machines. That is a capacity gain, not a speed gain. Exclusive mode removes disk reads only when VRAM + RAM cover the units. |
| **Dense decode** (one or a few tokens per step) | does **not** pay | Every token reads every weight. Whatever is not resident crosses PCIe on every token, far slower than VRAM, so the streamed fraction dominates the step time. Use a smaller quantization, more GPUs, or a unified-memory machine instead. |
| **MoE decode with near-uniform routing** | pays less | With no hot set, the miss rate approaches the fraction of experts that are not resident. |
| **KV cache, adapters** (later) | not yet | The same tiers fit other large, unevenly used byte ranges. Units today are weights only. |

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
(two Strix Halo nodes on USB4 and a PC with an RTX PRO 6000 96 GB):

- deployable: DeepSeek V4.1 Flash (pair, and the measured RTX + pair hetero placement), MiniMax M3.1 Flash, Qwen3.8
  Flash Next;
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

The HIP backend for Strix Halo (unified memory) is still a stub: it has not been built or run yet. Building and
testing it with the nodes' ROCm toolchain is the next step.

Device-gated decode is opt-in and CUDA only. Its GPU unit test passes on the RTX PRO 6000 (7 cases: graph-captured
decode with prediction, statistics, memory pressure, a stopped service, an out-of-range id, and two protocol
mutations that the test must catch). The bit-for-bit correctness and performance checks against the DeepSeek
replica's own cache are still pending.

## Credits

The design of this library was inspired by [mlx-stream](https://github.com/davidtai/mlx-stream) by David Tai, a
DeepSeek-V4.1-Flash plugin for mlx-serve that streams experts from SSD. Its approach to MoE weight streaming
(scored residency, next-layer router lookahead, unbuffered expert reads into device slots, hits-first gating) is
the starting point for the mechanisms here; thanks to its author for publishing it.

No third-party source code is included: every file in this repository is new code written for this project (see
NOTICE). If a ported file is ever added, it keeps its original license notice in the file and in NOTICE.

## License

MIT, see [LICENSE](LICENSE).
