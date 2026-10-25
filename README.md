# strix-stream

A weight cache for inference engines whose model does not fit in GPU memory. It keeps the weights that matter now
on the GPU, keeps more in page-locked host RAM, and reads the rest from NVMe. It moves weight *units* (an expert, a
layer's blob, a lookup table) between these tiers while the engine runs. Engines read every unit through an address
table, so where a unit lives never changes a single output bit.

- C ABI: `include/strix_stream/ss.h` (prefix `ss_`).
- Design and engine contract: [docs/design.md](docs/design.md). Recipes: [docs/recipes.md](docs/recipes.md).
- Backends: CPU (tests), CUDA (RTX PRO 6000, 4090), HIP for Strix Halo (unified memory; see Status).

## How it works

### Three tiers

```
     engine kernels ── read unit u of layer L at table[L][u] ──┐
                                                               v
  ┌─ 1. GPU memory (VRAM) ──────────────────────────────────────────────────────────┐
  │  resident slots (the plan's best units) │ dynamic slots (adaptive) │ ring        │
  └─────────────────────────────────────────────────────────────────────────────────┘
        ^ promote: copy RAM -> VRAM          │ evict
        │ (a miss, a prefetch)               v   inclusive: drop it (a RAM copy or the file still has it)
        │                                        exclusive: write it back VRAM -> RAM into a free block
  ┌─ 2. pinned host RAM ────────────────────────────────────────────────────────────┐
  │  page-locked blocks: units the GPU does not hold │ read-ahead buffers            │
  └─────────────────────────────────────────────────────────────────────────────────┘
        ^ read at open; read-ahead fills the buffers before the GPU needs the unit
  ┌─ 3. model files on NVMe ────────────────────────────────────────────────────────┐
  │  GGUF / safetensors / packed records, read unbuffered (direct I/O)               │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

On unified-memory machines (Strix Halo), tiers 1 and 2 are the same memory: a resident unit is mapped host memory,
and a miss is read from NVMe straight into its slot.

- **Descriptor.** Lists the model's units: their files, offsets, sizes and parts (for example an expert's gate, up
  and down matrices).
- **Plan.** Assigns each unit a starting tier from a memory budget.
  - The planner fills VRAM with the highest benefit per byte, then pinned RAM, and leaves the rest on disk.
  - By default every unit is worth the same; a measured routing frequency makes the plan smarter.
  - `ss-plan` prints a plan; `ss_plan_from_tiers` takes your own.

### One forward, from the engine's side

For each layer:
1. The engine says which units it needs: `ss_layer_routes` (the router's expert ids), or `ss_layer_all` for a dense
   layer.
2. `ss_acquire` gives it the layer's table once those units are on the GPU.
3. `ss_release` says the kernels are done with them.

Units already resident are free. The others are *misses*, copied from pinned RAM or read from disk into a free slot.
`ss_acquire_groups` lets the engine start on the resident units while the misses are still coming ("hits first").

### Promotion and eviction, in both directions

- **Promotion (RAM -> VRAM, disk -> VRAM).** A miss is copied into a slot. With `pin_planned = false` (decode), the
  plan's residents are only a warm start: the resident set adapts to what the router actually picks.
- **Eviction (VRAM -> RAM, or nowhere).** When a slot is needed, the cache picks a victim:
  - **LRU** (default) evicts the least recently used unit that this forward is not using. On our DeepSeek decode
    traces it hits 95-98 % of lookups, against 84-93 % for a fixed placement of the same size.
  - **SCORED** ranks units by a mix of routing transitions (which experts follow which), recent frequency and age. It
    is opt-in until it beats LRU on real traces.
- **Inclusive vs exclusive** (`tier_mode`) decides what an eviction costs:
  - **Inclusive** (default): a unit promoted from pinned RAM keeps its RAM copy, so eviction just drops the VRAM copy.
    It is simple, but the same unit can occupy VRAM and RAM at once.
  - **Exclusive**: each unit lives in exactly one place: a VRAM slot, a RAM block, or only its file.
    - Promotion frees the unit's RAM block.
    - Eviction writes the unit back from VRAM into a free RAM block (a device-to-host copy on its own stream, the
      opposite PCIe direction).
    - VRAM + RAM then hold the whole model whenever together they are just big enough. The DeepSeek RTX replica
      keeps 80 GiB of experts on the GPU and ~63 GiB in RAM (all 142 GiB), and never reads the disk after start-up.
    - On a GPU with a single copy engine (WDDM), write-backs and promotions share it.

### Fetching ahead

- **Certain prefetch** (`ss_prefetch_layer`): layer-by-layer prefill knows it needs layer L+1 next, so that layer is
  copied while layer L computes (recipe engine knob `prefetch_layers`).
- **Speculative lookahead** (`ss_prefetch` with `SS_PREFETCH_SPECULATIVE`):
  - For MoE decode, the engine runs layer L+1's router on layer L's input, and `ss_lookahead_select` turns the scores
    into the few experts most likely next (`lookahead_units`, default 2).
  - They are read at low priority, only while no demand is waiting.
  - By default they land in the ring and never evict a resident on a guess (`spec_persistent = true` lets them in).
- **Host read-ahead** (`readahead_units`, engine knob `host_prefetch_layers`):
  - Disk is the slowest tier, and a disk read normally cannot start until a GPU slot is free.
  - Read-ahead reads the next layers' disk units into pinned RAM buffers early, so when the slot frees up only the
    fast RAM -> VRAM copy is left.
  - Before it existed, MiniMax RTX prefill at 8K waited ~0.45 s per disk layer.
- **Seed** (`ss_seed`): between requests, load a usage profile's favourite units into free slots.

### What never changes

Bytes are copied, never transformed, and engine kernels must not depend on which slot a unit sits in. So every
placement, policy and prefetch setting gives bit-identical outputs. With `verify_hash`, every read is checked against
the descriptor's hash, and a mismatch is an error, never silently used.

## When it pays (and when it does not)

The question is how many weight bytes each token must move across PCIe, compared with what the GPU reads from its
own memory:
- from pinned RAM, ~50 GB/s;
- from NVMe, a few GB/s;
- from VRAM on an RTX PRO 6000, ~1.8 TB/s.

| workload | streaming | why |
|---|---|---|
| **MoE decode** (DeepSeek, MiniMax, Qwen MoE, GLM) | pays | A token uses only the top-k experts of each layer (6 of 384 for DeepSeek V4.1), so a hot set fits on the GPU. LRU keeps 95-98 % of lookups hits, and a miss costs one expert copy (~0.4 ms for 10 MB). |
| **Prefill and large batches** (dense or MoE) | pays | Thousands of tokens reuse every weight byte, so the copy of layer L+1 hides behind layer L's compute. |
| **Models that do not fit at all** | pays | The alternative is not running, or splitting across machines. Exclusive mode stretches VRAM + RAM to the model's size. |
| **Dense decode** (one or a few tokens per step) | does **not** pay | Every token reads every weight. Whatever is not resident crosses PCIe on every token at ~1/35 of VRAM bandwidth, so the streamed fraction dominates the step time. Use a smaller quantization, more GPUs, or a unified-memory machine instead. |
| **MoE decode with near-uniform routing** | pays less | With no hot set, the miss rate approaches the fraction of experts that are not resident. |

## Recipe knobs at a glance

A recipe (below) compiles to the library's budget and options. The knobs that shape streaming:

| knob | default | what it does |
|---|---|---|
| `device_gib`, `host_pinned_gib`, `workspace_gib` | from the hardware | the VRAM and pinned-RAM budgets, and the engine's own VRAM |
| `units` (`resident`, `pinned`, `rest`) | `fill` | how many units of each group start in VRAM, in RAM, and on disk |
| `pin_planned` | true | true: residents never move (prefill). false: they are a warm start for LRU / SCORED (decode). |
| `dynamic_slots_per_layer` | 0 | extra VRAM slots per layer for the adaptive set |
| `ring_slots` | auto | VRAM slots shared by misses that are not admitted, and by prefetches |
| `policy` | `lru` | `lru` or `scored` eviction |
| `tier_mode`, `host_spare_units` | `inclusive`, auto | inclusive or exclusive; free RAM blocks kept for write-backs |
| `lookahead_units`, `spec_persistent` | 2, false | speculative experts per layer; whether a guess may take a persistent slot |
| `readahead_units`, `host_prefetch_layers` | 0, 0 | host read-ahead buffers and distance |
| `prefetch_layers` | 1 | certain whole-layer prefetch distance |
| `reader_threads`, `staging_mib`, `direct_io` | 2, 256, true | the disk reader |

The full list, with units and checks, is in [docs/recipes.md](docs/recipes.md).

## Build

```
# Linux / WSL
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build && ctest --test-dir build
# Windows (MSVC + Ninja), CPU or CUDA
tools\build-win.bat          &  tools\build-win.bat cuda
```

The tests write small synthetic model files under `SS_TEST_DIR`, or the system temp directory if it is not set. The
CUDA test `ss-test-cuda` needs a GPU. It checksums every streamed unit on the device and probes pinned and NVMe
bandwidth with about 256 MiB each.

`ss-plan DESC.json --device-gib X --host-gib Y` validates a descriptor and prints its tier plan, in the same format
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
model, the devices, their drivers, the descriptors and the probed bandwidths; if any of them changed it is refused
and the recipe runs untuned. A recipe can pin fields the autotuner may not change. `ss-recipe explain` shows, for
every field, its value, which layer set it and what it replaced.

**Exactness is not an option.** Every placement a recipe can express produces bit-identical outputs; a recipe that
asks for "exact" anything is refused.

```
ss-recipe validate RECIPE                 # strict check, prints the effective recipe
ss-recipe explain  RECIPE --set routes.prefill_long.when.prompt_tokens_min=3072
ss-recipe match    RECIPE PROBE.json      # does this hardware fit? which devices go where
ss-recipe select   RECIPE STATE.json      # which route admits this request, and why not the others
ss-recipe plan     RECIPE --route R --device-set S --desc DESC.json   # budget, options and plan-v1
```

The same calls are in the C ABI (`ss_recipe_*`).

**Example recipes** are in [recipes/examples/](recipes/examples/README.md), calibrated on the lab's measured setups
(two Strix Halo nodes on USB4 and a PC with an RTX PRO 6000 96 GB):

- deployable: DeepSeek V4.1 Flash (pair, and the measured RTX + pair hetero placement), MiniMax M3.1 Flash, Qwen3.8
  Flash Next;
- `pending/`: recipes that wait for an engine run, a descriptor or hardware: a DeepSeek RTX decode replica, DeepSeek
  and MiniMax on a 32 GB card, MiniMax with >= 192 GB of host RAM, GLM 5.3 Flash.

Every number in them carries its evidence in `x-` keys (`x-evidence`, `x-cites`, `x-predicted` with a basis:
measured, derived, projected, bound or target). CTest validates each file with `ss-recipe` and checks that the
structural negative fixtures in `recipes/examples/tests/` are refused; `validate_v1.py` adds the citation checks
(it needs the lab notes the `ref`s point into).

## Status

v1. The CPU tests pass on MSVC and on GCC with ASan, UBSan and TSan. The CUDA backend is built and its GPU test runs
in one short window. Exclusive mode and host read-ahead are opt-in.

The HIP backend for Strix Halo (unified memory) is still a stub: it has not been built or run yet. Building and
testing it with the nodes' ROCm toolchain is the next step.

In development: a device-gated decode mode for graph-captured MoE decode. Route -> slot resolution and the residency
check run inside the engine's CUDA graph, and the host is touched only on a miss.

## Credits

The design of this library was inspired by [mlx-stream](https://github.com/davidtai/mlx-stream) by David Tai, a
DeepSeek-V4.1-Flash plugin for mlx-serve that streams experts from SSD. Its approach to MoE weight streaming
(scored residency, next-layer router lookahead, unbuffered expert reads into device slots, hits-first gating) is
the starting point for the mechanisms here; thanks to its author for publishing it.

No third-party source code is included: every file in this repository is new code written for this project (see
NOTICE). If a ported file is ever added, it keeps its original license notice in the file and in NOTICE.

## License

MIT, see [LICENSE](LICENSE).
