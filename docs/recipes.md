# Recipes (`strix-stream/recipe-v1`)

A recipe is a declarative, user-editable file that says how ONE model runs on ONE set of hardware, per workload class:

- **match**: which model and which hardware it is for (device sets, memory, host RAM, disk and link bandwidth);
- **routes**: which device sets serve a request, chosen from the workload (phase, prompt length, batch) and from what
  is available at admission (device busy, link down, low host RAM);
- **placement** per route and device set: which weight units are resident on the device, which come from the pinned
  host tier, which from NVMe, which another set computes; plus the streaming knobs (prefetch, staging ring, chunks);
- **limits and fallbacks**: what to do when a route cannot run.

The library compiles a recipe into the inputs it already has: an `ss_budget`, `ss_options` and a
`strix-stream/plan-v1` placement (the JSON `ss_plan_from_json` reads), plus the resources each member needs.

**Exactness is not a recipe option.** Every placement is bit-identical by construction (design.md §4). A key named
`exact`, `exactness` or `bit_exact` anywhere is refused with that message.

## 1. Files, formats, numbers

- TOML 1.0 or JSON. The reader takes JSON when the first non-blank character is `{`, else TOML. Both give the same
  document tree; examples here are TOML. UTF-8, at most 16 MiB.
- TOML subset: tables, arrays of tables, dotted and quoted keys, inline tables, arrays, basic and literal strings
  (single and multi-line), integers (decimal, `0x`/`0o`/`0b`, `_` separators), floats, booleans. Dates/times, `inf`
  and `nan` are refused.
- **Strict:** every unknown key is an error naming its path (`routes.prefill_long.placement.rtx.units.layers:
  unknown key "residnet"`), except keys starting with `x-`, which are free-form notes (kept, never read; cite the
  measurement behind a threshold there). Wrong types, out-of-range values and broken references are errors.
- Error paths name array elements by their `id` (`routes.<id>`, `match.devices.<id>`, `...units.<id>`), the same
  form as override paths (§6).
- Integers are integers (`4096`, not `4096.0`). Float fields accept integers. Every number must be finite.

### Units

| suffix | unit | conversion |
|---|---|---|
| `_gib` | GiB = 2^30 bytes, float in [0, 1048576] | bytes = floor(value × 2^30 + 0.5) (exact in double over this range) |
| `_mib` | MiB = 2^20 bytes, integer | bytes = value × 2^20 |
| `_gbps` | link bandwidth, Gb/s = 10^9 bit/s, float ≥ 0 | |
| `_gbytes_ps` | disk or copy bandwidth, GB/s = 10^9 bytes/s, float ≥ 0 | |
| `_tokens` | tokens, integer | |

All byte arithmetic is checked 64-bit; an overflow is a refusal, never a wrap.

## 2. Top level

```toml
schema = "strix-stream/recipe-v1"
name = "minimax-m3.1-flash-lab"
description = "MiniMax M3.1 Flash: pair TP2 decode, RTX PRO prefill from 4K"
generated_by = "human"
pin = ["routes.prefill_long.when.prompt_tokens_min"]
```

| field | type | default | meaning |
|---|---|---|---|
| `schema` | `"strix-stream/recipe-v1"` | required | |
| `name` | string `[A-Za-z0-9._-]{1,64}` | required | profiles and overrides name the recipe they apply to |
| `description` | string | `""` | |
| `generated_by` | `"human"` \| `"autotuner"` | `"human"` | provenance only |
| `pin` | array of distinct paths (§6.2) | `[]` | fields the autotuner may not change: a profile value at a pinned path, or under a pinned prefix, is ignored and explain says so. Overrides still apply. Each entry is a settable path or a segment prefix of one (`routes.prefill_long` pins the whole route). |
| `match` | table | required | §3 |
| `limits` | table | defaults | §4 |
| `routes` | array of tables | required | §5 |

## 3. `[match]`: model and hardware

```toml
[match]
model = "minimax-m3.1-flash"

[[match.devices]]
id = "rtx"
kind = "cuda"
name = "RTX PRO 6000"
memory_min_gib = 94.0
host_ram_min_gib = 120.0
disk_read_gbytes_ps_min = 5.0

[[match.devices]]
id = "pair"
kind = "hip"
count = 2
memory_min_gib = 110.0
internal_link = { kind = "usb4", gbps_min = 30.0 }

[[match.links]]
id = "pc-nodes"
between = ["rtx", "pair"]
kind = "ethernet"
gbps_min = 1.0
```

| field | type | default | meaning |
|---|---|---|---|
| `model` | string, non-empty | required | equals the probed model id and each descriptor's `model` |
| `model_sha256` | 64 lowercase hex | absent | when set, the probed model sha256 must equal it |
| `engine` | string | absent | when set, equals the probed engine id and each descriptor's `engine` |
| `devices` | array of tables, ≥ 1 | required | device sets, below |
| `links` | array of tables | `[]` | links between device sets, below |

`match.devices[]`: a device set = one device, or `count` identical devices that run as one unit (the two Strix nodes
of a TP pair). Its members are numbered 0..count-1 in binding order (§3.1).

| field | type | default | meaning |
|---|---|---|---|
| `id` | string `[a-z0-9_-]{1,32}`, unique among sets | required | the name routes and placements use |
| `kind` | `"cuda"` \| `"hip"` \| `"cpu"` | required | backend of its placements. `hip` and `cpu` are unified memory: device bytes are host RAM (§7.3). |
| `count` | integer 1..64 | 1 | members |
| `name` | string | absent | case-insensitive substring of the probed device name |
| `memory_min_gib` | float > 0 | required | device memory per member (VRAM, or GPU-mappable memory on unified devices) |
| `host_ram_min_gib` | float ≥ 0 | 0 | total RAM of each member's host |
| `disk_read_gbytes_ps_min` | float ≥ 0 | 0 | probed sequential read bandwidth of the disk each member streams from |
| `h2d_gbytes_ps_min` | float ≥ 0 | 0 | `cuda` sets only: probed pinned host-to-device copy bandwidth (e.g. thresholds that assume PCIe 5 x16 must not bind to a x4 slot) |
| `internal_link` | `{kind, gbps_min}` | absent | required by the hardware check when the members land on ≥ 2 hosts: a probed link of that kind and bandwidth between every two member hosts |

`match.links[]`:

| field | type | default | meaning |
|---|---|---|---|
| `id` | string `[a-z0-9_-]{1,32}`, unique | required | |
| `between` | array of 2 distinct set ids | required | |
| `kind` | `"usb4"` \| `"ethernet"` \| `"pcie"` \| `"other"` | `"other"` | `other` accepts any probed kind |
| `gbps_min` | float ≥ 0 | 0 | |

A link is satisfied when every host of one set has a probed link of that kind and bandwidth to every host of the
other set (hosts shared by both sets need none).

### 3.1 Hardware check and binding (`ss_recipe_match`)

Input: a probe report (§8.2) with one entry per physical device carrying a stable `id` (GPU UUID or PCI address; a
Strix node's id), and links between hosts. The check:

1. Model: `match.model` (and `model_sha256`, `engine` when set) equal the probe's.
2. Binding: each set needs `count` distinct probed devices that pass its kind, name, memory, host-RAM and disk
   minimums; no device serves two sets; and the bound hosts must satisfy every `match.links` entry and every needed
   `internal_link`. The binding is the first valid assignment of a deterministic backtracking search (sets and
   members in recipe order, candidates in ascending probe `id`; links checked as soon as both ends are bound), so it
   finds a topology-valid assignment whenever one exists and the same report always gives the same binding. The
   search is bounded (10^6 steps; beyond that it is refused as too large).

Success returns `strix-stream/recipe-match-v1` (§8.2): the binding (set → member → probe id, host, ordinal). A
failure is refused with every unmet requirement, e.g. `match.devices.rtx: needs 1 cuda device named "RTX PRO 6000"
with >= 94 GiB; eligible: 0 of 2 probed (NVIDIA GeForce RTX 4090 23.6 GiB: memory)`.

## 4. `[limits]`

| field | type | default | meaning |
|---|---|---|---|
| `host_free_ram_floor_gib` | float ≥ 0 | 8.0 | admission keeps at least this much free RAM on every host a route uses (§7.3) |
| `host_free_commit_floor_gib` | float ≥ 0 | 0 | when > 0, admission also keeps this much free commit on every host a route uses (Windows bills discrete-device allocations to commit; §7.3). 0 = commit is not checked. |

## 5. `[[routes]]`

```toml
[[routes]]
id = "prefill_long"
phase = "prefill"
devices = ["rtx"]
execute = "single"
handoff_route = "decode_pair"
links = ["pc-nodes"]
fallback = "prefill_nodes"
when = { prompt_tokens_min = 4096 }

[[routes]]
id = "prefill_nodes"
phase = "prefill"
devices = ["pair"]
execute = "tp"
handoff_route = "decode_pair"

[[routes]]
id = "decode_pair"
phase = "decode"
devices = ["pair"]
execute = "tp"
```

| field | type | default | meaning |
|---|---|---|---|
| `id` | string `[a-z0-9_-]{1,32}`, unique | required | |
| `phase` | `"prefill"` \| `"decode"` | required | |
| `devices` | array of set ids, ≥ 1, distinct | required | sets that run the route; in a `hetero` route also the ownership order (§5.3) |
| `execute` | `"single"` \| `"tp"` \| `"replicas"` \| `"hetero"` | required | `single`: one set with count 1. `tp`: one set with count ≥ 2 (one model split over its members). `replicas`: one set with count ≥ 2, a full copy per member. `hetero`: ≥ 2 sets compute disjoint units of one model (§5.3). |
| `handoff_route` | id of a `decode` route | absent | `prefill` only: the state is exported to that route's sets, which decode. Admission checks both routes (§7). |
| `links` | array of link ids | `[]` | links the route needs. Required to connect (as a graph) all of the route's sets plus the handoff route's sets. Admission needs each one up. |
| `when` | table | `{}` (always) | workload thresholds, inclusive, all must hold |
| `when.prompt_tokens_min` | integer ≥ 0 | 0 | |
| `when.prompt_tokens_max` | integer ≥ `prompt_tokens_min` | no limit | |
| `when.batch_min` | integer ≥ 1 | 1 | sequences in the step |
| `when.batch_max` | integer ≥ `batch_min` | no limit | |
| `fallback` | route id, same phase, not itself | absent | taken when this route matches but cannot be admitted (§7). Chains must not cycle. |
| `enabled` | bool | true | false: the route admits no request (e.g. shipped off until its quality receipt exists); an override turns it on |
| `placement` | table keyed by set id | `{}` | §5.1; keys must be in `devices` |

Cross-field: the recipe has at least one `decode` route; every phase that has routes ends (in file order) with a
catch-all (empty `when`); a route's sets plus its handoff route's sets are connected by its `links`; every set of
every route has a placement.

### 5.1 `routes.<id>.placement.<set>`: budget and knobs

One table per set of the route (required: the placement carries the context bound and the workspace, which are never
assumed). Every member of the set gets the same placement, each planned against its own descriptor (its TP shard;
§8.1 `member`). A placement without unit groups is planned by `ss_plan_make` (not allowed in `hetero` routes).

```toml
[routes.placement.rtx]
device_gib = 92.0
host_pinned_gib = 48.0
max_context_tokens = 32768
workspace_gib = 6.0
chunk_tokens = 4096
staging_buffers = 4
staging_mib = 256
```

Budget (`ss_budget`):

| field | type | default | ss field / meaning |
|---|---|---|---|
| `device_gib` | float > 0, ≤ the set's `memory_min_gib` | `memory_min_gib` | `device_bytes` |
| `host_pinned_gib` | float ≥ 0, ≤ `host_ram_min_gib` when that is > 0 | 0 | `host_pinned_bytes` |
| `workspace_gib` | float ≥ 0, < `device_gib` | derived, see below | `workspace_bytes` |
| `max_context_tokens` | integer 1..2^40 | required | positions of state the engine allocates per sequence on this route. Admission refuses a request whose context (prompt + generation budget) is larger (§7.2). |
| `max_batch` | integer 1..65536 | required | most sequences one forward on this route runs. Admission refuses a larger batch (§7.2). |
| `margin_gib` | float 0..64 | 2.0 | `margin_gib` |
| `margin_fraction` | float 0..0.5 | 0.03 | `margin_fraction` |
| `dynamic_slots_per_layer` | integer 0..65536 | 0 | `dynamic_slots_per_layer` |
| `ring_slots` | integer 0..65536 | 0 | `ring_slots` (0 = the descriptor's, else 2 × most units of a layer of that size) |
| `pinned_backing` | bool | false | `pinned_backing` (§5.2 step 6) |
| `host_spare_units` | integer 0..65536 | 0 | `host_spare_units` (exclusive tier mode: free pinned blocks wanted per unit size, from the host bytes left after the pinned tier; 0 = auto, min(64, units of that size)) |

**Forward rows** (the most rows one forward runs, across the whole batch): prefill = `max_batch` × (`chunk_tokens` if
set, else `min(when.prompt_tokens_max, max_context_tokens)`, an unset max meaning `max_context_tokens`); decode =
`max_batch`. They size the derived workspace and the route-id buffers (§7.3).

`workspace_gib` absent: derived from the descriptor as `workspace_bytes_per_row × forward rows +
state_bytes_per_position × max_context_tokens × max_batch`. When the descriptor gives 0 for either size, the mapping
is refused: "set workspace_gib". Zero is never assumed. On a `cuda` set the compiled `workspace_bytes` also includes
the device copies of the unit tables (2 × 8 bytes per unit), so the planner's cap check covers every device
allocation.

Streaming knobs (`ss_options`; `backend` comes from the set's `kind`; `device` is the caller's, §8.1):

| field | type | default | ss field |
|---|---|---|---|
| `policy` | `"lru"` \| `"scored"` | `"lru"` | `policy` |
| `lookahead_units` | integer 0..64 | 2 | `lookahead_budget` (speculative units per layer) |
| `reader_threads` | integer 1..64 | 2 | `reader_threads` |
| `staging_buffers` | integer 2..64 | 4 | `staging_buffers` |
| `staging_mib` | integer 1..16384 | 256 | `staging_bytes` (≥ 4 × the largest source alignment; rounded up to it by the library; a unit larger than a buffer is read in pieces) |
| `copy_piece_mib` | integer 1..4096 | 64 | `copy_piece_bytes` |
| `score_window` | integer 1..4096 | 16 | `score_window` |
| `pin_planned` | bool | true | `pin_planned` |
| `fill_residents` | bool | true | `fill_residents` |
| `direct_io` | bool | true | `direct_io` |
| `verify_hash` | bool | false | `verify_hash` |
| `group_units` | integer 0..65536 | 0 | `group_units` |
| `spec_persistent` | bool | false | `spec_persistent` |
| `tier_mode` | `"inclusive"` \| `"exclusive"` | `"inclusive"` | `tier_mode` of the budget. inclusive: a promoted unit keeps its pinned copy. exclusive: each unit lives in exactly one place (a device slot, a pinned block, or only its file); a promotion frees its block and an evicted unit is written back into a free block (design.md §3.4). Exclusive is refused on a `hip` set, with `fill_residents = false` and with `pinned_backing = true`. Pick it for a decode route with dynamic residency (`pin_planned = false`) whose device + host bytes cover the model only when no unit is held twice. |

Engine knobs (in the compiled plan's `engine` section; the library does not read them; every value has one concrete
meaning, and an engine refuses a value outside its legal set rather than substituting its own):

| field | type | default | meaning |
|---|---|---|---|
| `prefetch_layers` | integer 0..8 | 1 | whole-layer prefetch distance (`ss_prefetch_layer` of L+k during L); 0 = none |
| `chunk_tokens` | integer 0..1048576 | 0 | prefill chunk rows; 0 = the whole prompt is one chunk |
| `routed_only_max_tokens` | integer 0..1048576 | 0 | prefill prompts up to this many tokens upload only the routed units of a streamed layer, longer ones the whole layer; 0 = always the whole layer (decode is always routed-only) |
| `idle_preload_layers` | integer 0..64 | 0 | streamed layers a resident service loads ahead while idle between requests; 0 = none |
| `max_copies_per_token` | integer 0..65536 | 0 | most units copied into dynamic slots per decode token; 0 = no limit |

Mapping checks (need the descriptor): every stream ring has at least (`prefetch_layers` + 1) × (the most streamed
units of one layer of that ring's unit size) slots, else refused naming `ring_slots`.

### 5.2 `routes.<id>.placement.<set>.units[]`: unit groups and tiering

A group says how the units of one class, in some layers, are ranked and tiered. Units are the descriptor's units:
`moe` = routed experts of each MoE layer, `blob` = whole streamable layer blobs. Non-streamable units are always
resident. When a placement has groups, every streamable unit of the member's descriptor must be in exactly one group.

| field | type | default | meaning |
|---|---|---|---|
| `id` | string `[a-z0-9_-]{1,32}`, unique in the placement | required | |
| `class` | `"moe"` \| `"blob"` | required | |
| `layers` | array of layer indices and `[first, last]` inclusive ranges, no overlap | every layer of `class` | each must be a streamable layer of `class` in the descriptor |
| `rank` | `"model"` \| `"list"` \| `"frequency"` | `"model"` | unit order, below |
| `list` | array of distinct integers ≥ 0 | required iff `rank = "list"` | moe: expert ids (< `n_units` of every layer of the group); blob: layer indices of the group |
| `resident` | integer ≥ 0 or `"fill"` | `"fill"` | moe: per layer; blob: in the whole group |
| `pinned` | integer ≥ 0 or `"fill"` | 0 | next units in rank order: pinned host tier |
| `rest` | `"disk"` \| `"pinned"` \| `"peer"` | `"disk"` | the units left after `resident` and `pinned` |
| `peer` | set id | required iff `rest = "peer"` | the set that computes them (`hetero` routes only; a set after this one in `devices`) |

**Rank order** (total, deterministic). Keys compared in order, ties broken by the next key:
- `model`: (layer, unit).
- `list`: (position in `list`, layer); units not in `list` follow in (layer, unit).
- `frequency`: benefit per unit stride descending, then (layer, unit). Benefit is the caller's array (global unit
  index, `ss_desc_unit_index`), else the descriptor's `benefit`; refused unless one of them gives a value for every
  unit of the group (no unit is ranked on an invented default).
For integer `moe` counts the order is applied per layer; for fills and `blob` counts over the whole group.

**Integer counts never shrink.** For `moe`, `resident + pinned` (integers) must be ≤ `n_units` of every layer in the
group; for `blob`, ≤ the group's layers. Otherwise refused. `rest = "pinned"` with `pinned = "fill"` is refused (one
of them must bound the tier).

**Tiering of one member** (steps in this order; "assigned" units are not touched by later steps):

1. Integer `resident` counts, every group: the first N units in rank order → resident.
2. Integer `pinned` counts, every group: the next N → pinned.
3. Resident fill. Candidates = unassigned units of the `resident = "fill"` groups, concatenated in group file order,
   each group in rank order. With the planner's own arithmetic (design.md §5: room = cap − margin − fixed −
   workspace − dynamic-slot bound; unit stride; one ring per streamed unit size):
   - if the residents so far + every candidate + the rings of the units already not resident fit in room, every
     candidate is resident;
   - else the rings of every unit size not yet resident are reserved and candidates are taken in order, each one that
     still fits (a candidate that does not fit is skipped, the walk continues);
   - on a descriptor with `order: prefix` (every group's rank must be `model`), instead the candidates of all fill
     groups are ordered globally by (layer, unit), and the largest k is chosen such that (a) the residents of step 1
     plus the first k candidates are exactly a model-order prefix of the descriptor's streamable units (every unit
     before the cut resident, none after it), and (b) they plus the rings of exactly the units that would then not
     be resident fit in room; k is tried from all candidates downwards (the `ss_plan_make` prefix search, restricted
     to cuts compatible with the explicit assignments). No valid k: refused, naming the explicit assignment that
     breaks the prefix.
4. Pinned fill. Candidates = unassigned units of the `pinned = "fill"` groups (same order); each is pinned while its
   4 KiB host stride fits in `host_pinned_gib` − the descriptor's `host_fixed` − the pinned units so far; one that
   does not fit is skipped.
5. Every unit left takes its group's `rest`.
6. `pinned_backing = true`: residents get a pinned host copy (tier `RESIDENT_BACKED`) with what is left of the host
   budget, lowest benefit per byte first, stopping at the first that does not fit (as `ss_plan_make`).
7. Checks, each a refusal naming the unit: the descriptor's `tiers` allow every non-resident storage tier used
   (`pinned` needs `pinned` or `pinned|disk`; `disk` needs `disk` or `pinned|disk`); on `order: prefix` the residents
   are a model-order prefix; then `ss_plan_from_tiers` re-checks every total with the planner's exact billing and
   refuses with its message (`SS_E_CAPACITY`) if the placement does not fit. Nothing is silently dropped.

A placement without groups is `ss_plan_make(desc, budget, benefit)`.

### 5.3 Heterogeneous routes (`execute = "hetero"`)

Several sets compute disjoint units of one model (e.g. dense + hot experts on the RTX, the rest on the Strix pair).
Ownership is compiled ONCE per route from reference inputs: for every set of the route, the descriptor of its member 0
and its benefit array (§8.1 `ref_*`). It is decided in the route's `devices` order:
- every unit starts open; set i tiers (steps 1-7) exactly the units that are open or were sent to i; units its groups
  tier resident / pinned / disk become owned by i; units its groups send to `rest = "peer"` are sent to that later
  set; units sent to another set are `PAIR` for i (computed elsewhere, never fetched);
- integer counts are checked against the units available to the set in each layer (after the masking);
- at the end every unit is owned by exactly one set. Refused otherwise, naming the unit.

The result is a frozen ownership map (one owner per (layer, unit)); its sha256 is in every compiled plan of the route
(`ownership_sha256`), so the members of all sets can check they planned against the same map. Planning one member
then tiers only the units its set owns (with the member's own descriptor; units owned elsewhere are `PAIR`). If a
member's own descriptor cannot hold a unit its set owns (a `rest = "peer"` would be needed), it is refused.

All reference descriptors must have the same layer structure (layer count, kinds, `n_units`, `streamable`); unit
identity = (layer, unit). Members of a TP set index units the same way (TP splits bytes inside a unit). Dense parts
(attention, shared expert, head) are fixed bytes of each side's descriptor: the adapter emits a descriptor per side
with the fixed bytes that side computes. Splitting the experts of one layer across the members of one set (EP) is the
engine's internal layout and not a recipe placement in v1. `PAIR` units take no stream ring (the library never
fetches them).

## 6. Profiles, overrides and resolution

### 6.1 Layers and precedence

Highest first: **overrides** (files and `--set`, applied in order, a later one wins) > **profile** (the autotuner's;
at most one; ignored at pinned paths) > **recipe** > **default** (this document).

```toml
schema = "strix-stream/recipe-profile-v1"
recipe = "minimax-m3.1-flash-lab"
recipe_sha256 = "<sha256 of the recipe file bytes>"
source = "hyperloom tune, PRO, prefill-8k, 2026-10-24"
workload = "prefill-8k"

[key]
engine_sha256 = "<engine binary sha256>"
model_sha256 = "<model sha256>"
devices = { rtx = ["GPU-3f2a…"], pair = ["node1", "node2"] }
drivers = { rtx = "580.88", pair = "6.4.3" }
descriptors = { rtx = ["<sha256>"], pair = ["<sha256>", "<sha256>"] }

[probe.rtx]
disk_read_gbytes_ps = 6.8
h2d_gbytes_ps = 25.1

[probe.pair]
disk_read_gbytes_ps = 5.9

[set]
"routes.prefill_long.placement.rtx.units.layers.resident" = 33
"routes.prefill_long.when.prompt_tokens_min" = 3072
```

Profile (`strix-stream/recipe-profile-v1`):

| field | type | default | meaning |
|---|---|---|---|
| `recipe` | string | required | must equal the recipe's `name` |
| `recipe_sha256` | 64 lowercase hex | required | must equal the sha256 of the recipe's bytes as loaded |
| `workload` | string, non-empty | required | the workload class it was tuned for; must equal the identity's `workload` |
| `source` | string | `""` | shown by explain |
| `key.engine_sha256`, `key.model_sha256` | 64 lowercase hex | required | must equal the identity's |
| `key.devices` | table: every set → array of probe ids | required | must equal the binding |
| `key.drivers` | table: every `cuda` / `hip` set → string | required | must equal the probed driver of every member |
| `key.descriptors` | table: every set → array of descriptor sha256, one per member | required | must equal the identity's `descriptors` |
| `probe.<set>.disk_read_gbytes_ps` | float > 0 | required for every set | within `max_drift` of the current probe (min over members) |
| `probe.<set>.h2d_gbytes_ps` | float > 0 | required for every `cuda` set | same |
| `max_drift` | float 0..1 | 0.10 | |
| `set` | table path → value | required | |

A profile is applied against the current **identity**: the probe report plus `workload` and `descriptors` (§8.2;
`ss_recipe_add_profile` takes it). A profile that fails any check is **stale**: it is refused with the reason
(`profile stale: key.drivers.rtx is 580.88, probed 581.02`), the effective recipe stays as it was, and the
orchestrator runs with the recipe alone and marks the run "untuned" (tuner-design.md §5). A new probe, binding,
descriptor or workload class means building the stack again (the old one keeps the identity it was checked with).

A profile is tuned for the budgets it was measured with. Once a profile is applied, an override that changes a budget
field (`device_gib`, `host_pinned_gib`, `workspace_gib`, `max_context_tokens`, `max_batch`, `margin_gib`,
`margin_fraction`, `dynamic_slots_per_layer`, `ring_slots`, `pinned_backing`, `host_spare_units`, `staging_buffers`,
`staging_mib`, `tier_mode`) of a placement the
profile sets anything in, or any `match.*` field, is refused ("drop the profile or retune"); the same holds when the
profile is added after such an override.

Overrides (`strix-stream/recipe-overrides-v1`): `recipe` (required, must equal the name), `recipe_sha256` (optional,
checked when set), `source`, `set`. `ss_recipe_set(path, value)` is a one-entry overrides layer whose value is TOML
value text (`33`, `"fill"`, `[[0, 3], 7]`).

Each layer is applied atomically: the whole effective recipe is rebuilt and validated with the layer; on any error the
layer is refused and the effective recipe does not change. Fields that depend on each other change in one layer: a
group's `rank = "list"` and its `list` go in one overrides file (a lone `--set` of either is refused).

### 6.2 Settable paths

Only these leaf fields can be set (values must be the field's type; a path to a table is refused, so pins cannot be
bypassed by replacing a parent):

- `limits.host_free_ram_floor_gib`, `limits.host_free_commit_floor_gib`
- `match.devices.<set>.` `name`, `memory_min_gib`, `host_ram_min_gib`, `disk_read_gbytes_ps_min`, `h2d_gbytes_ps_min`
- `match.links.<link>.gbps_min`
- `routes.<route>.fallback`, `routes.<route>.enabled`
- `routes.<route>.when.` `prompt_tokens_min`, `prompt_tokens_max`, `batch_min`, `batch_max`
- `routes.<route>.placement.<set>.` every field of §5.1 (a placement the recipe does not have is created for that set
  when the set is in the route's `devices`)
- `routes.<route>.placement.<set>.units.<group>.` `layers`, `rank`, `list`, `resident`, `pinned`

Structure belongs to the recipe and is not settable: `schema`, `name`, `pin`, every `id`, `match.model`,
`model_sha256`, `engine`, a set's `kind`, `count`, `internal_link`, a link's `between` and `kind`, a route's `phase`,
`devices`, `execute`, `handoff_route`, `links`, and a group's `class`, `rest`, `peer`.

### 6.3 Explain

`ss_recipe_explain` returns `strix-stream/recipe-explain-v1`: every leaf of the effective recipe (defaults included)
with its value, the layer it came from (`default`, `recipe`, `profile`, `override`), that layer's origin (file name
or `--set`), and every value it shadowed with the reason (`overridden`, `pinned by recipe: <pin entry>`).

## 7. Route selection (admission time)

`ss_recipe_select` decides the route of a NEW request (admission). A request keeps its route for its whole life; a
running request is never moved (state migration is not defined in v1). A decode step of an existing session passes
`session_route`: that route is returned if it can still be admitted, else the call is refused (the orchestrator ends
or restarts the request; no automatic fallback).

### 7.1 Algorithm

1. Walk the routes of the phase in file order. A route **matches** when its `when` holds.
2. A matched route is **admissible** when (§7.2) all of its sets are available, its links are up and its RAM fits; a
   prefill route with `handoff_route` also needs that decode route admissible (its sets, links and RAM, counted
   together with the prefill route's on shared hosts).
3. Matched but not admissible: follow its `fallback` chain. A fallback target is used only if its own `when` holds and
   it is admissible; a target already visited ends the chain. When the chain ends without a route, the walk resumes
   after the originally matched route. Visited routes are never visited again.
4. No admissible route: refused, listing for each visited route why it was not admitted.

The result `strix-stream/recipe-select-v1` names the route, its handoff route, the per-host RAM it was charged, and the
trace (every route visited: `no match` / `admitted` / why not).

### 7.2 Availability and bounds

From the request state (§8.3):
- a set is available when every one of its members' devices is `available` (false when busy, e.g. the PC-BIG
  ledger taken, or down);
- a link is up when `up`; a set with an `internal_link` also needs the link `<set>.internal` up;
- a route whose compiled plan reads from disk (`disk_bytes` > 0) needs `disk_available: true` on each such member's
  device (false when the disk is busy, e.g. over its duty limit);
- a disabled route (`enabled = false`) is not admitted, as a match or as a fallback target;
- the request's `context_tokens` (prompt + generation budget) must be ≤ `max_context_tokens`, and its `batch` ≤
  `max_batch`, of every placement of the route and of its handoff route; the handoff route's `when` must hold for
  the same request too.
A device, link or host missing from the state is unavailable / down / unknown RAM, which makes every route that needs
it inadmissible.

### 7.3 RAM admission

Admission uses **compiled requirements**: for every member of the route (and of its handoff route), the
`resources.host_bytes` of its `ss_recipe_plan` output, which counts everything the library allocates in host memory
for that member:
- the pinned tier (the plan's pinned bytes, with `host_fixed`);
- the staging ring: `staging_buffers` × `staging_mib` rounded up to the largest source alignment (pinned);
- per layer with n units: the two unit tables, 2 × 8 × n bytes (pinned), and the route-id buffer, 4 × max(1024,
  forward rows × top_k) bytes (pinned, routed layers);
- the library's metadata, bounded from allocation capacities: 64 KiB + 16 bytes per unit + 128 bytes per stream-ring
  slot (slot record, owner pointer) + per layer with n units 4096 + 768 × n bytes (per-unit state, admission vectors at
  twice their size for capacity growth, in-flight read jobs), plus, with `policy = "scored"`, 4 × n² bytes (the
  transition matrix) + (score_window + 1) × (64 + 8 × n) bytes (the history window, its transient extra entry and
  container overhead);
- on unified sets (`hip`, `cpu`), the device bytes too: the plan's peak + margin.

**Plan identity.** Each compiled plan has a `plan_id` = sha256 of its inputs: the effective recipe hash, route, set,
member, the member's bound probe `id`, the canonical descriptor hash (`descriptor_sha256` = sha256 of
`ss_desc_to_json`), the benefit hash, and the ownership hash (hetero). The state's `requirements` lists, per member,
`{route, set, member, id, effective_sha256, descriptor_sha256, ownership_sha256, plan_id, host_bytes}`. A requirement
counts only when its `effective_sha256` is the current one, its `id` is the member's bound device, and its
`descriptor_sha256` equals the state's `descriptors[set][member]` (the descriptors in use). A route is inadmissible
("not compiled for this recipe") unless every member of it and of its handoff route has exactly one counting
requirement, and, for a hetero route, all its members carry the same `ownership_sha256`. Requirements are summed per
host (from the binding).

A route in `running` (`{route, effective_sha256, plan_ids}`; its service is up and its memory is already counted as
used) charges 0 when its `effective_sha256` is the current one and its `plan_ids` are exactly the `plan_id`s of the
route's counting requirements; otherwise it gets no credit. A credited route's hosts are still checked: each must
report its free RAM (and commit, when that floor is set) and stay at or above the floors. A route is admissible when on every host it charges,
`free_ram_gib − charge ≥ host_free_ram_floor_gib`, and, when `host_free_commit_floor_gib` > 0, `free_commit_gib −
commit charge ≥ host_free_commit_floor_gib` (the commit charge is `commit_bytes`: host bytes plus, on a discrete
device, its device bytes). Select is a pure decision: the orchestrator reserves what it admits (atomically, under its
own lock) before starting work.

## 8. API

### 8.1 C ABI (`ss.h`)

```c
#define SS_E_RECIPE 9   /* ss_status: recipe parse / validation / stale layer / hardware refusal / no route */
typedef struct ss_recipe ss_recipe;
ss_status ss_recipe_from_text(const char* text, size_t len, const char* origin, ss_recipe** out);
ss_status ss_recipe_from_file(const char* path, ss_recipe** out);
void      ss_recipe_free(ss_recipe* r);
ss_status ss_recipe_add_profile(ss_recipe* r, const char* text, size_t len, const char* origin,
                                const char* identity_json, size_t identity_len);
ss_status ss_recipe_add_overrides(ss_recipe* r, const char* text, size_t len, const char* origin);
ss_status ss_recipe_set(ss_recipe* r, const char* path, const char* value, const char* origin);
ss_status ss_recipe_to_json(const ss_recipe* r, char** out, size_t* len);   /* effective recipe, canonical */
ss_status ss_recipe_explain(const ss_recipe* r, char** out, size_t* len);
ss_status ss_recipe_match(const ss_recipe* r, const char* probe_json, size_t len, char** out, size_t* olen);
ss_status ss_recipe_select(const ss_recipe* r, const char* state_json, size_t len, char** out, size_t* olen);
ss_status ss_recipe_plan(const ss_recipe* r, const char* route, const char* set, int member,
                         const char* device_id, const ss_desc* desc, const double* benefit, size_t n_benefit,
                         const ss_desc* const* ref_descs, const double* const* ref_benefits,
                         const size_t* ref_n_benefits, size_t n_refs,
                         ss_budget* budget, ss_options* options, ss_plan** plan, char** out, size_t* olen);
```

- Every `char** out` is NUL-terminated, allocated by the library and freed with `ss_free`; a length pointer may be
  NULL. On failure nothing is written to any output and `ss_last_error` has the message.
- `ss_recipe_from_file` uses the file name as origin; `ss_recipe_from_text` with a NULL origin uses `"<text>"`.
- `ss_recipe_plan` compiles one member: `desc` is that member's descriptor, `benefit` per global unit of `desc`
  (`ss_desc_unit_index`; NULL = the descriptor's). `ref_*` are indexed like the route's `devices` (member 0 of each
  set, with its benefit or NULL): required in `hetero` routes (ownership, §5.3), else ignored (pass NULL / 0).
  `budget`, `options`: initialize with `ss_budget_defaults` / `ss_options_defaults` (they set struct_size); the
  library writes min(struct_size, its own size) bytes and sets every recipe-controlled field; `options->device`
  keeps the caller's value (the ordinal from the binding). `plan` (NULL to skip) is the object `ss_plan_from_json`
  makes from the returned plan-v1 with this budget, freed with `ss_plan_free`.
- `out` of `ss_recipe_plan` is `strix-stream/recipe-plan-v1`: `{recipe, effective_sha256, route, set, member,
  descriptor_sha256, benefit_sha256, ownership_sha256 (hetero), plan_id, budget, options, engine, resources:
  {device_bytes, host_pinned_bytes, staging_bytes, tables_bytes, metadata_bytes, host_bytes, commit_bytes,
  disk_bytes}, plan, trace}`. `device_id` is
  the member's bound probe `id` (from the match binding); it enters `plan_id`. (budget/options/engine as recipe field names with byte
  values; `trace` = one line per tiering step).

### 8.2 Probe report / identity

```json
{"model": "minimax-m3.1-flash", "model_sha256": "…", "engine": "…", "engine_sha256": "…",
 "workload": "prefill-8k",
 "descriptors": {"rtx": ["<sha256 of member 0's descriptor>"], "pair": ["…", "…"]},
 "devices": [{"id": "GPU-3f2a…", "kind": "cuda", "name": "NVIDIA RTX PRO 6000 Blackwell", "memory_gib": 95.6,
              "host": "pc", "host_ram_gib": 127.7, "disk_read_gbytes_ps": 6.8, "h2d_gbytes_ps": 25.1,
              "ordinal": 0, "driver": "580.88"}],
 "links": [{"kind": "ethernet", "hosts": ["pc", "node1"], "gbps": 2.5}]}
```

Required per device: `id` (stable: GPU UUID, PCI address, node id), `kind`, `name`, `memory_gib`, `host`; the others
are optional (a minimum or profile check on an absent value fails). `workload` and `descriptors` are needed only to
apply a profile. Match result `strix-stream/recipe-match-v1`: `{recipe, effective_sha256, binding: {set: [{id, host,
ordinal}]}}`.

### 8.3 Request state (input of select)

```json
{"phase": "prefill", "prompt_tokens": 8192, "context_tokens": 12288, "batch": 1, "session_route": null,
 "binding": {"rtx": [{"id": "GPU-3f2a…", "host": "pc"}],
             "pair": [{"id": "node1", "host": "node1"}, {"id": "node2", "host": "node2"}]},
 "devices": {"GPU-3f2a…": {"available": true, "disk_available": true}, "node1": {"available": true}, "node2": {"available": true}},
 "links": {"pc-nodes": {"up": true}, "pair.internal": {"up": true}},
 "hosts": {"pc": {"free_ram_gib": 70.0}, "node1": {"free_ram_gib": 20.0}, "node2": {"free_ram_gib": 21.0}},
 "descriptors": {"rtx": ["<sha256>"], "pair": ["<sha256>", "<sha256>"]},
 "running": [{"route": "decode_pair", "effective_sha256": "…", "plan_ids": ["…", "…"]}],
 "requirements": [{"route": "prefill_long", "set": "rtx", "member": 0, "id": "GPU-3f2a…",
                   "effective_sha256": "…", "descriptor_sha256": "…", "plan_id": "…",
                   "host_bytes": 56160000000, "commit_bytes": 143000000000, "disk_bytes": 25769803776}]}
```

`prompt_tokens` is the length the prefill computes: the prompt minus any prefix-cache hit (thresholds such as a
measured RTX/nodes crossover are in uncached tokens). `context_tokens` is the whole context the session may reach
(prompt + generation budget). Each requirement also carries `commit_bytes` and `disk_bytes` from the compiled plan's
`resources`; `hosts.<h>.free_commit_gib` is needed when `limits.host_free_commit_floor_gib` > 0.

`binding` is the match result's. Result `strix-stream/recipe-select-v1`: `{recipe, effective_sha256, route,
handoff_route, charge: {host: bytes}, trace: [{route, verdict, why}]}`.

### 8.4 CLI `ss-recipe`

```
ss-recipe validate RECIPE [LAYERS]
ss-recipe explain  RECIPE [LAYERS] [--json]
ss-recipe match    RECIPE PROBE.json [LAYERS]
ss-recipe select   RECIPE STATE.json [LAYERS]
ss-recipe plan     RECIPE --route R --device-set S [--member M] [--id PROBE_ID] --desc DESC.json
                   [--benefit FILE] [--ref SET=DESC.json[,BENEFIT]]... [LAYERS]
LAYERS: [--profile P --identity IDENTITY.json] [--overrides O]... [--set path=value]...
```

JSON on stdout. Exit 0 on success, 1 on a refusal (message on stderr), 2 on a usage error.
