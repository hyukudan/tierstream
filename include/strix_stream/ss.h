/*
 * strix-stream C ABI (v1).
 *
 * Model-generic weight streaming for engines that hold a model larger than the device: a descriptor says which
 * placeable units (expert slices, layer blobs) exist and where their bytes live; the library keeps a resident set in
 * device memory, streams the rest from a pinned host tier or from files read unbuffered through a staging ring, and
 * gives the engine one pointer table per layer to read weights through.
 *
 * Threading: one context is driven by one engine thread (every ss_* call on a context from the same thread, or
 * externally serialized), except ss_stats_snapshot* (any thread, see Stats). The library runs its own reader threads.
 *
 * Streams and lifetimes (design.md §6.1):
 *   - every stream argument of one forward (ss_layer_routes_dev, ss_acquire*, ss_group_wait, ss_release) must be the
 *     same engine stream; an engine that computes a layer on several streams joins them into that stream before
 *     ss_release (the library records its events there);
 *   - a ss_layer_view (tables, groups) is valid from ss_acquire* of L until ss_release of L;
 *   - CUDA / HIP graphs: ss_* calls are host calls that issue copies and events; they cannot be captured or run from
 *     a graph host node. Capture only the work between ss_acquire and ss_release, and pass tables by pointer
 *     indirection the graph reads at run time (table_dev changes between forwards).
 *
 * Versioning: ss_budget and ss_options start with struct_size. ss_budget_defaults / ss_options_defaults are inline
 * wrappers that pass the caller's sizeof to the library, which writes only that many bytes; every call that takes the
 * struct reads min(struct_size, its own size) bytes and defaults the rest. Fields are only ever appended, so callers
 * built against an older or newer header keep working.
 *
 * Failures: a library error (SS_E_INVALID, SS_E_IO, SS_E_HASH, SS_E_CAPACITY) leaves the context consistent: a failed
 * read is reported at ss_acquire / ss_end and the unit is read again at its next use. An out-of-memory, device
 * (SS_E_BACKEND / SS_E_NOMEM) or internal failure inside a context call poisons the context: every later call except
 * ss_close and the stats calls returns an error.
 *
 * Exactness: the library never changes bytes. A unit's bytes on the device are identical whether it is resident,
 * landed from the pinned tier or read from a file. Which slot holds a unit, and when it landed, is visible to the
 * engine only through the pointer table; kernels must give the same bits whatever the table says (design.md §4).
 */
#ifndef STRIX_STREAM_SS_H
#define STRIX_STREAM_SS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SS_BUILD_SHARED)
#define SS_API __declspec(dllexport)
#elif defined(_WIN32) && defined(SS_USE_SHARED)
#define SS_API __declspec(dllimport)
#else
#define SS_API
#endif

#define SS_ABI_VERSION 1

typedef enum ss_status {
  SS_OK = 0,
  SS_E_INVALID = 1,     /* bad argument or call order */
  SS_E_DESC = 2,        /* descriptor parse / validation error */
  SS_E_IO = 3,          /* file open / read error */
  SS_E_NOMEM = 4,       /* device or host allocation failed */
  SS_E_BACKEND = 5,     /* CUDA / HIP runtime error */
  SS_E_CAPACITY = 6,    /* plan or ring cannot hold what a layer needs */
  SS_E_HASH = 7,        /* landed bytes differ from the descriptor hash */
  SS_E_UNSUPPORTED = 8, /* backend not compiled in, or feature not in this backend */
  SS_E_RECIPE = 9       /* recipe parse / validation, stale profile, hardware refusal, no admissible route */
} ss_status;

/* Last error message of the calling thread (static storage, valid until the next failing call on that thread). */
SS_API const char* ss_last_error(void);
SS_API int ss_abi_version(void);

/* ---------------------------------------------------------------------------------------------------------------
 * Descriptor (design.md §2). Opaque; loaded from a desc-v1 JSON manifest or built by an adapter through the JSON
 * text. The descriptor carries engine bytes per unit (what the engine stores), not necessarily file bytes.
 * ------------------------------------------------------------------------------------------------------------- */
typedef struct ss_desc ss_desc;

SS_API ss_status ss_desc_from_json(const char* json_text, size_t len, ss_desc** out);
SS_API ss_status ss_desc_from_file(const char* path, ss_desc** out);
/* Same as ss_desc_from_file; the name the prefill adapters use (spec-rtx-prefill §6.4). Model-file readers (GGUF,
 * safetensors) live in the adapters / hyperloom and emit desc-v1; the library only reads the manifest. */
SS_API ss_status ss_desc_from_model(const char* manifest_path, ss_desc** out);
SS_API void ss_desc_free(ss_desc* d);
SS_API int ss_desc_n_layers(const ss_desc* d);
/* Units of a layer (experts for a routed layer, 1 for a layer blob, 0 for a layer that only has fixed bytes). */
SS_API int ss_desc_layer_units(const ss_desc* d, int layer);
SS_API uint64_t ss_desc_unit_bytes(const ss_desc* d, int layer);
/* Canonical JSON (sorted keys, no whitespace): what hyperloom's vramplan adapter reads. Caller frees with ss_free. */
SS_API ss_status ss_desc_to_json(const ss_desc* d, char** out_json, size_t* out_len);
SS_API void ss_free(void* p);

/* ---------------------------------------------------------------------------------------------------------------
 * Planner (design.md §5): which units are resident, which stream from the pinned tier, which from disk. Same rule as
 * hyperloom vramplan's third tier: fixed device bytes, margin max(margin_gib, margin_fraction x cap), stream slots
 * whenever anything streams, then units by order (prefix | ranked by benefit/byte, ties in model order); a
 * non-resident unit takes the pinned tier while host budget lasts (best benefit/byte first) if its tiers allow it,
 * else disk.
 * ------------------------------------------------------------------------------------------------------------- */
typedef enum ss_tier {
  SS_TIER_RESIDENT = 0,         /* on the device at open; reloaded from its file if evicted (dynamic mode) */
  SS_TIER_PINNED = 1,           /* not resident; copied from the pinned host tier */
  SS_TIER_DISK = 2,             /* not resident; read from its file */
  SS_TIER_PAIR = 3,             /* not here: another device computes it (never fetched) */
  SS_TIER_RESIDENT_BACKED = 4   /* resident at open, with a pinned host copy for when the dynamic set evicts it */
} ss_tier;

/* How the pinned host tier and the device slots share units (design.md §3.4).
 * INCLUSIVE: a unit promoted into a device slot keeps its pinned copy; an evicted unit without one reloads from its
 *   file. EXCLUSIVE: each unit lives in exactly one place at a forward boundary: a device slot, a pinned block, or (when
 *   neither holds it) only its file. A promotion frees the unit's block; an evicted unit is written back from its slot
 *   into a free block (device to host, on a second copy stream); when no block is free the miss is deferred (served
 *   from the stream ring, the victim stays), or with ss_options.exclusive_discard the victim falls back to its file.
 *   With device + host room for every unit, files are read only at open. */
typedef enum ss_tier_mode {
  SS_TIER_MODE_INCLUSIVE = 0,
  SS_TIER_MODE_EXCLUSIVE = 1
} ss_tier_mode;

typedef struct ss_budget {
  uint32_t struct_size;        /* sizeof(ss_budget), set by ss_budget_defaults */
  uint64_t device_bytes;       /* device memory the library and the engine's fixed bytes may use (the cap) */
  uint64_t host_pinned_bytes;  /* pinned host the library may lock for the pinned tier (0 = no pinned tier) */
  uint64_t workspace_bytes;    /* engine workspaces/state already sized by the engine for this request class */
  double margin_gib;           /* default 2.0 */
  double margin_fraction;      /* default 0.03 */
  /* Persistent decode slots per routed layer beyond the planned residents (0 = none; the dynamic resident set). */
  uint32_t dynamic_slots_per_layer;
  /* Slots per stream ring (one ring per streamed unit size; unit-sized device slots shared by every layer of that
   * size for transient units). 0 = descriptor ring_slots, else 2 x the most units of a layer of that size. */
  uint32_t ring_slots;
  /* 1 = after the pinned tier is placed, give resident units a pinned host copy (tier RESIDENT_BACKED) while the
   * host budget lasts, lowest benefit per byte first (the likeliest to be evicted): a decode replica whose dynamic set
   * evicts planner residents then reloads them over PCIe instead of from NVMe. */
  int pinned_backing;
  /* Never read: the bytes where older callers' structs had padding (their struct_size covers them). */
  uint32_t reserved0;
  /* ss_tier_mode (default INCLUSIVE). EXCLUSIVE refuses pinned_backing = 1, fill_residents = 0 and the HIP backend
   * (unified memory). */
  int tier_mode;
  /* EXCLUSIVE: free pinned blocks wanted per unit size (units whose 4 KiB-strided host bytes are equal share blocks).
   * Coverage first: they come from the host budget left after host_fixed and the pinned tier, never displacing a
   * unit's home (the plan JSON says how many fit). A miss that needs an eviction while no block is free is served from
   * the stream ring and its victim stays (counted as deferred; ss_options.exclusive_discard discards the victim to its
   * file instead); blocks freed by promotions are reusable from the next forward. 0 = auto: min(64, units of that
   * size). */
  uint32_t host_spare_units;
} ss_budget;

typedef struct ss_plan ss_plan;

/* Writes the defaults into the first `size` bytes of *b (struct_size = size). Use the inline ss_budget_defaults. */
SS_API void ss_budget_defaults_sized(ss_budget* b, uint32_t size);
static inline void ss_budget_defaults(ss_budget* b) { ss_budget_defaults_sized(b, (uint32_t)sizeof *b); }
/* benefit: optional per-unit benefit (global unit order, see ss_desc_unit_index), NULL = descriptor benefit. */
SS_API ss_status ss_plan_make(const ss_desc* d, const ss_budget* b, const double* benefit, size_t n_benefit,
                              ss_plan** out);
/* Tiers decided elsewhere (one ss_tier per unit in global order, or a plan-v1 JSON's "tiers"): same totals and
 * capacity checks as ss_plan_make. This is how an engine runs the placement hyperloom planned. */
SS_API ss_status ss_plan_from_tiers(const ss_desc* d, const ss_budget* b, const uint8_t* tiers, size_t n,
                                    ss_plan** out);
SS_API ss_status ss_plan_from_json(const ss_desc* d, const ss_budget* b, const char* plan_json, size_t len,
                                   ss_plan** out);
SS_API void ss_plan_free(ss_plan* p);
SS_API ss_tier ss_plan_unit_tier(const ss_plan* p, int layer, int unit);
typedef struct ss_plan_totals {
  uint64_t fixed_bytes, margin_bytes, resident_bytes, dynamic_bytes, ring_bytes, peak_device_bytes;
  uint64_t pinned_bytes, disk_bytes;
  uint64_t n_resident, n_pinned, n_disk;
} ss_plan_totals;
SS_API void ss_plan_get_totals(const ss_plan* p, ss_plan_totals* out);
SS_API ss_status ss_plan_to_json(const ss_plan* p, char** out_json, size_t* out_len);
/* Global unit index: layer-major, ascending unit id inside a layer (the order of per-unit arrays in this ABI). */
SS_API int64_t ss_desc_unit_index(const ss_desc* d, int layer, int unit);
SS_API int64_t ss_desc_total_units(const ss_desc* d);

/* ---------------------------------------------------------------------------------------------------------------
 * Context: one engine's streaming state on one device.
 * ------------------------------------------------------------------------------------------------------------- */
typedef enum ss_backend_kind {
  SS_BACKEND_CPU = 0,   /* host memory as "device" (tests, CPU engines); unified */
  SS_BACKEND_CUDA = 1,  /* discrete GPU: resident pool + ring in VRAM, H2D copy stream */
  SS_BACKEND_HIP = 2    /* Strix Halo unified memory: slots are host-mapped, file reads land in place */
} ss_backend_kind;

typedef enum ss_policy {
  SS_POLICY_LRU = 0,     /* least recently used among unpinned residents; hits refresh recency */
  SS_POLICY_SCORED = 1,  /* design.md §3.2: 0.7 transition + 0.2 window frequency + 0.1 recency (device-gated:
                            design.md §3.8, the boundary's victims) */
  SS_POLICY_SCORED_AUTO = 2 /* device-gated only: learn every cycle; scored victims after rows > 1, LRU otherwise */
} ss_policy;

struct ss_dg_unit; /* device-gated seed entry (below) */
/* appended: a file's identity as the operating system names it, for callers that authenticated a file by their own
 * handle (ss_options.source_identities, ss_mfile_location_info). Windows: the volume serial number and the file index
 * (BY_HANDLE_FILE_INFORMATION); Linux: st_dev and st_ino; size in bytes. All zero = none. */
typedef struct ss_file_identity {
  uint64_t volume;
  uint64_t index;
  uint64_t size;
} ss_file_identity;
typedef struct ss_options {
  uint32_t struct_size;           /* sizeof(ss_options), set by ss_options_defaults */
  ss_backend_kind backend;
  int device;                     /* CUDA/HIP device ordinal */
  ss_policy policy;
  uint32_t lookahead_budget;      /* speculative units per layer (default 2; 0 = off) */
  uint32_t reader_threads;        /* file reader threads (default 2) */
  uint32_t staging_buffers;       /* staging ring buffers (default 4) */
  uint64_t staging_bytes;         /* bytes per staging buffer (default 256 MiB; >= largest unit + 2 x align) */
  uint64_t copy_piece_bytes;      /* H2D pieces (default 64 MiB) */
  int verify_hash;                /* 1 = check xxh64 of every unit's bytes on the host side before it lands */
  int fill_residents;             /* 1 = load resident units at open (default 1) */
  uint32_t score_window;          /* routes in the frequency window (default 16) */
  int pin_planned;                /* 1 = planner residents are never evicted (prefill); 0 = they seed the dynamic set */
  int direct_io;                  /* 1 = unbuffered file reads (default); falls back to buffered if the FS refuses */
  uint32_t group_units;           /* misses per landing group in ss_acquire_groups (default 0 = one group) */
  /* 0 (default) = speculative reads land only in ring slots and never evict a resident; 1 = they go through
   * admission into persistent slots (only after a trace shows the extra evictions pay). */
  int spec_persistent;
  uint32_t reserved0;             /* never read: the bytes where older callers' structs had padding */
  /* Host read-ahead: pinned host buffers, each as large as the largest unit
   * that can be read from its file. ss_host_prefetch* and units waiting for a ring slot read their file into one
   * ahead of time; the unit then lands in its device slot (ring, or persistent when admitted) from that buffer. 0 = off
   * (default), at most 64. */
  uint32_t readahead_units;
  uint32_t reserved1;             /* explicit padding (never read) */
  /* Device-gated decode mode (docs/design.md §3.7): SS_FORWARD_DEVICE_GATED moves route -> slot resolution and the
   * residency check into the engine's (captured) graph; see "Device-gated decode" below. Older callers read as
   * SS_FORWARD_HOST. */
  int forward_mode;               /* ss_forward_mode, default SS_FORWARD_HOST */
  uint32_t dg_max_ids;            /* routed ids per gated layer per token, 1..64 (rows x top_k); 0 = not set */
  uint32_t dg_prefetch_reserve;   /* free slots a prefetch never takes; 0xFFFFFFFF (default) = min(24, spares / 4) */
  uint32_t dg_prefetch_per_layer; /* prefetch copies per (token, layer), default 4 */
  uint32_t dg_prefetch_per_token; /* prefetch copies per token, default 48 */
  uint32_t dg_wait_timeout_ms;    /* device wait and service deadlines, default 10000 */
  uint32_t dg_service_boost;      /* service thread: 0 off, 1 thread priority + MMCSS (default on Windows), 2 + process
                                     priority class */
  uint32_t dg_diag;               /* 1 = keep the per-layer timeline of the last token (ss_dg_timeline) */
  uint32_t dg_test_hooks;         /* test-only: 1 lets the ss_test_dg_* hooks (ss_test.h) act on this context; 0 */
  /* The open-time spare target: free slots each boundary keeps; 0xFFFFFFFF (default) = every billed spare (D);
   * otherwise <= D (ss_dg_set_budgets changes it between tokens). */
  uint32_t dg_spare_target;
  uint32_t dg_seed_n;             /* entries of dg_seed (0 = none) */
  /* Read only during ss_open (the array must stay valid for that call): gated units by priority, best first. The
   * device slots go to the first distinct entries (then the plan's residents, then key order), and that order is the
   * initial recency order. The plan's per-unit tiers then give the counts, not which units are resident. */
  const struct ss_dg_unit* dg_seed;
  /* appended: the spare slots D as one global count (the engine caches' --spares), instead of the plan's
   * dynamic_slots_per_layer summed over the gated layers; 0 = that sum. At most the non-resident gated units; the
   * arena is admitted with this D. */
  uint32_t dg_spare_slots;
  /* 1 = right after each layer's ready signal, push the demand stream's queued work to the GPU (a stream query: under
   * WDDM it flushes the driver's batch; it never waits); 0 (default) = leave batching to the driver */
  uint32_t dg_flush_ready;
  /* appended: 1 = a boundary's write-backs are not enqueued at the boundary: the service submits them part by part
   * after the layers it serves (after the layer's ready and prefetches, where the copy engine is idle; never after
   * the last gated layer); a boundary carries what is left into the next token, up to the larger of its own and the
   * previous boundary's victim count, and submits the older excess; a victim demanded or prefetched before any of its parts was submitted costs no
   * copy at all; the default (also for an older caller's struct, which ends before this field). 0 = every write-back
   * is enqueued at the boundary, right after the table upload. (The timeline's wb_left_ms measures the boundary's
   * write-backs: meaningful with 0 only.) */
  uint32_t dg_wb_defer;
  /* dg_wb_defer = 1: 1 (ss_options_defaults) = the parts go on the write-back stream, made to wait for the layer's
   * ready by an event, so they overlap the next layer's demand copies; 0 (also a zeroed struct) = on the demand stream
   * right after the layer's ready (stream order keeps them behind it, and the next layer's demand copies behind them).
   * Transport only: the same parts, bytes, destinations, credit, cap and completion events either way. */
  uint32_t dg_wb_stream;
  /* dg_wb_defer = 1: write-back bytes submitted per served layer (no more parts once the next layer has exported);
   * a part larger than what is left still goes and the excess is repaid by the next layers. 0 (default) = the bytes
   * of a unit's first part */
  uint32_t dg_wb_layer_bytes;
  /* dg_wb_defer = 1: no part is submitted while this many write-back bytes are in flight; 0 (default) = two units */
  uint32_t dg_wb_max_outstanding;
  /* appended, window prefetch (ss.h: Window prefetch): a tierstream/hint-prior-v1 file's bytes, read only during
   * ss_open (validated, copied into the context); 0 / NULL = no prior */
  uint64_t dg_prior_len;
  const void* dg_prior;
  /* EXCLUSIVE tier mode (host forward mode; ignored otherwise): 1 = an admission whose victims outnumber the free
   * blocks of their size is not deferred: the victims past the free blocks are discarded to their files (no device to
   * host copy, no block; exact, since weights are immutable and every unit's bytes are in its file) and the misses
   * take their slots, so the resident set keeps adapting when the pinned budget is full. A discarded unit is then
   * file-homed like a disk unit (read from its file, host read-ahead included, when it is fetched again); the blocks
   * keep their units. Speculative admissions (spec_persistent) keep the cap: a guess never discards. Counted in
   * ss_exclusive_stats.discards. 0 (default) = such misses are served from the stream ring and their victims stay
   * (deferred). ss_open refuses it with a descriptor whose tiers are "pinned". ss_recipe_plan rebuilds the options it
   * fills from the recipe: set this field after that call. */
  uint32_t exclusive_discard;
  uint32_t reserved2;             /* explicit padding (never read) */
  /* appended, scored eviction (policy SS_POLICY_SCORED in device-gated mode, docs/design.md §3.8): the score's integer
   * weights out of 1000 (transition, window frequency, recency), each 0..1000 and summing to 1000; all three 0 = the
   * defaults 700 / 200 / 100 (also what ss_options_defaults sets). Not read under SS_POLICY_LRU. */
  uint32_t dg_score_w_trans, dg_score_w_freq, dg_score_w_recency;
  uint32_t reserved3;  /* explicit padding (never read) */
  /* appended, scored eviction: a score profile's bytes (the transition counts the table starts from; format and
   * identity in "Device-gated decode" below), read only during ss_open (validated, copied into the context); 0 / NULL
   * = an empty table (online learning only). Needs SS_POLICY_SCORED and a descriptor with per-unit hashes. */
  uint64_t dg_score_profile_len;
  const void* dg_score_profile;
  /* appended, multi-disk weight streaming ("Multi-location files" below): ';'-separated directories holding stripe
   * copies of the descriptor's sources; the source's own directory and duplicates are ignored, and with no other
   * directory (NULL, empty) every source is read exactly as before. Copied at open. */
  const char* weight_dirs;
  uint32_t drive_readers;         /* reader threads per volume with weight_dirs: 1..32 (default 4), always validated */
  int weight_verify;              /* CACHED (default) or ALWAYS; always validated, including without weight_dirs */
  /* appended, device-gated three-tier placement (SS_DG_DISK_TIER below). dg_disk_tier: 1 = gated units of tier DISK
   * are accepted (file homes: read from their file on demand); 0 (default) = refused as before. dg_disk_prefetch: 1 =
   * predictions / window prefetches may read a disk unit too (the read runs while the layer computes; its copy and
   * entry follow the read, before the unit's layer is signalled ready); 0 (default) = disk units arrive on demand
   * only. dg_disk_staging: pinned one-unit buffers the file reads land in before their copy to the device (0 = 8;
   * only with dg_disk_tier). dg_inclusive_blocks: at most this many device-resident units keep their pinned block as
   * a clean copy after a promotion (their eviction is then a drop: no device-to-host copy); the blocks come from the
   * spare blocks (host_spare_units), so it must leave at least one spare block for write-backs; 0 (default) =
   * exclusive as before. */
  uint32_t dg_disk_tier;
  uint32_t dg_disk_prefetch;
  uint32_t dg_disk_staging;
  uint32_t dg_inclusive_blocks;
  /* appended, three-tier: file reads of disk units in flight at once (reader threads; also bounded by
   * dg_disk_staging): the misses of a layer are read together and each copy is enqueued once its read finished (the
   * same bytes, slots and table entries); a prefetch's read stays in flight across the service's calls until its
   * layer is demanded; 0 = 1 (default: one read at a time inside its fetch, as before), 2..64 = that many in flight;
   * only with dg_disk_tier. */
  uint32_t dg_disk_queue;
  uint32_t reserved4;  /* explicit padding (never read) */
  /* appended, bound source opens: every descriptor source is resolved once to its real path, then opened through
   * held handles of its directory and of each ancestor directory (see ss_mfile_location_info below), never reopened
   * by path. source_identities[i] (i < source_identity_n <= the descriptor's source count) is the identity
   * desc source i must have (volume, index and size; an all-zero entry is not checked): any other file fails ss_open
   * with SS_E_DESC naming the source. Read only during ss_open; 0 / NULL (default) = no expected identity. */
  uint32_t source_identity_n;
  uint32_t reserved5;  /* explicit padding (never read) */
  const ss_file_identity* source_identities;
  /* appended, whole-device budgets ("Whole-device budgets" below): a ledger the context charges its own device and
   * pinned allocations to (device-gated: the arena, slot table, controls, pinned blocks, staging, bounce and mapped
   * buffers; host mode: every allocation of its backend). An allocation that would exceed the ledger's cap fails
   * ss_open with SS_E_CAPACITY naming it. The ledger must outlive the context; closing the context releases its
   * charges. NULL (default) = nothing charged. */
  struct ss_budget_ledger* budget;
  /* appended, device-gated no-admission misses (SS_DG_ROUNDS below): 1 = a demand miss is served through a bank slot
   * (pinned block -> bank, or file -> staging -> bank) and released at ss_end without evicting a resident and without
   * changing the unit's home; no prefetch is issued and no boundary evicts, so placement changes only through
   * ss_dg_round, ss_dg_make_resident and ss_seed. Every distinct miss of a layer takes one of a bank's dg_max_ids
   * slots: ss_open refuses (SS_E_CAPACITY, naming the count) when the plan's ring cannot hold both banks and the
   * guard. 0 (default) = misses are admitted as before. */
  uint32_t dg_no_admit;
  uint32_t reserved6;  /* explicit padding (never read) */
} ss_options;

typedef enum ss_forward_mode { SS_FORWARD_HOST = 0, SS_FORWARD_DEVICE_GATED = 1 } ss_forward_mode;

typedef struct ss_ctx ss_ctx;

SS_API void ss_options_defaults_sized(ss_options* o, uint32_t size);
static inline void ss_options_defaults(ss_options* o) { ss_options_defaults_sized(o, (uint32_t)sizeof *o); }
/* Opens the sources named by the descriptor, allocates the resident pool, ring and pinned tier per the plan, fills
 * residents and the pinned tier. plan may be NULL (made from budget with descriptor benefits); a given plan must have
 * been made with this budget (SS_E_INVALID otherwise). The placement is fixed for the context's lifetime: a new
 * placement (another P bucket, another cap) is a new context. */
SS_API ss_status ss_open(const ss_desc* d, const ss_budget* budget, const ss_plan* plan, const ss_options* opt,
                         ss_ctx** out);
SS_API void ss_close(ss_ctx* c);
/* 1 if source i is read unbuffered (O_DIRECT / FILE_FLAG_NO_BUFFERING), 0 if the filesystem forced buffered IO. */
SS_API int ss_source_direct(const ss_ctx* c, int source);

/* Optional warm seed: make these units resident now (usage profile), ranked best first, within the dynamic slots. */
SS_API ss_status ss_seed(ss_ctx* c, int layer, const int32_t* units, int n);

/* One forward (a prefill request or a decode step). rows_per_layer may be NULL (all layers run); else
 * rows_per_layer[L] == 0 marks a layer the forward skips (e.g. a cone). Device-gated: a gated layer with 0 rows is
 * not exported by the step (no ss_dg_enqueue_export / predict / wait for it): the service does not wait for it and the
 * boundary treats it as unused this token (no lookups, misses or recency, no score history; its units age as unrouted
 * ones). A gated layer holding an ss_dg_make_resident set is never exported either, whatever its rows. */
SS_API ss_status ss_begin(ss_ctx* c, const uint32_t* rows_per_layer, int n_layers);

/* Routes of layer L for this forward: n ids (rows x top_k, any order, duplicates allowed, -1 ignored). Host ids. */
SS_API ss_status ss_layer_routes(ss_ctx* c, int layer, const int32_t* ids, int n);
/* Device ids: the library copies them back on its copy stream (async) and reads them at ss_acquire. */
SS_API ss_status ss_layer_routes_dev(ss_ctx* c, int layer, const int32_t* ids_dev, int n, void* engine_stream);
/* The whole layer is needed (layer-major prefill of a streamed layer, or a dense blob). */
SS_API ss_status ss_layer_all(ss_ctx* c, int layer);

typedef enum ss_prefetch_kind {
  SS_PREFETCH_CERTAIN = 0,      /* will be used by this forward: issue now, admission as a demand */
  SS_PREFETCH_SPECULATIVE = 1   /* predicted (router lookahead): budgeted, only while no demand is waiting */
} ss_prefetch_kind;
/* Ranked predicted ids (best first). Residents are skipped; a later demand for an in-flight unit claims it. */
SS_API ss_status ss_prefetch(ss_ctx* c, int layer, const int32_t* ids, int n, ss_prefetch_kind kind);
/* Whole-layer prefetch (layer-major prefill: issue layer L+k's streamed part during layer L). */
SS_API ss_status ss_prefetch_layer(ss_ctx* c, int layer);

/* Host-only lookahead (readahead_units > 0, else a no-op): start reading these file-backed units of layer L into host
 * read-ahead buffers now, without a device slot. A later fetch of the unit into a ring slot (routes, ss_layer_all,
 * ss_prefetch*) lands it from that buffer. Repeats in the same forward are ignored; allowed between forwards. Units that
 * are resident, in flight, computed by a peer or have a pinned source are skipped. */
SS_API ss_status ss_host_prefetch(ss_ctx* c, int layer, const int32_t* ids, int n);
SS_API ss_status ss_host_prefetch_layer(ss_ctx* c, int layer);

/* Lookahead helper (host): rows x n_experts scores of layer L+1's router applied to layer L's input; per row its
 * top_k, union ordered by (best rank, best score, id). Writes up to max_out ids, returns the count in *n_out. */
SS_API ss_status ss_lookahead_select(const float* scores, int rows, int n_experts, int top_k, int32_t* out,
                                     int max_out, int* n_out);

/* Layer view. table_dev[u] = device address of unit u's bytes (0 when u is not available this forward); table_host
 * is the same table in host memory. Groups: group 0 = units already resident when routes were given; groups 1..n
 * land in order. ss_acquire waits for everything; ss_acquire_groups returns at once and the engine gates each
 * group with ss_group_wait before reading it (hits first, misses as they land). */
typedef struct ss_group {
  const int32_t* units;  /* ascending unit ids of this group */
  int n_units;
} ss_group;
typedef struct ss_layer_view {
  const uint64_t* table_dev;   /* device pointer to n_units uint64 addresses (a device array) */
  const uint64_t* table_host;  /* host copy */
  int n_units;
  const ss_group* groups;
  int n_groups;
} ss_layer_view;

/* The engine stream waits (device-side) for every unit of L routed this forward; returns the view. */
SS_API ss_status ss_acquire(ss_ctx* c, int layer, void* engine_stream, ss_layer_view* out);
SS_API ss_status ss_acquire_groups(ss_ctx* c, int layer, void* engine_stream, ss_layer_view* out);
SS_API ss_status ss_group_wait(ss_ctx* c, int layer, int group, void* engine_stream);
/* Recorded on the engine stream after the last kernel that reads layer L's units: slots may be reused after it. */
SS_API ss_status ss_release(ss_ctx* c, int layer, void* engine_stream);
SS_API ss_status ss_end(ss_ctx* c);

/* Byte check (tests): copies unit (L, u)'s landed bytes back and compares them with the source and the descriptor
 * hash. Only valid between ss_acquire and ss_release of L. */
SS_API ss_status ss_check_unit(ss_ctx* c, int layer, int unit);

/* ---------------------------------------------------------------------------------------------------------------
 * Stats (design.md §7). Counters since open (or since ss_stats_reset); per layer and total.
 * ------------------------------------------------------------------------------------------------------------- */
typedef struct ss_layer_stats {
  uint64_t routed;          /* distinct routed units seen (per forward, summed) */
  uint64_t hits;            /* routed units found resident */
  uint64_t admitted;        /* misses that took a persistent slot */
  uint64_t transient;       /* misses served from the stream ring */
  uint64_t evicted;
  uint64_t prefetched;      /* routed units already issued by a certain prefetch of this forward */
  uint64_t spec_issued, spec_used, spec_dropped;
  uint64_t bytes_pinned;    /* bytes copied from the pinned tier */
  uint64_t bytes_disk;      /* bytes read from files */
  uint64_t wait_ns;         /* host time spent inside ss_acquire waiting for reads to be issued / to land */
} ss_layer_stats;

SS_API ss_status ss_stats_layer(const ss_ctx* c, int layer, ss_layer_stats* out);
SS_API ss_status ss_stats_total(const ss_ctx* c, ss_layer_stats* out);
SS_API void ss_stats_reset(ss_ctx* c);
/* Called at ss_end with the totals of that forward. */
typedef void (*ss_stats_hook)(void* user, const ss_layer_stats* forward_total, const ss_layer_stats* per_layer,
                              int n_layers);
SS_API void ss_set_stats_hook(ss_ctx* c, ss_stats_hook hook, void* user);
SS_API ss_status ss_stats_to_json(const ss_ctx* c, char** out_json, size_t* out_len);

/* EXCLUSIVE tier counters since open (not reset by ss_stats_reset); all zero in INCLUSIVE mode. Versioned like
 * ss_budget: use the inline ss_stats_exclusive. */
typedef struct ss_exclusive_stats {
  uint32_t struct_size;
  uint64_t writebacks;        /* evicted units written back from their slot into a pinned block */
  uint64_t writeback_bytes;
  uint64_t deferred;          /* misses served from the stream ring because no block was free for their victim */
  uint64_t promoted;          /* units whose pinned block was freed by a promotion into a device slot */
  uint64_t free_blocks;       /* free pinned blocks now (all sizes) */
  uint64_t min_free_blocks;   /* fewest free blocks at any forward start since open */
  uint64_t rereads;           /* copies out of a block whose write-back of the same forward had to land first */
  /* appended: victims discarded to their files (ss_options.exclusive_discard; not write-backs: no bytes copied) */
  uint64_t discards;
} ss_exclusive_stats;
SS_API ss_status ss_stats_exclusive_sized(const ss_ctx* c, ss_exclusive_stats* out, uint32_t size);
static inline ss_status ss_stats_exclusive(const ss_ctx* c, ss_exclusive_stats* out) {
  return ss_stats_exclusive_sized(c, out, (uint32_t)sizeof *out);
}

/* Stats snapshot (design.md §7.1): the one exception to the threading rule. ss_stats_snapshot* may be called from any
 * thread, concurrently with any call on the context and with each other, but never concurrently with or after
 * ss_close (stop the readers first). They work on a poisoned context and return the last publication.
 *
 * The first call arms the context: from then on the engine thread publishes an immutable copy of the counters at the
 * end of each ss_end that folds a forward's numbers (also one that reports a read failure) and of each ss_stats_reset
 * between forwards. While nothing is published (so normally on that first call) every call returns SS_E_INVALID with
 * its outputs untouched. Unarmed, the only added work is one relaxed load per ss_end / ss_stats_reset. A publication
 * holds every counter since open (or since the last reset) exactly as ss_stats_total / ss_stats_exclusive /
 * ss_stats_to_json gave them on the engine thread at that moment. Live changes between publications (inside a
 * forward, ss_seed promotions, read-ahead requested between forwards, a reset during a forward) appear at the next
 * publication; a poisoned context publishes no more. During the stats hook the previous publication is still shown.
 *
 * Versioned like ss_exclusive_stats: use the inline ss_stats_snapshot; min(info_size, sizeof) bytes are written and
 * only the fields wholly inside them are meaningful. Fields are only ever appended. SS_HAVE_STATS_SNAPSHOT says that
 * this header declares the calls (the library linked must export them). */
#define SS_HAVE_STATS_SNAPSHOT 1
#define SS_SNAPSHOT_READAHEAD 1u  /* read-ahead buffers exist: the readahead_* fields apply */
#define SS_SNAPSHOT_EXCLUSIVE 2u  /* EXCLUSIVE tier mode: the exclusive_* fields apply */
typedef struct ss_snapshot_info {
  uint32_t struct_size;
  uint32_t flags;                     /* SS_SNAPSHOT_* */
  uint64_t sequence;                  /* publications since arming (1 = the first) */
  uint64_t forwards;                  /* forwards ended at that publication (the JSON's "forwards") */
  /* Host read-ahead (0 unless SS_SNAPSHOT_READAHEAD), as the JSON's "readahead" object. */
  uint64_t readahead_buffers, readahead_buffer_bytes, readahead_requested, readahead_read, readahead_landed;
  uint64_t readahead_unused, readahead_failed, readahead_fallback, readahead_bytes;
  /* EXCLUSIVE tier (0 unless SS_SNAPSHOT_EXCLUSIVE), as ss_exclusive_stats plus the block count (the JSON's
   * "exclusive" object); exclusive_discards is appended after the first version's last field. */
  uint64_t exclusive_writebacks, exclusive_writeback_bytes, exclusive_deferred, exclusive_promoted;
  uint64_t exclusive_free_blocks, exclusive_min_free_blocks, exclusive_rereads, exclusive_blocks;
  uint64_t exclusive_discards;
} ss_snapshot_info;
/* total_out (may be NULL): the per-layer counters of the publication summed, as ss_stats_total. info_out (may be
 * NULL; not both): struct_size is set to the bytes written; info_size < 4 with info_out = SS_E_INVALID. */
SS_API ss_status ss_stats_snapshot_sized(const ss_ctx* c, ss_layer_stats* total_out, ss_snapshot_info* info_out,
                                         uint32_t info_size);
static inline ss_status ss_stats_snapshot(const ss_ctx* c, ss_layer_stats* total_out, ss_snapshot_info* info_out) {
  return ss_stats_snapshot_sized(c, total_out, info_out, (uint32_t)sizeof *info_out);
}
/* strix-stream/stats-v1 of the last publication: byte for byte the document ss_stats_to_json gave at that moment.
 * Freed with ss_free. */
SS_API ss_status ss_stats_snapshot_json(const ss_ctx* c, char** out_json, size_t* out_len);

/* ---------------------------------------------------------------------------------------------------------------
 * Device-gated decode (docs/design.md §3.7; CUDA backend, tier_mode EXCLUSIVE, policy LRU or SCORED, pin_planned 0).
 *
 * Sequence: ss_begin authorizes exactly one step (its begin kernel compares its sequence with the host's; any other
 * step is aborted on the device and reads only the guard slot). ss_begin and ss_end also check that no step ran
 * outside a pair and fail at once if one did. Residual: a step outside the pair that is still queued when ss_begin
 * runs can take the authorization, leaving the engine's real step to read only the guard slot; ss_end may then return
 * OK for that token, and the error surfaces at the next ss_begin / ss_end. So stream-synchronize any warm-up step
 * before ss_begin.
 *
 * Every streamable layer is a gated layer g = 0..G-1 (descriptor order; ss_dg_gated_index maps a layer). Their units
 * share one device arena per part: unit (g, u) is at parts[p].base + slot * parts[p].stride for
 * slot = slot_of[unit_base[g] + u] (-1: not on the device). Per token, inside the engine's graph (capturable, no host
 * work per token): ss_dg_enqueue_begin once; for each exported g in order (ss_begin: every gated layer except one with
 * 0 rows or holding an ss_dg_make_resident set): ss_dg_enqueue_export of g's routed ids, the engine's
 * work that does not read g's slots, ss_dg_enqueue_wait(g), then the readers of g's slots, which must resolve every id
 * through ss_dg_slot and must finish before the next export (a side branch must be joined before it);
 * optionally ss_dg_enqueue_predict(g) with g + 1's predicted ids on a side stream (the export of g promised their
 * count); ss_dg_enqueue_end after every reader and join. On the host: ss_begin before launching the step and ss_end
 * after it (ss_end waits for the end kernel, evicts and writes back, and fails the token on any device or service
 * error: the token's outputs must then be discarded and the context is poisoned). Exactly one step per
 * ss_begin / ss_end. The host-contract calls (ss_layer_routes*, ss_layer_all, ss_prefetch*, ss_host_prefetch*,
 * ss_acquire*, ss_group_wait, ss_release) are refused. Keep the context alive while a gated graph runs.
 * ------------------------------------------------------------------------------------------------------------- */
#define SS_DG_MAX_PARTS 4
#define SS_DG_MAX_IDS 64
#define SS_DG_PROTOCOL 1
typedef struct ss_dg_part {
  void* base;       /* device array of this part, `slots` entries */
  uint64_t stride;  /* bytes between slots (== bytes) */
  uint64_t bytes;
} ss_dg_part;
typedef struct ss_dg_device_view {  /* passed by value to the engine's kernels */
  uint32_t struct_size, protocol;   /* protocol = SS_DG_PROTOCOL */
  uint32_t n_gated, n_parts, guard_slot, slots;
  const int32_t* slot_of;           /* device, unit_base[n_gated] entries */
  const uint32_t* unit_base;        /* device, n_gated + 1 entries */
  /* device word, sticky: a wait timed out, an id was out of range, the host aborted the token, the step was not
   * authorized (sequence desync), or a lookup met an absent entry */
  const uint32_t* abort;
  ss_dg_part parts[SS_DG_MAX_PARTS];
  /* device word (appended, versioned by struct_size): a reader that meets an entry < 0 without abort sets it; ss_end
   * then fails the token. An engine kernel that reads slot_of itself must apply ss_dg_slot's two rules: abort set ->
   * the guard slot; an entry < 0 -> atomicOr(absent_err, 1) and the guard slot. */
  uint32_t* absent_err;
} ss_dg_device_view;
/* Kept units in any token: a unit kept on the device (an ss_dg_make_resident set, or an ss_dg_keep unit from the
 * boundary at which it is on the device) resolves through ss_dg_slot / ss_dg_slot_if_ready in every token, its layer
 * exported or not (no wait needed: its entry was published before the token began), and keeps its slot while it is
 * kept: entries change inside a token only from -1 to a slot, and the boundaries never evict a kept unit. The release
 * (or ss_dg_reset_open, which drops every keep) makes it an ordinary unit from the next boundary on. */
typedef struct ss_dg_unit {  /* a gated unit: gated ordinal and unit index */
  int32_t gated, unit;
} ss_dg_unit;
SS_API ss_status ss_dg_get_device_view_sized(const ss_ctx* c, ss_dg_device_view* out, uint32_t size);
static inline ss_status ss_dg_get_device_view(const ss_ctx* c, ss_dg_device_view* out) {
  return ss_dg_get_device_view_sized(c, out, (uint32_t)sizeof *out);
}
/* The gated ordinal of a descriptor layer, -1 when it is not gated (or the context is not device-gated). */
SS_API int ss_dg_gated_index(const ss_ctx* c, int layer);
/* Kernel launchers on `stream` (legal during stream capture; ids_dev, n and n_pred are baked into a captured graph). */
SS_API ss_status ss_dg_enqueue_begin(ss_ctx* c, void* stream);
SS_API ss_status ss_dg_enqueue_export(ss_ctx* c, int g, const int32_t* ids_dev, int n, int n_pred, void* stream);
SS_API ss_status ss_dg_enqueue_predict(ss_ctx* c, int g, const int32_t* ids_dev, int n, void* stream);
SS_API ss_status ss_dg_enqueue_wait(ss_ctx* c, int g, void* stream);
SS_API ss_status ss_dg_enqueue_end(ss_ctx* c, void* stream);
/* Between tokens: units of gated layer g the next boundary must not evict (opt-in; replaces the previous set of g,
 * also an ss_dg_make_resident set: g is exported again; n = 0 releases). A kept unit on the device stays in its slot
 * (no boundary evicts or reverts it) while kept. */
SS_API ss_status ss_dg_keep(ss_ctx* c, int g, const int32_t* ids, int n);
/* Between tokens: the spare target (<= the billed spares; the next boundary's evictions) and the prefetch budgets
 * (from the next token). The default prefetch reserve follows the spare target, min(24, spare_target / 4); an
 * explicit dg_prefetch_reserve stays. */
/* Between tokens: dg_wb_defer and dg_flush_ready at runtime (0 / 1 each; the same meaning as at open; turning
 * deferral off submits the deferred write-backs first). Placement / submission only: the same bytes. */
#define SS_DG_SET_SCHED 1
/* Between tokens: the device-resident units in recency order, most recent first (*total = how many; at most cap
 * written). A file of them written at shutdown seeds the next ss_open (dg_seed): the hot experts first. */
#define SS_DG_RESIDENCY 1
SS_API ss_status ss_dg_residency(ss_ctx* c, ss_dg_unit* out, uint32_t cap, uint32_t* total);
SS_API ss_status ss_dg_set_sched(ss_ctx* c, uint32_t wb_defer, uint32_t flush_ready);
SS_API ss_status ss_dg_set_budgets(ss_ctx* c, uint32_t spare_target, uint32_t prefetch_per_layer,
                                   uint32_t prefetch_per_token);
/* Per-layer timeline of the last completed token (dg_diag = 1), between tokens; the replica's definitions: ids = unique
 * ids, misses = unique ids with a masked occurrence, covered / fetched per masked occurrence in export order (covered =
 * on the device or coming at service time), seen_us of layer 0 from the previous token's end, demand_us = flag seen ->
 * demand copies + ready enqueued, dma_ms = device time of those copies + ready (-1 = no miss or no diag),
 * predicted_true = unique ids found in the previous layer's prediction, covered_waits = demand waits on a prefetch.
 * Appended, from the device's clock (layers with a miss; -1 otherwise): gpu_wait_us = the wait kernel's entry -> ready
 * seen (the exposed stall), gpu_export_ready_us = the export kernel's flag write -> ready seen (the whole miss path:
 * service, enqueue, submission, copies, ready). wb_left_ms = how long the previous boundary's write-backs still ran
 * after this layer's demand copies were submitted (0 = already done; -2 = still running at this boundary; -1 = no
 * miss, none issued, or not measurable). dma_first_ms = demand submission -> the first fetched unit's copies done (-1 =
 * no fetch): under a FIFO copy engine about wb_left_ms + one unit's copy time. */
typedef struct ss_dg_layer_timeline {
  uint32_t struct_size, layer;
  double seen_us, demand_us, pred_wait_us, prefetch_us, serve_us;
  uint32_t ids, misses, covered, fetched, prefetched, reserved0;
  double dma_ms;
  uint32_t predicted_true, covered_waits;
  double gpu_wait_us, gpu_export_ready_us, wb_left_ms, dma_first_ms;
} ss_dg_layer_timeline;
SS_API ss_status ss_dg_timeline_sized(const ss_ctx* c, int g, ss_dg_layer_timeline* out, uint32_t size);
/* The last successful boundary (ss_end) of a dg_diag context, host microseconds: sync = the demand and prefetch
 * streams' synchronize, select = admissions, completed write-backs and victim selection, upload = the table upload
 * and its synchronize, issue = enqueueing the write-backs; writebacks = how many were issued. */
typedef struct ss_dg_boundary_timeline {
  uint32_t struct_size, writebacks;
  double sync_us, select_us, upload_us, issue_us;
} ss_dg_boundary_timeline;
SS_API ss_status ss_dg_boundary_timeline_sized(const ss_ctx* c, ss_dg_boundary_timeline* out, uint32_t size);
/* Scored eviction (policy = SS_POLICY_SCORED; docs/design.md §3.8). Only the boundary's choice of victims changes: the
 * eligible units (on the device, no write-back in flight, not kept by ss_dg_keep) are taken lowest score first,
 * score = w_t P + w_f F + w_r Rec in integers: P = the transition estimate from this token's union of the layer's
 * routed ids (counts of "previous token's union -> this union" per gated layer, uint32 cells that saturate), F = how
 * many of the layer's last 16 unions hold the unit, Rec = 1 / (1 + tokens since its last routed use); ties go to the
 * older use, then to the less recent unit. The history lives as long as the context (ss_stats_reset keeps it;
 * ss_dg_reset_open restores the open state: the profile's counts or an empty table). One ss_begin / ss_end is one
 * history step, every exported row included. Pressure tests and the write-backs are the same in both policies.
 * Profile file (little endian, frozen): "TSSCPRF1", u32 version 1, u32 n_gated, u32 units[n_gated], identity[32],
 * u32 prov_len (<= 4096), prov[prov_len] (provenance, reported in the stats JSON), then per gated layer units^2 u32
 * counts row-major [previous unit][unit], then sha256[32] of every byte before it. identity = sha256 of u32 n_gated,
 * then per gated layer: u32 model layer index, u32 units, u64 unit_bytes, the descriptor's per-unit xxh64 list (a
 * descriptor without them cannot load a profile). Any mismatch (size, magic, version, units, identity, checksum) fails
 * ss_open. Host memory: 4 x units^2 bytes per gated layer (twice that with a profile: its counts are kept for
 * reset), pageable, outside the pinned admission. Device-gated ss_open refuses a policy other than LRU / SCORED / AUTO with
 * SS_E_INVALID (before scored eviction, SCORED itself was SS_E_UNSUPPORTED). */
#define SS_DG_SCORED 1
/* Counters since open (or the last ss_stats_reset), folded at each ss_end. */
typedef struct ss_dg_stats {
  uint32_t struct_size, reserved0;
  uint64_t tokens, lookups, misses, covered, fetched, prefetched, prefetch_used, pred_invalid;
  uint64_t resurrected, bank_serves, collected_mid, evicted, evict_short, writebacks_done, stalled;
  uint64_t free_slots, blocks, slots;
  /* appended: predicted (deduplicated valid predicted ids), predicted_true, miss_layers (gated layers with >= 1 miss,
   * per token), device_timeouts (device waits that timed out: since open, not reset by ss_stats_reset), covered_waits */
  uint64_t predicted, predicted_true, miss_layers, device_timeouts, covered_waits;
  /* appended, write-back scheduling (dg_wb_defer = 1; 0 otherwise unless noted): wb_deferred = victims whose
   * write-back was deferred at a boundary; wb_cancelled = of those, demanded or prefetched before any part was
   * submitted (no copy; also counted in resurrected); wb_forced = submitted early and waited for because a demand
   * found no free slot; wb_boundary_flushed = not fully submitted by the next boundary, submitted there. Bytes:
   * issued = every write-back copy submitted (both modes), completed = of those, collected; cancelled = never copied
   * (cancelled victims, and the unsent parts of a partly submitted victim that was resurrected); boundary_flushed =
   * submitted by a boundary's flush; outstanding = submitted, not collected yet, at the last boundary (both modes);
   * forced_wait_ns = the host's waits of the forced path */
  uint64_t wb_deferred, wb_cancelled, wb_forced, wb_boundary_flushed;
  uint64_t wb_issued_bytes, wb_completed_bytes, wb_cancelled_bytes, wb_boundary_flushed_bytes, wb_outstanding_bytes;
  uint64_t wb_forced_wait_ns;
  /* appended, window prefetch (winpf; 0 when off): hints and hinted rows; prior (between tokens) and l+1 (in-token)
   * prefetch copies issued, used by their token, late (a demand of their token met the copy in flight and waited for
   * it), reverted (unused, back to the host block at the boundary); l+1 candidates ranked and priced out by their
   * bucket, skipped because layer l+1 had already exported. Wasted bytes = (issued - used) x the unit bytes. */
  uint64_t winpf_hints, winpf_hint_rows, prior_issued, prior_used, prior_late, prior_reverted;
  uint64_t l1_candidates, l1_issued, l1_used, l1_late, l1_reverted, l1_priced_out, l1_skipped_ahead;
  uint64_t prior_budget_stops;  /* hints whose prior prefetches stopped at the copy-engine byte budget */
  /* appended, scored eviction (0 under LRU): score_fallbacks = boundaries that used the recency order because the
   * score lists were not this token's (an internal consistency guard: should stay 0); scored_victims = victims taken
   * in score order; score_layers / score_candidates = layers ranked and the units they ranked; score_saturated =
   * transition increments refused by a saturated (2^32 - 1) cell; score_ns = ranking / selection time (the bounded
   * selector runs at the boundary, scores only eligible keys, and skips ranking on AUTO R1 or when no victims are needed) */
  uint64_t score_fallbacks, scored_victims, score_layers, score_candidates, score_saturated, score_ns;
  /* appended, scored eviction: score_deferred = layers whose ranking waited behind a later layer because the device
   * had already exported it; score_last_ns = ranking time after the last gated layer was served (the terminal path,
   * part of score_ns) */
  uint64_t score_deferred, score_last_ns;
  /* appended, three-tier placement (SS_DG_DISK_TIER; 0 when off): disk_reads / disk_read_bytes / disk_read_ns = file
   * reads of disk-homed units (demand, prefetch) and their summed read time; disk_stage_waits = reads that waited for a
   * staging buffer's previous copy; disk_drops = pinned units dropped to their file to free a block for a write-back
   * (RAM -> drop, no copy); disk_discards = device victims dropped to their file because no block could be freed (no
   * copy); inclusive_kept = promotions that kept their block as a clean copy; inclusive_drops = evictions of such units
   * (no write-back); disk_units / backed_units = file-homed units and device units holding a clean copy at the last
   * boundary (gauges, not differences) */
  uint64_t disk_reads, disk_read_bytes, disk_read_ns, disk_stage_waits, disk_drops, disk_discards;
  uint64_t inclusive_kept, inclusive_drops, disk_units, backed_units;
  /* appended, three-tier: the service's wall time waiting for file reads to finish (disk_read_ns sums the reads' own
   * times: with dg_disk_queue > 1 they overlap, so this is the smaller one) */
  uint64_t disk_wait_ns;
  /* appended, three-tier overlap receipts: disk_overlapped = file reads begun while another was still in flight;
   * disk_max_in_flight = the most in flight at once since open (a gauge, not a difference); disk_carried /
   * disk_carried_bytes = prefetch reads that ended in a later service call than the one that began them (they ran
   * while the layer computed) */
  uint64_t disk_overlapped, disk_max_in_flight, disk_carried, disk_carried_bytes;
  /* appended, kept-resident layers (SS_DG_MAKE_RESIDENT): units ss_dg_make_resident copied in (from a pinned block or
   * a file), device units it evicted, the bytes it copied in, its host time. Its copies count nowhere else (not
   * misses, fetches, prefetches, evictions, write-backs or file reads). */
  uint64_t resident_units, resident_evicted, resident_bytes, resident_ns;
  /* appended, placement rounds (SS_DG_ROUNDS): rounds run, units whose home they changed (promoted to the device or to
   * the pinned tier, demoted from the device, dropped to their file), the bytes they moved (file reads + host-to-device
   * copies + write-backs), their host time, rounds that stopped at a cancellation. Their copies count nowhere else. */
  uint64_t round_count, round_units, round_bytes, round_ns, round_cancelled;
} ss_dg_stats;
SS_API ss_status ss_stats_dg_sized(const ss_ctx* c, ss_dg_stats* out, uint32_t size);
static inline ss_status ss_stats_dg(const ss_ctx* c, ss_dg_stats* out) {
  return ss_stats_dg_sized(c, out, (uint32_t)sizeof *out);
}
/* Window prefetch (winpf): an opt-in placement policy for multi-row (speculative verify)
 * windows. Off until ss_dg_winpf_set enables it; off = today's prefetch byte for byte. Only placement changes (which
 * units are copied when): the copies, the table protocol and the engine's arithmetic are those of every prefetch.
 * - l+1, for a token that got a window hint (otherwise today's prefetch): the prediction an export promised (rows x K
 *   ids, row-major, the predictor's order inside a row; rows = the export's ids / top_k; an export whose ids or
 *   predicted ids are not such multiples fails the token) is ranked by weighted row votes, then by the best rank; the
 *   head is admitted while votes >= l1_min_votes, at most l1_cap per layer per token (and under the generic prefetch
 *   budgets), skipping a (votes, rank) bucket whose online precision (prefetches their token used) is below l1_pmin
 *   (every 64th skipped candidate is still admitted so the estimate can recover; the buckets live as long as the
 *   context, ss_stats_reset does not touch them). Nothing is issued when layer l+1 already exported (the device ran
 *   ahead). Order on the copy engine: after the layer's demand and ready, before deferred write-back parts.
 * - Prior: the ss_open option dg_prior (tierstream/hint-prior-v1, below) maps a symbol (token id) to ranked (unit,
 *   P(used)) entries for gated layers 0..P-1. ss_dg_window_hint, between tokens (after ss_end, before ss_begin), appends
 *   the next window's rows (symbols + weights in [0, 1]; call it for the bonus row right after ss_end and again with the
 *   drafts after the draft step) and issues prior prefetches at once: per gated layer < min(prior_layers, P) the units
 *   with P(used by a hinted row) = 1 - prod(1 - w p) >= prior_pmin, best first, at most prior_cap new ones per call,
 *   while the write-back bytes already queued + this window's prior bytes stay <= prior_window_us x the copy rate
 *   (copy_gbps, replaced by the context's own first-copy measurements under dg_diag). A step queued before ss_begin
 *   (a future pre-launch) may export before a prior copy lands: it then reads -1 and its demand waits for the copy
 *   (exact, as any covered prefetch).
 * - revert_unused: in a hinted window, a prefetch its token did not use goes back to its host block at the boundary
 *   (no write-back). A hinted window's exports must hold rows x top_k ids (else the token fails); at most 64 rows may be
 *   hinted per window (SS_E_INVALID).
 * Prior file (little endian, frozen): "TSHPRIO1", u32 version 1, u32 n_gated, u32 units[n_gated], identity[32],
 * u32 prov_len (<= 4096), prov[prov_len], u32 layers (1..n_gated), u32 k (1..64), u32 symbols, then per symbol: i32
 * symbol, layers x k u16 units (0xFFFF = none), layers x k u16 P(used) x 65535 (best first per layer); then sha256[32]
 * of every byte before it. identity = sha256 of u32 n_gated, then per gated layer: u32 model layer index, u32 units,
 * u64 unit_bytes, per part u64 offset, u64 stride, u64 bytes, then the descriptor's per-unit xxh64 list if it has one.
 * Any mismatch (size, magic, version, units, identity, checksum, a duplicate symbol) fails ss_open. */
typedef struct ss_dg_winpf_params {
  uint32_t struct_size;
  uint32_t enable;           /* 1 = on (default 0) */
  uint32_t prior_layers;     /* default 4 */
  uint32_t prior_cap;        /* default 1 (0 = no prior prefetch) */
  float prior_pmin;          /* default 0.7 */
  uint32_t prior_window_us;  /* default 1500 (the copy-engine time the draft step leaves before the first export) */
  float copy_gbps;           /* default 30 (bytes -> copy-engine time for the prior budget) */
  uint32_t l1_cap;           /* default 1 */
  float l1_min_votes;        /* default 1 */
  float l1_pmin;             /* default 0.4 (0 = no precision floor) */
  uint32_t revert_unused;    /* default 1 */
  uint32_t top_k;            /* ids per row in every export (the engine's top_k): required (1..64) when enable = 1 */
} ss_dg_winpf_params;
SS_API void ss_dg_winpf_defaults_sized(ss_dg_winpf_params* p, uint32_t size);
static inline void ss_dg_winpf_defaults(ss_dg_winpf_params* p) { ss_dg_winpf_defaults_sized(p, (uint32_t)sizeof *p); }
/* Between tokens. NULL = the defaults (enable 0). */
SS_API ss_status ss_dg_winpf_set(ss_ctx* c, const ss_dg_winpf_params* p);
/* Between tokens (see above). weights may be NULL (all 1). At most SS_DG_MAX_IDS rows per window. */
SS_API ss_status ss_dg_window_hint(ss_ctx* c, const int32_t* symbols, const float* weights, uint32_t rows);
#if defined(__CUDACC__)
/* The slot an engine kernel reads exported unit `id` of gated layer g from, after ss_dg_enqueue_wait(g): the zeroed
 * guard slot when the token was aborted or the id is out of range (every read stays in bounds; the token is then
 * refused). An absent entry without abort cannot happen when the protocol is followed: it is reported (absent_err)
 * and read as the guard. */
static __device__ __forceinline__ int32_t ss_dg_slot(const ss_dg_device_view* v, uint32_t g, int32_t id) {
  if (*(volatile const uint32_t*)v->abort) return (int32_t)v->guard_slot;
  const uint32_t b = v->unit_base[g];
  if ((uint32_t)id >= v->unit_base[g + 1] - b) return (int32_t)v->guard_slot;
  const int32_t s = ((volatile const int32_t*)v->slot_of)[b + (uint32_t)id];
  if (s >= 0) return s;
  if ((uint32_t)((const char*)&v->absent_err - (const char*)v) + (uint32_t)sizeof(void*) <= v->struct_size &&
      v->absent_err)
    atomicOr(v->absent_err, 1u);
  return (int32_t)v->guard_slot;
}
/* Before ss_dg_enqueue_wait(g) (resident-first execution): the slot of exported unit `id` of gated layer g if its
 * bytes are already complete, else -1 (wait, then use ss_dg_slot); the guard slot when the token was aborted or the
 * id is out of range. Inside a token an entry only changes from -1 to a slot, published after its copies, so a slot
 * returned here stays valid until the token ends (readers of g still finish before the export of g + 1). The load is
 * an acquire at system scope; an absent entry is not reported. */
static __device__ __forceinline__ int32_t ss_dg_slot_if_ready(const ss_dg_device_view* v, uint32_t g, int32_t id) {
  if (*(volatile const uint32_t*)v->abort) return (int32_t)v->guard_slot;
  const uint32_t b = v->unit_base[g];
  if ((uint32_t)id >= v->unit_base[g + 1] - b) return (int32_t)v->guard_slot;
  int32_t s;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(s) : "l"(v->slot_of + b + (uint32_t)id) : "memory");
  return s >= 0 ? s : -1;
}
#endif

/* ---------------------------------------------------------------------------------------------------------------
 * Recipes (docs/recipes.md): how one model runs on one hardware set per workload class, compiled into ss_budget /
 * ss_options / plan-v1. Every char** output is NUL-terminated and freed with ss_free (a length pointer may be NULL);
 * on failure nothing is written to any output. A recipe holds its profile / override layers; each change is atomic
 * (a refused layer leaves the effective recipe as it was).
 * ------------------------------------------------------------------------------------------------------------- */
typedef struct ss_recipe ss_recipe;

/* TOML, or JSON when the first non-blank character is '{'. origin names the text in messages (NULL = "<text>"). */
SS_API ss_status ss_recipe_from_text(const char* text, size_t len, const char* origin, ss_recipe** out);
SS_API ss_status ss_recipe_from_file(const char* path, ss_recipe** out);
SS_API void ss_recipe_free(ss_recipe* r);
/* The autotuner's profile (strix-stream/recipe-profile-v1), checked against the current identity (probe report +
 * workload + descriptors, docs/recipes.md §8.2); a stale profile is refused. At most one. */
SS_API ss_status ss_recipe_add_profile(ss_recipe* r, const char* text, size_t len, const char* origin,
                                       const char* identity_json, size_t identity_len);
/* A human's overrides (strix-stream/recipe-overrides-v1). */
SS_API ss_status ss_recipe_add_overrides(ss_recipe* r, const char* text, size_t len, const char* origin);
/* One override: path (docs/recipes.md §6.2) = value as TOML value text ("33", "\"fill\"", "[[0, 3], 7]"). */
SS_API ss_status ss_recipe_set(ss_recipe* r, const char* path, const char* value, const char* origin);
/* The effective recipe (every default written out), canonical JSON. */
SS_API ss_status ss_recipe_to_json(const ss_recipe* r, char** out, size_t* len);
/* strix-stream/recipe-explain-v1: every field with its value, layer, origin and the values it shadowed. */
SS_API ss_status ss_recipe_explain(const ss_recipe* r, char** out, size_t* len);
/* Hardware check of a probe report: strix-stream/recipe-match-v1 with the device binding, or SS_E_RECIPE listing
 * every unmet requirement. */
SS_API ss_status ss_recipe_match(const ss_recipe* r, const char* probe_json, size_t len, char** out, size_t* olen);
/* Admission of a new request (or the next step of a session) from the request state: strix-stream/recipe-select-v1
 * naming the route, or SS_E_RECIPE listing why each visited route was not admitted. */
SS_API ss_status ss_recipe_select(const ss_recipe* r, const char* state_json, size_t len, char** out, size_t* olen);
/* Compiles one member of a route's device set (docs/recipes.md §8.1). device_id: the member's bound probe id.
 * desc / benefit: that member's descriptor and per-unit benefit (NULL = the descriptor's). ref_*: indexed like the
 * route's devices (member 0 of each set), required for hetero routes, else NULL / 0. budget and options must be
 * initialized with ss_budget_defaults / ss_options_defaults; options->device keeps the caller's value. plan may be
 * NULL; out (may be NULL) receives strix-stream/recipe-plan-v1. */
SS_API ss_status ss_recipe_plan(const ss_recipe* r, const char* route, const char* set, int member,
                                const char* device_id, const ss_desc* desc, const double* benefit, size_t n_benefit,
                                const ss_desc* const* ref_descs, const double* const* ref_benefits,
                                const size_t* ref_n_benefits, size_t n_refs, ss_budget* budget, ss_options* options,
                                ss_plan** plan, char** out, size_t* olen);

/* Opt-in device-gated cache reset, engine thread between tokens only. Restores the resolved ss_open placement,
   canonical slot/block numbering, recency, budgets and policy learning in place; captured device addresses and
   sequence/protocol words stay valid. Drains/collects outstanding work and invalidates previous-token diagnostics.
   Moves only misplaced units. Refuses non-gated, in-token and poisoned contexts; a reset failure poisons the context.
   Core counters remain lifetime counters. Resettable statistics are rebased as by ss_stats_reset (exclusive lifetime
   counters retain their existing semantics). Consumers computing deltas must rebase their own before snapshots.
   Identical reset state does not promise identical live timing-dependent scheduler decisions/counters. */
/* feature macro: engines built against older trees compile the reset out */
#define SS_DG_RESET_OPEN 1
SS_API ss_status ss_dg_reset_open(ss_ctx* c);

/* Additive row-aware boundary. rows must be positive and describes the completed window, including uncommitted
   verify rows. ss_end / ts_end is exactly one row. AUTO learns as SCORED on every cycle, but R1 uses the unchanged
   LRU walk without ranking. The public integer score, weights and optional TSSCPRF1 profile are shared by both. */
#define SS_DG_AUTO_EVICT 1
/* Three-tier device-gated placement (ss_options dg_disk_tier / dg_disk_prefetch / dg_disk_staging /
 * dg_inclusive_blocks): device slots, pinned blocks, then the unit's file. With dg_disk_tier = 1 a plan may give gated
 * units tier DISK: such a unit has no block; a miss reads it from its file (through dg_weight_dirs stripes when given)
 * into a pinned staging buffer and copies it to its slot. Victims go down one tier: a device victim is written back
 * into a free block (device -> pinned); when no block is free, the least recently homed pinned unit is dropped to its
 * file (no copy: the file holds the same bytes) and its block takes the victim; with no pinned unit left the victim
 * itself is dropped to its file. Bank serves, reverted prefetches and their returns of a disk unit go back to the file.
 * The pinned units are kept in their own recency order (most recently homed first). Inclusive share
 * (dg_inclusive_blocks > 0): a promotion from a pinned block keeps the block as a clean copy while fewer than that many
 * device units hold one; evicting such a unit makes the block its home again with no copy. Placement only: every byte
 * a slot receives is the unit's file bytes, so the outputs are those of the exclusive two-tier cache. Both off (the
 * defaults) = the two-tier exclusive behaviour byte for byte. ss_dg_reset_open restores the open placement (clean
 * copies dropped, disk units read again where the open placed them). */
#define SS_DG_DISK_TIER 1
/* ss_options.dg_disk_queue and ss_dg_stats.disk_wait_ns exist (overlapped three-tier file reads) */
#define SS_DG_DISK_QUEUE 1
/* ss_dg_stats.disk_overlapped / disk_max_in_flight / disk_carried / disk_carried_bytes exist (overlap receipts) */
#define SS_DG_DISK_RECEIPTS 1
SS_API ss_status ss_dg_end_rows(ss_ctx* c, uint32_t rows);
/* Engine thread, between tokens after the device and service have returned ownership (as after ss_end).
   Drains/collects write-backs, switches LRU / SCORED / SCORED_AUTO, and resets learning to the open profile (or zero)
   and demand epochs, retaining placement and recency. Counters remain lifetime counters. To switch to scoring the
   context must have been opened with SCORED or SCORED_AUTO (tables provisioned at open); an LRU-only open refuses
   that switch with SS_E_INVALID. reset_open restores the open policy as well as its learning and placement. */
#define SS_DG_SET_POLICY 1
SS_API ss_status ss_dg_set_policy(ss_ctx* c, ss_policy policy);


/* Multi-location files (docs/design.md §4.4). A model file (the primary) plus stripe copies of some of its chunks on
 * other drives: each extra directory holds "<primary basename>.stripe.json" (tierstream/stripe-v1: chunk size, the
 * primary's SHA-256, the placed chunks and their xxh64) and the data file it names. Reads are split at chunk
 * boundaries and served by one reader pool per drive. Every byte served from a secondary location belongs to a chunk
 * whose xxh64 matched the manifest, checked through the serving handle at open (verify_mode ALWAYS) or at the first
 * open of that data-file version (CACHED, the result kept by file identity); with ALWAYS also again on every read
 * covering a whole chunk. Any read error, short read or mismatch serves that chunk from the primary from then on. A missing drive,
 * manifest or data file, or a manifest bound to other primary bytes, disables that location (never an error).
 * Sessions are looked up by (canonical primary, directories, direct flag), reused only while file identities and
 * manifests hold, drive_readers matches and verification is at least as strict as requested. Absent manifests stay
 * reusable while absent. Reader pools are shared process-wide by (volume identity, drive_readers), across files.
 * Windows opens files with FILE_SHARE_READ only, excluding writers for the session's lifetime. Linux has no such
 * exclusion: stripe data is published 0444; owners or privileged processes can still modify it. Identity (including
 * Linux ctime) is checked on reuse and after verification, not on every partial read. Keep files immutable while
 * sessions live; use ALWAYS on Linux where stripe directories are not exclusively owned (partial reads still rely
 * on the open-time check). Stripe directories must be as trusted as the primary: xxh64 detects accidental corruption,
 * not deliberate tampering. First open without a supplied primary_sha256 or identity cache hashes the entire primary
 * and verifies the stripe; hundreds of GiB can take minutes. CACHED persists the stripe verification by identity.
 * Concurrent reads, counter queries and accessors on a live handle are supported; read destinations must not overlap
 * concurrently. Each call waits for all its own jobs, including on error. Close must not race any call on that handle;
 * closing a separate handle to the same shared session is safe.
 * Read contract: offset, bytes and dst are multiples of ss_mfile_alignment (4096); a read may extend past the end
 * of the file up to the aligned end (got = the bytes that exist; bytes past EOF are zero-filled on every route). */
#define SS_MFILE 1
typedef struct ss_mfile ss_mfile;
typedef enum ss_mfile_verify {
  SS_MFILE_VERIFY_CACHED = 0,     /* verify a data-file version once; reuse the result while its identity holds */
  SS_MFILE_VERIFY_ALWAYS = 1,     /* verify every placed chunk when a session is created (a reused session is not
                                    re-verified) and re-check whole-chunk reads inline (qualification) */
  SS_MFILE_VERIFY_OFF = 2         /* C++ tests only; rejected by ss_open and ss_mfile_open */
} ss_mfile_verify;
typedef struct ss_mfile_options {
  uint32_t struct_size;           /* sizeof(ss_mfile_options), set by ss_mfile_options_defaults */
  int direct_io;                  /* 1 = unbuffered reads (default) */
  uint32_t drive_readers;         /* reader threads per volume: 1..32 (default 4), always validated */
  int verify_mode;                /* ss_mfile_verify */
  const char* primary_sha256;     /* optional admitted SHA-256 (hex), trusted without checking: MUST describe the
                                    current primary bytes; NULL = full-file hash computed / cached by identity */
} ss_mfile_options;
typedef struct ss_mfile_io {
  void* dst;
  uint64_t offset;
  uint64_t bytes;
} ss_mfile_io;
#define SS_MFILE_MAX_LOCATIONS 8  /* the primary + 7 secondary directories */
typedef struct ss_mfile_counters {
  uint32_t struct_size;           /* set by the caller */
  uint32_t locations;             /* location 0 = the primary */
  uint64_t bytes[SS_MFILE_MAX_LOCATIONS];
  uint64_t reads[SS_MFILE_MAX_LOCATIONS];
  uint64_t busy_ns[SS_MFILE_MAX_LOCATIONS];   /* summed read time of that location's reads */
  uint64_t fallback_bytes;        /* bytes re-read from the primary after a secondary failure */
  uint64_t demoted_chunks;
  uint64_t verify_fail;           /* chunks whose xxh64 did not match (open or inline) */
  uint64_t read_errors;           /* secondary read errors and short reads */
  uint64_t verified_bytes;        /* bytes hashed by the open-time verification of this process */
  uint64_t verify_ns;
  uint32_t disabled_mask;         /* bit i: location i disabled (at open or after errors) */
  uint32_t direct_mask;           /* bit i: location i is read unbuffered */
  int verify_mode;                /* the session's effective ss_mfile_verify */
  uint32_t reserved;
} ss_mfile_counters;
SS_API void ss_mfile_options_defaults_sized(ss_mfile_options* o, uint32_t size);
static inline void ss_mfile_options_defaults(ss_mfile_options* o) {
  ss_mfile_options_defaults_sized(o, (uint32_t)sizeof *o);
}
/* weight_dirs: ';'-separated directories (NULL / empty: the primary only). opt may be NULL (defaults). */
SS_API ss_status ss_mfile_open(const char* primary, const char* weight_dirs, const ss_mfile_options* opt,
                               ss_mfile** out);
SS_API void ss_mfile_close(ss_mfile* f);
SS_API uint64_t ss_mfile_size(const ss_mfile* f);
SS_API uint32_t ss_mfile_alignment(const ss_mfile* f);
/* 1 when at least one secondary location serves chunks */
SS_API int ss_mfile_multi(const ss_mfile* f);
SS_API ss_status ss_mfile_pread(ss_mfile* f, void* dst, uint64_t bytes, uint64_t offset, uint64_t* got);
/* Every io in flight at once (the latency path of small scattered reads); each io follows the read contract. */
SS_API ss_status ss_mfile_preadv(ss_mfile* f, const ss_mfile_io* ios, uint32_t n);
SS_API ss_status ss_mfile_get_counters(const ss_mfile* f, ss_mfile_counters* out);
/* The location's file path (0 = the primary), NULL past the last; valid until ss_mfile_close. */
SS_API const char* ss_mfile_location_path(const ss_mfile* f, int loc);
/* Why secondary locations were disabled at open ("" when none); valid until ss_mfile_close. */
SS_API const char* ss_mfile_open_report(const ss_mfile* f);
/* appended, bound opens: what a session holds, to compare with what the caller authenticated by its own handles.
 * Every file of a session (the primary, each stripe directory's manifest and data file) is resolved once to its real
 * final path (a link that exists then is followed once), then opened through held handles of its directory and of
 * every ancestor directory from the volume root: a directory or file found to be a symlink, junction or other reparse
 * point during that walk (swapped in after the resolve) is refused (the primary: the open fails; a secondary
 * location: disabled, with the reason in the open report). Windows holds those directories without
 * FILE_SHARE_DELETE (none can be renamed or deleted while the session lives) and the manifest and data files with
 * FILE_SHARE_READ (no writer, no rename); the leaf's final path must be its held directory's plus its name. Linux holds
 * the directory fds (openat O_NOFOLLOW walk from "/"; renames are not prevented, the identity checks above apply).
 * The manifest stays open for the session's lifetime and is read through its held handle. */
#define SS_MFILE_LOCATION_OPEN 1u     /* the location's data file is open (held) */
#define SS_MFILE_LOCATION_ENABLED 2u  /* and the location serves reads (not disabled at open or after errors) */
typedef struct ss_mfile_location {
  uint32_t struct_size;           /* set by the caller: sizeof(ss_mfile_location) */
  uint32_t flags;                 /* SS_MFILE_LOCATION_* */
  ss_file_identity data;          /* the held file this location reads (location 0: the primary); zero when none */
  ss_file_identity manifest;      /* the held manifest (zero for location 0 or when none is held) */
  ss_file_identity primary;       /* the session's primary (every location) */
  char manifest_digest[24];       /* the held manifest's "digest" field (16 lowercase hex digits), "" when none */
} ss_mfile_location;
/* loc: 0 = the primary .. ss_mfile_get_counters' locations - 1; SS_E_INVALID past the last. */
SS_API ss_status ss_mfile_location_info(const ss_mfile* f, uint32_t loc, ss_mfile_location* out);

/* Whole-device budgets (docs/design.md §5.1). One cap for the whole device (--vram-total-gib: as if the card had that
 * much memory) and one for the host (--ram-total-gib, pinned included). ss_budget_plan subtracts what is fixed (the
 * measured device context; the engine's named device components: trunk / dense weights, KV for the context,
 * workspaces, drafter; its pinned and pageable host components; the load staging) and gives the expert tiers what is
 * left: the arena gets the device's rest, the pinned tier the host's rest, the file tier the experts left over. Pure
 * integer arithmetic on bytes (saturating); no device call. With both caps 0 it is the engines' three-tier placement
 * unchanged (DeepSeek's tier_plan), so an engine moves to it with no change of behaviour. A cap that cannot hold the
 * fixed parts is refused (SS_E_CAPACITY, ss_last_error = out->refuse) naming the components and the smallest cap that
 * would hold them (min_vram_total_bytes / min_ram_total_bytes); the outputs are written on refusal too.
 * The ledger tracks what is really allocated against the same caps: every add is checked against its cap, and the
 * stats give the peak per kind with the components at that peak. Thread-safe. */
#define SS_BUDGET 1
#define SS_BUDGET_MARGINS 1            /* ss_budget_in.vram_margin_bytes / ram_margin_bytes */
#define SS_BUDGET_JSON 1               /* ss_budget_plan_json; ss_budget_out.device_room_bytes .. host_capped */
#define SS_BUDGET_UNIFIED 1            /* ss_budget_in.unified */
#define SS_BUDGET_OBSERVE 1            /* ss_budget_ledger_observe / _within_caps; ss_budget_stats.measured_host_last .. */
#define SS_BUDGET_NO_SAMPLE (~(uint64_t)0)  /* ss_budget_ledger_observe: nothing measured on that side */
#define SS_BUDGET_AUTO (~(uint64_t)0)  /* the "auto" value of the uint64_t knobs below */
#define SS_BUDGET_TOP 16               /* components kept per kind at its peak */
#define SS_BUDGET_NAME 48              /* bytes of a component name in the stats (longer names are cut) */
#define SS_BUDGET_REFUSE 256           /* bytes of ss_budget_out.refuse */
typedef enum ss_budget_kind {
  SS_BUDGET_DEVICE = 0,   /* device memory (cudaMalloc): the VRAM cap */
  SS_BUDGET_PINNED = 1,   /* page-locked host memory (cudaHostAlloc): the RAM cap */
  SS_BUDGET_PAGEABLE = 2  /* ordinary host memory: the RAM cap */
} ss_budget_kind;
#define SS_BUDGET_KINDS 3
typedef struct ss_budget_component {
  const char* name;     /* NUL-terminated, named in refusals; NULL = "unnamed" */
  uint64_t bytes;
} ss_budget_component;
typedef struct ss_budget_in {
  uint32_t struct_size;           /* sizeof(ss_budget_in), set by ss_budget_in_defaults */
  uint32_t file_tier;             /* 1 (default) = experts that fit nowhere else are read from their files */
  uint64_t vram_total_bytes;      /* the whole device's cap; 0 (default) = uncapped */
  uint64_t ram_total_bytes;       /* the host's cap, pinned included; 0 (default) = uncapped */
  uint64_t device_context_bytes;  /* the measured device context (CUDA context, library handles) */
  const ss_budget_component* device_fixed;    /* the engine's device components (trunk, KV, workspaces, drafter) */
  const ss_budget_component* pinned_fixed;    /* the engine's pinned host buffers (outside the expert tiers) */
  const ss_budget_component* pageable_fixed;  /* the engine's pageable host memory (the RAM cap only) */
  uint32_t n_device_fixed, n_pinned_fixed, n_pageable_fixed;
  uint32_t host_spares;           /* free pinned blocks for write-backs */
  uint64_t keys;                  /* routed experts (layers x experts) */
  uint64_t slot_bytes;            /* one expert's device slot */
  uint64_t unit_bytes;            /* one expert's pinned host stride (every host count is in these units) */
  uint64_t arena_spares;          /* free arena slots beside the residents */
  uint32_t staging;               /* the file tier's pinned staging buffers (default 8) */
  uint32_t reserved0;             /* explicit padding (never read) */
  uint64_t arena_units;           /* resident experts asked for; SS_BUDGET_AUTO (default) = what the VRAM cap allows
                                     (every expert when uncapped); never more than the cap allows */
  uint64_t ram_expert_bytes;      /* the pinned expert tier asked for (spares and staging included); SS_BUDGET_AUTO
                                     (default) = what fits */
  uint64_t inclusive_bytes;       /* clean copies of device experts, from RAM left over: 0 (default) = off,
                                     SS_BUDGET_AUTO = at most inclusive_auto_cap_bytes */
  uint64_t inclusive_auto_cap_bytes;  /* default 16 GiB */
  uint64_t host_avail_bytes;      /* host memory available now; SS_BUDGET_AUTO (default) = not limited */
  uint64_t host_margin_bytes;     /* kept free of host_avail_bytes (default 0) */
  uint64_t max_pinned_bytes;      /* engine + expert pinned bytes at most; SS_BUDGET_AUTO (default) = not limited */
  uint64_t load_staging_bytes;    /* the load's staging the host budget always adds (a fixed host part) */
  uint64_t min_resident;          /* the smallest arena (resident experts) a capped plan accepts (default 0) */
  /* SS_BUDGET_MARGINS: kept free under each cap for what is allocated after the plan (graph capture, warmup,
     per-request buffers, allocator slack). With a cap the arena gets cap - fixed parts - vram_margin_bytes, the
     pinned tier cap - host fixed parts - ram_margin_bytes; the minimum caps include them (an uncapped plan: only
     there). host_margin_bytes above is the margin of host_avail_bytes, not of the RAM cap. */
  uint64_t vram_margin_bytes;     /* default 0 */
  uint64_t ram_margin_bytes;      /* default 0 */
  /* SS_BUDGET_UNIFIED (appended): the device's memory IS host RAM (Strix Halo and successors). 1 = one pool: the cap
     is ram_total_bytes (or the smaller of both caps when vram_total_bytes is set too); the arena gets the pool minus
     the host fixed parts and ram_margin_bytes (and the device fixed parts and vram_margin_bytes), the pinned tier what
     the planned device bytes leave; min_vram_total_bytes / min_ram_total_bytes are then both the pool's minimum.
     Default 0 (two caps, unchanged); with no cap it changes nothing. */
  uint32_t unified;
  uint32_t reserved1;             /* explicit padding (never read) */
} ss_budget_in;
typedef struct ss_budget_out {
  uint32_t struct_size;           /* set by the caller; the library writes min(struct_size, sizeof) bytes */
  uint32_t refused;               /* 1 = the plan cannot run: refuse says why */
  uint32_t three_tier;            /* disk > 0: the file tier is on */
  uint32_t ram_auto;              /* ram_expert_bytes was auto */
  uint64_t resident, arena_spares, arena_slots;  /* arena_slots = resident + arena_spares */
  uint64_t pinned, disk;          /* non-resident experts in pinned blocks / read from their files */
  uint64_t inclusive;             /* spare blocks device experts may keep as clean copies */
  uint64_t spare_blocks;          /* host_spares + inclusive (host_spare_units) */
  uint64_t staging;               /* file-tier staging buffers (three-tier only) */
  uint64_t device_fixed_bytes;    /* context + device components */
  uint64_t device_planned_bytes;  /* device_fixed_bytes + arena_slots x slot_bytes */
  uint64_t host_fixed_bytes;      /* pinned + pageable components + load staging */
  uint64_t host_planned_bytes;    /* host_fixed_bytes + (pinned + spare_blocks + staging) x unit_bytes */
  uint64_t room_bytes;            /* pinned bytes the host can give the expert tiers (auto's ceiling) */
  uint64_t need_host_bytes;       /* available host memory the load needs (blocks + load staging + margin) */
  uint64_t pinned_peak_bytes;     /* engine + expert pinned bytes at peak */
  uint64_t min_vram_total_bytes;  /* context + device components + vram_margin_bytes + (arena_spares + min_resident)
                                     slots */
  uint64_t min_ram_total_bytes;   /* host_fixed_bytes + ram_margin_bytes + the blocks the smallest placement needs */
  char refuse[SS_BUDGET_REFUSE];  /* "" unless refused */
  /* SS_BUDGET_JSON (appended): */
  uint64_t device_room_bytes;     /* with a VRAM cap: cap - device_fixed_bytes - vram_margin_bytes (0: none left, or
                                     no cap) */
  uint64_t host_cap_units;        /* the pinned-tier ceiling the placement used, in units (with the file tier, or
                                     a RAM cap: pinned + spare blocks + staging never exceed it) */
  uint32_t device_capped;         /* 1 = the VRAM cap, not the arena request, sized the arena */
  uint32_t host_capped;           /* 1 = the RAM cap lowered the pinned ceiling below what the non-resident experts
                                     and the spares want */
} ss_budget_out;
SS_API void ss_budget_in_defaults_sized(ss_budget_in* in, uint32_t size);
static inline void ss_budget_in_defaults(ss_budget_in* in) { ss_budget_in_defaults_sized(in, (uint32_t)sizeof *in); }
/* SS_OK, or SS_E_CAPACITY when refused (out written either way); SS_E_INVALID for a bad argument. in is read up to
 * its struct_size (the rest defaulted). */
SS_API ss_status ss_budget_plan(const ss_budget_in* in, ss_budget_out* out);
/* The plan as JSON members (no braces), for a receipt: the caps and margins, the fixed components by name, the expert
 * geometry and requests, the placement, the planned bytes and the minimum caps (bytes, GiB rounded up to 0.01, and
 * the whole-GiB flag value), the refusal. in / out as passed to / returned by ss_budget_plan (each read up to its
 * struct_size). Writes at most len bytes (NUL-terminated when len > 0) and returns the length of the whole text
 * (0 when in or out is NULL). */
SS_API size_t ss_budget_plan_json(const ss_budget_in* in, const ss_budget_out* out, char* buf, size_t len);

typedef struct ss_budget_ledger ss_budget_ledger;
typedef struct ss_budget_entry {
  char name[SS_BUDGET_NAME];
  uint64_t bytes;
} ss_budget_entry;
typedef struct ss_budget_kind_stats {
  uint64_t cap;                   /* the cap this kind is charged against (PINNED / PAGEABLE: the RAM cap); 0 = none */
  uint64_t current, peak;
  uint32_t n_top, reserved;
  ss_budget_entry top[SS_BUDGET_TOP];  /* the components when this kind peaked, largest first */
} ss_budget_kind_stats;
typedef struct ss_budget_stats {
  uint32_t struct_size;           /* set by the caller; the library writes min(struct_size, sizeof) bytes */
  uint32_t refusals;              /* adds refused by a cap */
  ss_budget_kind_stats kind[SS_BUDGET_KINDS];  /* indexed by ss_budget_kind */
  uint64_t host_current, host_peak;            /* PINNED + PAGEABLE (what the RAM cap holds) */
  uint64_t measured_device_last, measured_device_peak;  /* ss_budget_ledger_observe_device */
  uint64_t observations;
  /* SS_BUDGET_OBSERVE (appended): the measured samples of ss_budget_ledger_observe (and observe_device) against the
     caps. observations counts the device samples. */
  uint64_t measured_host_last, measured_host_peak;
  uint64_t host_observations;
  uint32_t device_over, host_over;  /* samples above the VRAM / RAM cap (0 without a cap) */
  char measured_device_peak_at[SS_BUDGET_NAME];  /* the stage of the device peak ("" from observe_device) */
  char measured_host_peak_at[SS_BUDGET_NAME];
  uint32_t within_caps;             /* ss_budget_ledger_within_caps */
  uint32_t reserved1;
} ss_budget_stats;
/* Caps in bytes, 0 = uncapped. */
SS_API ss_status ss_budget_ledger_create(uint64_t vram_total_bytes, uint64_t ram_total_bytes, ss_budget_ledger** out);
SS_API void ss_budget_ledger_free(ss_budget_ledger* l);
/* Charges bytes to a named component of a kind (adds to one name accumulate). SS_E_CAPACITY when that would exceed
 * the kind's cap: nothing is charged and ss_last_error names the component. */
SS_API ss_status ss_budget_ledger_add(ss_budget_ledger* l, ss_budget_kind kind, const char* name, uint64_t bytes);
/* Releases bytes of a component; SS_E_INVALID (nothing changed) when that is more than it holds. */
SS_API ss_status ss_budget_ledger_sub(ss_budget_ledger* l, ss_budget_kind kind, const char* name, uint64_t bytes);
/* An externally measured device use (cudaMemGetInfo total - free, NVML), kept as the last and the peak. */
SS_API ss_status ss_budget_ledger_observe_device(ss_budget_ledger* l, uint64_t measured_used_bytes);
SS_API ss_status ss_budget_ledger_stats(const ss_budget_ledger* l, ss_budget_stats* out);
/* One measured sample at a named stage ("load", "warmup", "request:NAME", ...): the device use and the host use
 * (process commit, pinned included) the engine measured, SS_BUDGET_NO_SAMPLE for a side it cannot measure. Keeps the
 * last and the peak per side with the stage of the peak, and counts the samples above each cap. Charges nothing and
 * never fails because of a cap (the caller decides what a peak above a cap means). */
SS_API ss_status ss_budget_ledger_observe(ss_budget_ledger* l, const char* stage, uint64_t device_used_bytes,
                                          uint64_t host_used_bytes);
/* 1 when every measured peak is <= its cap and no sample was above it (no cap on a side = ok; no sample = ok); 0
 * otherwise or for NULL. The ledger's own charges never exceed the caps (an add over a cap is refused). Measurements
 * of the whole device (cudaMemGetInfo) include other processes' memory; the contexts never sample, they only charge their
 * own allocations - measured samples are the engine's (e.g. relative to a baseline taken before it allocates). */
SS_API int ss_budget_ledger_within_caps(const ss_budget_ledger* l);

/* Kept-resident layers (layer-wide prefill through the device-gated cache, docs/design.md §3.9). Engine thread,
   between tokens only, with the state rules of ss_dg_set_policy. Makes units ids[0..n) (distinct, < the layer's units)
   of gated layer g resident and kept: when it returns every one is in an arena slot, its entry published, its bytes the
   unit's file bytes (the cache decode uses, one copy of each unit: arena slot <- pinned block <- file). The kept set of
   g becomes exactly ids (replacing any keep of g); n = 0 releases, as ss_dg_keep(c, g, NULL, 0) does. Every keep stays
   protected during the call, g's current one included (replaced only on success). SS_E_CAPACITY, with nothing changed,
   when n + the other kept units (every layer's ss_dg_keep / make_resident set, g's current one included, distinct, not
   in ids) + the spare target > the persistent slots (residents + billed spares; the bank slots and the guard are never
   kept), when the slots that are free or hold an evictable unit cannot take the missing units, or (two-tier) when no
   free slot and no free pinned block exist to start with. Otherwise it drains every read, copy and write-back, then
   units of g already on the device stay, the others take free slots, else evict device units no layer keeps in the
   policy's order (LRU: least recent first; SCORED / SCORED_AUTO: the boundary's score order) with the boundary's
   lower-tier rules (a clean copy back to its block, a write-back into a free pinned block, three-tier: a pinned unit
   dropped to its file for that block, else the victim to its file), and copies the rest in from their pinned blocks or
   files (synchronously). Counted only in ss_dg_stats.resident_*. A held set is never exported: the next steps leave g
   out of the export / wait sequence (rows ignored) and read its kept ids through ss_dg_slot (any other id reads the
   guard and fails the token). Release it (ss_dg_keep(c, g, NULL, 0)) before g is exported again. Usage per layer:
   make_resident(g, all units), one step exporting no other layer (rows_per_layer[h] = 0 for h != g), release; several
   layers may be held at once. A read failure (SS_E_IO) leaves the cache consistent and every keep unchanged, its units
   in their slots (the units moved so far stay). A device failure (a copy, publication, event or synchronize:
   SS_E_BACKEND) poisons the context. */
#define SS_DG_MAKE_RESIDENT 1
SS_API ss_status ss_dg_make_resident(ss_ctx* c, int g, const int32_t* ids, int n);

/* Placement rounds (docs/design.md §3.10). Engine thread, between tokens, with the state rules of ss_dg_set_policy.
   Each gated unit has a decayed count s = decay x s + hits, hits = the steps that exported it since the previous round
   (its layer's unique routed ids, counted on the host: no device work). The rank by s is the target placement: the
   top (persistent slots - spare target - kept units) on the device, the next in the pinned tier, the rest in their
   files. A round moves units toward it, at most max_units promotions and budget_bytes bytes (file reads + host-to-
   device copies + write-backs; 0 = no limit), hottest first: (a) a pinned unit to the device, into a free slot while
   the device holds less than its share, else in place of the coldest device unit when its s > hysteresis x that
   unit's; the demoted unit goes back to its pinned block when it holds a clean copy (no write), else it is written back
   into a free pinned block (dg_wb_stream = 1: on the write-back stream, else the demand stream; three-tier with no
   free block: the coldest pinned unit is dropped to its file for it); (b) three-tier: a file unit read into the block
   of the coldest pinned unit (dropped to its file, no copy) when its s > hysteresis x that unit's, up to
   dg_disk_queue reads in flight. A unit with s = 0 is never promoted; units kept by ss_dg_keep or held by
   ss_dg_make_resident, units with a write-back in flight and prefetches still coming never move. Transactional as
   ss_dg_make_resident: a unit's home changes only after its copy and entry were accepted and the stream that copied
   it synced (a source block or slot is freed only then); a file read failure returns SS_E_IO with the cache
   consistent (the moves made so far stay); any device failure poisons the context (SS_E_BACKEND). The table is
   uploaded before it returns. Placement only: outputs are those of any other placement. Counted only in
   ss_dg_stats.round_* and the result. ss_dg_reset_open zeroes every count. */
#define SS_DG_ROUNDS 1
typedef struct ss_dg_round_params {
  uint32_t struct_size;  /* sizeof, set by ss_dg_round_defaults */
  uint32_t flags;        /* 0 (none defined) */
  uint64_t budget_bytes; /* default 0 = no byte limit */
  uint32_t max_units;    /* promotions per round, default 96 (0 = only the counts decay) */
  float hysteresis;      /* default 1.25 (>= 1) */
  float decay;           /* default 0.85 (0..1) */
  uint32_t reserved0;    /* explicit padding (never read) */
} ss_dg_round_params;
typedef struct ss_dg_round_result {
  uint32_t struct_size;
  uint32_t promoted_to_device, promoted_to_pinned, demoted_from_device, dropped_to_file;
  uint32_t cancelled;    /* 1: ss_dg_round_cancel stopped it at a unit boundary (no unit half moved) */
  uint64_t bytes_read, bytes_h2d, bytes_written_back, ns;
} ss_dg_round_result;
SS_API void ss_dg_round_defaults_sized(ss_dg_round_params* p, uint32_t size);
static inline void ss_dg_round_defaults(ss_dg_round_params* p) { ss_dg_round_defaults_sized(p, (uint32_t)sizeof *p); }
/* p NULL = the defaults; out may be NULL. The result is filled on success and on a failure after the round began. */
SS_API ss_status ss_dg_round_sized(ss_ctx* c, const ss_dg_round_params* p, ss_dg_round_result* out, uint32_t size);
static inline ss_status ss_dg_round(ss_ctx* c, const ss_dg_round_params* p, ss_dg_round_result* out) {
  return ss_dg_round_sized(c, p, out, (uint32_t)sizeof *out);
}
/* Any thread: the round in progress stops at its next unit boundary; a copy made but not committed is dropped (its
   source keeps its home) and the round returns SS_OK with cancelled = 1. No effect when no round is running. */
SS_API ss_status ss_dg_round_cancel(ss_ctx* c);
/* Engine thread, between tokens: the decayed counts of units[0..n) (finite, >= 0), e.g. from a shipped routing
   profile; every entry is checked before any is set. */
SS_API ss_status ss_dg_round_seed_counts(ss_ctx* c, const ss_dg_unit* units, const float* counts, uint32_t n);

/* Device profiles (docs/design.md §5.2, schema strix/device-profile-v1). ONE machine as the engine sees it: host RAM
 * (total, available, unified with the GPU or not), CPU cores, NVMe read rates, every GPU (api, name, arch, compute
 * units, wave, LDS, memory / GTT / visible bytes, memory and host-to-device bandwidth), the inter-host links (kind,
 * count, measured Gbps) and the margins (host floor, device reserve). Built by ss_devprof_detect: Linux sysfs / procfs
 * (KFD topology, amdgpu DRM mem_info, /proc/meminfo, Thunderbolt / USB4 domains; GPUs the runtime hides are dropped), the HIP runtime (hipMemGetInfo,
 * device properties) and on NVIDIA the CUDA runtime + NVML (both loaded at run time when present: the library needs
 * neither to build), then a device CLASS picked by api + arch + memory size gives what detection cannot know (class
 * name, conservative margins, measured bandwidths with their evidence ids, limits; matched on api + arch + name +
 * a memory range, so a qualified class never covers another arch or size: those get a generic class and a notice;
 * what a class filled is marked "class:..." in "evidence" and taken back when an override changes the class or device
 * 0's identity), then the override (--device-profile FILE / STRIX_DEVICE_PROFILE: a full or partial profile merged
 * over the detected one; "overrides" in the result records every merged path). The effective profile is canonical
 * JSON with a 16-hex-digit hash (xxh64 of that text) for receipts. Values a source could not give are 0 (JSON null for
 * bandwidths). Thread-safe for concurrent reads; nothing here allocates device memory. */
#define SS_DEVPROF 1
#define SS_DEVPROF_SCHEMA "strix/device-profile-v1"
#define SS_DEVPROF_NO_RUNTIME 1u  /* detect: sysfs / procfs only, no HIP / CUDA / NVML call */
#define SS_DEVPROF_NO_ENV 2u      /* detect: ignore STRIX_DEVICE_PROFILE / STRIX_DEVICE_PROFILE_ROOT */
#define SS_DEVPROF_NO_CACHE 4u    /* detect: neither read nor write the measurement cache */
typedef struct ss_devprof ss_devprof;
typedef struct ss_devprof_detect_opts {
  uint32_t struct_size;        /* sizeof, set by ss_devprof_detect_opts_defaults */
  uint32_t flags;              /* SS_DEVPROF_* (default 0) */
  const char* root;            /* filesystem root for /proc and /sys (tests: a fake tree); NULL (default) = "/", or
                                  $STRIX_DEVICE_PROFILE_ROOT */
  const char* override_path;   /* --device-profile FILE; NULL (default) = $STRIX_DEVICE_PROFILE (none when unset) */
  const char* cache_dir;       /* measured values cached per device identity; NULL (default) = %LOCALAPPDATA%\strix,
                                  $XDG_CACHE_HOME/strix or ~/.cache/strix */
} ss_devprof_detect_opts;
SS_API void ss_devprof_detect_opts_defaults_sized(ss_devprof_detect_opts* o, uint32_t size);
static inline void ss_devprof_detect_opts_defaults(ss_devprof_detect_opts* o) {
  ss_devprof_detect_opts_defaults_sized(o, (uint32_t)sizeof *o);
}
/* o NULL = the defaults. SS_E_IO / SS_E_RECIPE when the override file cannot be read / is not a valid profile
 * (nothing is returned); detection itself never fails (what it cannot read stays 0, with a notice). */
SS_API ss_status ss_devprof_detect(const ss_devprof_detect_opts* o, ss_devprof** out);
/* A whole profile from its JSON (schema must be strix/device-profile-v1; unknown keys are refused). */
SS_API ss_status ss_devprof_from_json(const char* text, size_t len, ss_devprof** out);
/* Merges a partial (or a whole, e.g. saved) profile over p: objects key by key, "devices" / "links" / "nvme" element
 * by element (index), "limits" / "evidence" by key, scalars replaced; "overrides" / "notices" / "sources" of the text
 * are ignored (p's provenance stays); the merged paths are appended to p's "overrides" (origin names the text). A
 * changed "class" or device 0 identity re-derives the class (see above). Atomic (a refused text changes nothing). */
SS_API ss_status ss_devprof_merge_json(ss_devprof* p, const char* text, size_t len, const char* origin);
SS_API void ss_devprof_free(ss_devprof* p);
/* Canonical JSON (malloc'd, free with free()) / its hash (17 bytes: 16 lowercase hex digits + NUL). */
SS_API ss_status ss_devprof_to_json(const ss_devprof* p, char** out, size_t* len);
SS_API ss_status ss_devprof_hash(const ss_devprof* p, char out[17]);
/* The probe report of the recipes (docs/recipes.md §8.2: devices with id / kind / name / memory_gib / host /
 * host_ram_gib / disk_read_gbytes_ps / h2d_gbytes_ps, plus class and arch) for one host named host. */
SS_API ss_status ss_devprof_probe_report(const ss_devprof* p, const char* host, char** out, size_t* len);

#define SS_DEVPROF_NAME 128
typedef struct ss_devprof_info {
  uint32_t struct_size;        /* set by the caller; the library writes min(struct_size, sizeof) bytes */
  uint32_t detected;           /* 1 = values come from this machine (0: a file / test profile only) */
  uint32_t unified;            /* 1 = the GPU's memory is host RAM (one pool: Strix Halo and successors) */
  uint32_t cpu_cores;
  uint32_t n_devices, n_links, n_nvme;
  uint32_t generic;            /* 1 = no known class matched (detected values only) */
  uint64_t ram_bytes, ram_available_bytes;
  uint64_t host_floor_bytes;   /* kept free of the host RAM (class margin) */
  uint64_t device_reserve_bytes;  /* kept free of each device's memory (class margin) */
  char class_name[64];
  char hash[17];
  char reserved[7];
} ss_devprof_info;
typedef struct ss_devprof_device {
  uint32_t struct_size;        /* set by the caller; the library writes min(struct_size, sizeof) bytes */
  ss_backend_kind api;         /* SS_BACKEND_HIP / SS_BACKEND_CUDA (SS_BACKEND_CPU: a host-only profile) */
  uint32_t compute_units;      /* CUs (HIP) / SMs (CUDA) */
  uint32_t wave;               /* wavefront / warp size */
  uint64_t lds_bytes;          /* LDS / shared memory per block */
  uint64_t memory_bytes;       /* what the device can allocate (HIP: hipMemGetInfo total, the GTT on an APU) */
  uint64_t gtt_bytes, visible_bytes;  /* amdgpu GTT / VRAM carve-out (0 on CUDA) */
  double mem_gbps;             /* device memory bandwidth, GB/s (0 = unknown) */
  double h2d_gbps;             /* pinned host -> device copies, GB/s (0 = unknown; unified: 0) */
  int32_t cc_major, cc_minor;  /* CUDA compute capability (0 on HIP) */
  int32_t pcie_gen, pcie_width;  /* NVML current link (-1 = unknown) */
  char name[SS_DEVPROF_NAME];
  char arch[32];               /* "gfx1151", "sm_120", ... */
  char id[64];                 /* GPU UUID / PCI bus id / KFD node */
} ss_devprof_device;
SS_API ss_status ss_devprof_get(const ss_devprof* p, ss_devprof_info* out);
/* A class limit engines read instead of a literal ("limits" in the JSON; e.g. max_piece_rows, uma_ceiling_bytes,
 * persist_blocks, long_vram_cap_bytes): today's value on the class it was qualified on. SS_E_INVALID (*value
 * untouched) when the profile has none: the engine keeps its own rule. */
SS_API ss_status ss_devprof_limit(const ss_devprof* p, const char* key, uint64_t* value);
/* SS_E_INVALID past the last device. */
SS_API ss_status ss_devprof_get_device(const ss_devprof* p, uint32_t i, ss_devprof_device* out);

/* The whole-device planner's caps from the profile (design §2): only the fields still at their defaults are written,
 * so an explicit --vram-total-gib / --ram-total-gib (or margins) always wins. Discrete device i: vram_total_bytes =
 * memory_bytes - device_reserve_bytes, ram_total_bytes = ram_available_bytes - host_floor_bytes; unified: one pool =
 * min(ram_available_bytes - host_floor_bytes, memory_bytes - device_reserve_bytes) as ram_total_bytes with
 * unified = 1 (vram_total_bytes stays 0). A size the profile does not know leaves its cap at 0 (uncapped). SS_E_INVALID
 * past the last device, and for a unified profile when in's struct_size predates the unified field (two independent
 * caps could each take the whole pool); SS_E_CAPACITY when the margins eat a known size (a zero cap would mean
 * "uncapped"). */
SS_API ss_status ss_devprof_budget(const ss_devprof* p, uint32_t device, ss_budget_in* in);

/* Single box vs pair (design §2): a planner output, not a literal. in: the model's fixed bytes at the requested
 * context (trunk, KV, workspaces; summed over the boxes), the smallest expert tier that still runs (summed), the
 * layer count; profiles[0] = this box, one profile per extra candidate box. Single box when profiles[0]'s budget
 * (ss_devprof_budget's cap) holds fixed + min experts; else world = the fewest boxes whose summed budgets hold it,
 * provided profiles[0] has a MEASURED link (links[] with count > 0 and gbps_measured > 0: discovery alone does not
 * qualify a pair); the layers split by budget share (box b starts at round-half-up of the cumulative share before it
 * x layers; equal boxes: ceil(layers / 2) on box 0). pinned_world (> 0) keeps a qualified recipe's choice and only
 * checks that it fits. */
typedef struct ss_devprof_topology_in {
  uint32_t struct_size;        /* sizeof, set by ss_devprof_topology_in_defaults */
  uint32_t layers;             /* the model's layers (> 0) */
  uint64_t fixed_bytes;        /* the whole model's fixed parts at the context (summed over the boxes) */
  uint64_t min_expert_bytes;   /* the smallest expert tier (summed over the boxes) */
  uint32_t pinned_world;       /* 0 (default) = decide; else keep it */
  uint32_t reserved0;          /* explicit padding (never read) */
} ss_devprof_topology_in;
#define SS_DEVPROF_MAX_WORLD 8
typedef struct ss_devprof_topology_out {
  uint32_t struct_size;        /* set by the caller; the library writes min(struct_size, sizeof) bytes */
  uint32_t world;              /* boxes to run on (1 = single box) */
  uint32_t fits;               /* 1 = the chosen world's budgets hold fixed + min experts */
  uint32_t reserved0;
  uint32_t first_layer[SS_DEVPROF_MAX_WORLD + 1];  /* box b runs layers [first_layer[b], first_layer[b+1]) */
  uint32_t reserved1;
  uint64_t budget_bytes[SS_DEVPROF_MAX_WORLD];     /* each box's cap the decision used */
  char why[256];
} ss_devprof_topology_out;
SS_API void ss_devprof_topology_in_defaults_sized(ss_devprof_topology_in* in, uint32_t size);
static inline void ss_devprof_topology_in_defaults(ss_devprof_topology_in* in) {
  ss_devprof_topology_in_defaults_sized(in, (uint32_t)sizeof *in);
}
/* SS_E_CAPACITY (out written, fits = 0) when no world up to n_profiles holds the model or pinned_world does not fit;
 * SS_E_INVALID for n_profiles 0 or > SS_DEVPROF_MAX_WORLD, layers 0 or >= 2^20, a pinned_world above n_profiles,
 * fixed_bytes + min_expert_bytes past 2^64. */
SS_API ss_status ss_devprof_topology(const ss_devprof* const* profiles, uint32_t n_profiles,
                                     const ss_devprof_topology_in* in, ss_devprof_topology_out* out);

/* The prefill piece (rows per piece) from the memory left after the plan (design §2): rows = min(floor(room /
 * bytes_per_row), the largest multiple of row_align <= max_rows (or <= 2^32 - 1 without max_rows)) rounded down to a
 * multiple of row_align; room = the budget cap of p's device (ss_devprof_budget's) - planned_bytes -
 * piece_reserve_bytes. The caller passes its own rule's bounds; on the class the rule was tuned on (the Halo for
 * today's auto pieces) the bounds bind, so its outputs are unchanged. A device whose memory the profile does not
 * know gets that aligned maximum (SS_E_CAPACITY without max_rows); a known one the margins exhaust gets SS_E_CAPACITY. */
typedef struct ss_devprof_piece_in {
  uint32_t struct_size;        /* sizeof, set by ss_devprof_piece_in_defaults */
  uint32_t device;             /* the profile's device (default 0) */
  uint64_t planned_bytes;      /* what the plan already placed in that budget */
  uint64_t bytes_per_row;      /* activation / workspace bytes one prompt row adds (> 0) */
  uint64_t piece_reserve_bytes;  /* kept free besides the piece (default 0) */
  uint32_t min_rows, max_rows; /* bounds (max_rows 0 = none) */
  uint32_t row_align;          /* default 1 */
  uint32_t reserved0;          /* explicit padding (never read) */
} ss_devprof_piece_in;
SS_API void ss_devprof_piece_in_defaults_sized(ss_devprof_piece_in* in, uint32_t size);
static inline void ss_devprof_piece_in_defaults(ss_devprof_piece_in* in) {
  ss_devprof_piece_in_defaults_sized(in, (uint32_t)sizeof *in);
}
/* *rows = the piece; SS_E_CAPACITY when even min_rows (or one aligned row) does not fit (*rows = 0); SS_E_INVALID
 * when no multiple of row_align lies in [min_rows, max_rows]. */
SS_API ss_status ss_devprof_piece_rows(const ss_devprof* p, const ss_devprof_piece_in* in, uint32_t* rows);
/* ---- Indexed prepared pack tierstream/pack-v1 (docs/pack.md) ----------------------------------------------------
 * One pack per model: every tensor / expert part in the engine's final device format (the bytes its loader would
 * produce before the host-to-device copy), whole, contiguous and 4 KiB aligned, in data files pack-NNNN.data, plus a
 * binary index pack.index and a manifest pack.json (source identities, format tags, data / index SHA-256). A record's
 * key is (layer, expert, name): layer -1 = global, expert -1 = a layer's dense tensor; a unit = the records of one
 * (layer, expert >= 0), adjacent in the file. The library never produces device bytes: the engine's repacker (or
 * ts-pack build from an engine-provided layout) writes them; the reader gives each app its records, row ranges and
 * units. An engine that finds no matching pack (or is told not to use one) keeps its original loader. */
#define SS_HAVE_PACK 1
typedef enum ss_pack_verify {
  SS_PACK_VERIFY_CACHED = 0, /* xxh64 of a record (or segment) the first time it is read with these data files; the
                              * verified set is kept in the user's cache directory by data-file identity (default) */
  SS_PACK_VERIFY_ALWAYS = 1, /* on every read */
  SS_PACK_VERIFY_OFF = 2     /* never (the operator verified the pack: ts-pack verify --full) */
} ss_pack_verify;
typedef struct ss_pack ss_pack;
typedef struct ss_pack_options {
  uint32_t struct_size;
  int verify;                        /* ss_pack_verify (default CACHED) */
  const char* manifest_sha256;       /* the pin: SHA-256 of pack.json (NULL / empty = not pinned) */
  const char* weight_dirs;           /* stripe directories of the data files (ss_options.weight_dirs rules) */
  uint32_t drive_readers;            /* default 4 */
  int direct_io;                     /* default 1 */
  const char* const* source_sha256;  /* the app's authenticated source SHA-256s, in pack.json order (NULL = no check) */
  uint32_t source_sha256_n;
  uint32_t layouts_n;
  const char* const* layouts;        /* optional layouts the app reads (e.g. "tp2-packed"); a pack built without one of
                                      * them is refused with SS_E_DESC naming the layouts it has */
} ss_pack_options;
SS_API void ss_pack_options_defaults_sized(ss_pack_options* o, uint32_t size);
static inline void ss_pack_options_defaults(ss_pack_options* o) { ss_pack_options_defaults_sized(o, (uint32_t)sizeof *o); }
/* Opens DIR: pack.json (pinned when asked, held share-read), pack.index (its SHA-256 from pack.json, every rule
 * validated), the data files through bound opens (and their stripes). A source SHA-256 mismatch is SS_E_DESC. */
SS_API ss_status ss_pack_open(const char* dir, const ss_pack_options* opt, ss_pack** out);
SS_API void ss_pack_close(ss_pack* p);
typedef struct ss_pack_entry {
  uint32_t struct_size;
  int32_t layer, expert;
  uint32_t file;           /* data file index */
  uint64_t index;          /* record index */
  const char* name;        /* valid while the pack is open */
  const char* format;
  uint64_t offset, bytes, xxh64;
  uint32_t rows;           /* 0 = not row-addressable */
  uint32_t segments;
  uint64_t row_bytes;      /* uniform rows; 0 with segments = non-uniform (read by whole segments) */
} ss_pack_entry;
typedef struct ss_pack_segment {
  uint32_t row0, row1;
  uint64_t offset, bytes, xxh64;  /* offset relative to the record */
} ss_pack_segment;
SS_API uint64_t ss_pack_records(const ss_pack* p);
SS_API ss_status ss_pack_entry_at_sized(const ss_pack* p, uint64_t index, ss_pack_entry* out, uint32_t size);
/* SS_E_INVALID when no record has the key. */
SS_API ss_status ss_pack_find_sized(const ss_pack* p, int layer, int expert, const char* name, ss_pack_entry* out,
                                    uint32_t size);
static inline ss_status ss_pack_find(const ss_pack* p, int layer, int expert, const char* name, ss_pack_entry* out) {
  return ss_pack_find_sized(p, layer, expert, name, out, (uint32_t)sizeof *out);
}
SS_API ss_status ss_pack_segment_at(const ss_pack* p, uint64_t record, uint32_t k, ss_pack_segment* out);
typedef struct ss_pack_unit {
  uint32_t file, reserved0;
  uint64_t offset, bytes;      /* the unit's contiguous span: its records and the zero padding between them */
  uint64_t first, count;       /* its records */
} ss_pack_unit;
SS_API ss_status ss_pack_unit_find(const ss_pack* p, int layer, int expert, ss_pack_unit* out);
/* Reads: row1 == 0 (and row0 == 0) = the whole record; else rows [row0, row1) (uniform rows: any range; non-uniform:
 * whole segments). All reads of one call are in flight together per data file; each is verified per the verify
 * mode (a range: by the segments it covers exactly, else through one read of the whole record). SS_E_HASH on a
 * mismatch (nothing is marked verified). Any thread; destinations must not overlap. struct_size = sizeof
 * (ss_pack_read_req), the same in every element of one call (the array is walked with it). */
typedef struct ss_pack_read_req {
  size_t struct_size;
  uint64_t record;
  uint32_t row0, row1;
  void* dst;
} ss_pack_read_req;
SS_API ss_status ss_pack_read(ss_pack* p, uint64_t record, uint32_t row0, uint32_t row1, void* dst);
SS_API ss_status ss_pack_readv(ss_pack* p, const ss_pack_read_req* reads, uint32_t n);
SS_API ss_status ss_pack_read_unit(ss_pack* p, int layer, int expert, void* dst);  /* unit.bytes into dst */
/* desc-v1 for the device-gated cache: base_json is the engine's own descriptor; the pack's data files are appended to
 * its sources and every part of a routed layer (part name = the record name of each unit) is pointed into the pack
 * (offset of unit 0, one stride; units in ascending order and at most 1024 sources in all, else SS_E_UNSUPPORTED).
 * The parts' bytes must equal the records'. A routed layer without unit hashes gets
 * them from the pack (verify CACHED / ALWAYS): its units' records are read once with the pack's verification and the
 * xxh64 of each unit's parts back to back is listed (cached per selection and data-file identities), so the cache's
 * verify_hash checks every landing. Canonical JSON; free with ss_free. */
SS_API ss_status ss_pack_dg_descriptor(ss_pack* p, const char* base_json, size_t len, char** out_json,
                                       size_t* out_len);
SS_API uint32_t ss_pack_files(const ss_pack* p);
/* 1 when the pack was built with that optional layout (pack.json "layouts"), else 0. */
SS_API int ss_pack_has_layout(const ss_pack* p, const char* layout);
/* The identity of data file i (for ss_options.source_identities with that descriptor). */
SS_API ss_status ss_pack_file_identity(const ss_pack* p, uint32_t file, ss_file_identity* out);
/* Manifest fields, per-format totals, files, verify mode and read counters (JSON; free with ss_free). */
SS_API ss_status ss_pack_info_json(const ss_pack* p, char** out_json, size_t* out_len);
/* Full check: every data file's SHA-256 and every record / segment xxh64 (JSON report; SS_E_HASH when any fails). */
SS_API ss_status ss_pack_verify_full(ss_pack* p, char** out_json, size_t* out_len);

/* Writer, in-process (engine repackers, ts-pack build). meta_json: {"model": str, "sources": [{"name", "size",
 * "sha256"}], "formats": {tag: {"transform", "version", "tool"}}, "split_bytes": u64 (0 = one data file; else a new
 * file starts at the first layer boundary past it), "layouts": [optional layouts this pack includes], "meta": {...},
 * "receipt": {...}}. Every index record is kind 0 (stored bytes); other kinds are reserved for later versions and a
 * v1 reader refuses them. Records go in unit-major order:
 * layers ascending (-1 first), a layer's dense records (expert -1) before its units, a unit's records adjacent. The
 * pack is built in DIR.incomplete and published by renaming it to DIR at finish; abort (or a failure) removes it. A
 * refused record spec leaves the writer as it was; any other error (I/O, a data-file split, finish) fails the build:
 * DIR.incomplete is removed at once and every later call except abort returns an error. */
typedef struct ss_pack_writer ss_pack_writer;
typedef struct ss_pack_record_spec {
  uint32_t struct_size;       /* sizeof; a smaller size of an older header gets the defaults of the fields it lacks */
  int32_t layer, expert;
  uint32_t rows;              /* 0 = not row-addressable */
  const char* name;
  const char* format;         /* a tag listed in meta_json.formats */
  uint64_t bytes;
  uint64_t row_bytes;         /* uniform rows (rows x row_bytes = bytes); 0 with segments = non-uniform */
  const uint32_t* seg_rows;   /* n_seg + 1 boundaries 0 < ... < rows (NULL = none) */
  const uint64_t* seg_bytes;  /* n_seg sizes (required for non-uniform rows; NULL otherwise) */
  uint32_t n_seg;
  uint32_t reserved0;
} ss_pack_record_spec;
SS_API ss_status ss_pack_writer_begin(const char* dir, const char* meta_json, size_t len, ss_pack_writer** out);
SS_API ss_status ss_pack_writer_record_begin(ss_pack_writer* w, const ss_pack_record_spec* r);
SS_API ss_status ss_pack_writer_record_write(ss_pack_writer* w, const void* data, uint64_t n);
SS_API ss_status ss_pack_writer_record_end(ss_pack_writer* w);
SS_API ss_status ss_pack_writer_add(ss_pack_writer* w, const ss_pack_record_spec* r, const void* data);
/* Publishes the pack; manifest_sha256 (65 bytes, may be NULL) receives the pin. The writer is freed either way. */
SS_API ss_status ss_pack_writer_finish(ss_pack_writer* w, char* manifest_sha256);
SS_API void ss_pack_writer_abort(ss_pack_writer* w);

#ifdef __cplusplus
}
#endif
#endif /* STRIX_STREAM_SS_H */
