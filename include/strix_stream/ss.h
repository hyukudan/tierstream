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
 *   into a free block (device to host, on a second copy stream), or falls back to its file when no block is free. With
 *   device + host room for every unit, files are read only at open. */
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
   * the stream ring and its victim stays (counted as deferred); blocks freed by promotions are reusable from the next
   * forward. 0 = auto: min(64, units of that size). */
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
  SS_POLICY_SCORED = 1   /* design.md §3.2: 0.7 transition + 0.2 window frequency + 0.1 recency */
} ss_policy;

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
  /* Host read-ahead (design: wave4/gapfix/wstream-readahead): pinned host buffers, each as large as the largest unit
   * that can be read from its file. ss_host_prefetch* and units waiting for a ring slot read their file into one
   * ahead of time; the unit then lands in its ring slot from that buffer. 0 = off (default), at most 64. */
  uint32_t readahead_units;
  uint32_t reserved1;             /* explicit padding (never read) */
} ss_options;

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
 * rows_per_layer[L] == 0 marks a layer the forward skips (e.g. a cone). */
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
} ss_exclusive_stats;
SS_API ss_status ss_stats_exclusive_sized(const ss_ctx* c, ss_exclusive_stats* out, uint32_t size);
static inline ss_status ss_stats_exclusive(const ss_ctx* c, ss_exclusive_stats* out) {
  return ss_stats_exclusive_sized(c, out, (uint32_t)sizeof *out);
}

/* Stats snapshot (design.md §7.1): the one exception to the threading rule. ss_stats_snapshot* may be called from any
 * thread, concurrently with any call on the context and with each other, but never concurrently with or after
 * ss_close (stop the readers first). They work on a poisoned context and return the last publication.
 *
 * The first call arms the context and returns SS_E_INVALID (outputs untouched): from then on the engine thread
 * publishes an immutable copy of the counters at the end of each ss_end that folds a forward's numbers (also one that
 * reports a read failure) and of each ss_stats_reset between forwards. Until the first publication every call
 * returns SS_E_INVALID. Unarmed, the only added work is one relaxed load per ss_end / ss_stats_reset. A publication
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
  /* EXCLUSIVE tier (0 unless SS_SNAPSHOT_EXCLUSIVE), as ss_exclusive_stats plus the block count. */
  uint64_t exclusive_writebacks, exclusive_writeback_bytes, exclusive_deferred, exclusive_promoted;
  uint64_t exclusive_free_blocks, exclusive_min_free_blocks, exclusive_rereads, exclusive_blocks;
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

#ifdef __cplusplus
}
#endif
#endif /* STRIX_STREAM_SS_H */
