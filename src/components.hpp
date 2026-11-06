// tierstream/components-v1 (placement contract §2.4): the engine's own account of every memory consumer for the
// configuration it was asked about, written by `<engine> --describe-components <flags>`. Separate from desc-v1, so
// descriptor hashes and plan ids never change. Integer bytes only; the planner never extrapolates from it.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "json.hpp"

namespace ts {

// Allocation lifetimes (contract §1 Phase), in this fixed order.
constexpr int kPlacePhases = 4;
const char* place_phase_word(int p);  // load, prefill, decode, spec

enum class ChargeAt { Device, HostPinned, HostPageable, File };
const char* charge_at_word(ChargeAt a);

struct Charge {
  std::string name;  // optional, unique within its component; "dg." names are the device-gated adapter's six
  ChargeAt at = ChargeAt::Device;
  uint64_t bytes = 0, peak_extra = 0;  // peak_extra: bounded transient on top of bytes inside its phases
  unsigned phases = 0;                 // bit p = place_phase_word(p)
  // separate allocations behind the charge (rev 17; default 1, 0 = carved from another allocation, e.g. the window
  // slots inside the arena): the planner bills G x allocs on a device and 4,096 B x allocs on pinned host
  uint64_t allocs = 1;  // default 1 for payload, 0 for driver charges (rev 18)
  bool driver = false;  // kind (rev 18): "payload" (default) | "driver" (driver-materialised; the same arithmetic)
};

struct Receipt {  // qualified[]: bit-exact receipt of a movable component on one arch for one build
  std::string arch, build_sha256, receipt_sha256, content_sha256, arithmetic_sha256;
  std::vector<std::string> variants;  // graph / eager (empty = not stated)
};

struct ComponentDecl {
  std::string id;
  int priority = 0;  // contract §1.1 (component_priority)
  bool has_variants = false;
  std::vector<Charge> charges;          // location-independent
  bool has_anchor = false, has_assist = false;
  std::vector<Charge> anchor, assist;   // variants
  std::vector<Receipt> qualified;
  bool requires_relay = false;
  // Charges for a destination role: "anchor" for the anchor (and host destinations), "assist" off it.
  const std::vector<Charge>* for_role(bool assist_role) const;
};

struct Components {
  std::string engine, build_sha256, model_sha256, desc_sha256;
  Json content, arithmetic;  // identity objects (null when absent), canonical
  // elastic metadata (S3); absent: nothing elastic is planned
  bool has_elastic = false;
  uint64_t unit_bytes = 0, units_total = 0, per_layer_units = 0, spares = 0, host_spares = 0;
  uint64_t slot_allocs = 1;  // elastic.slot_allocs (rev 17): the arena's separate allocations (one per unit part)
  uint64_t host_chunk_bytes = uint64_t(1) << 30;  // elastic.host_chunk_bytes (rev 18): expert homes' chunk size
  uint64_t per_layer_max_active = 0;  // 0 = no per-layer limit (DS: one global arena)
  bool exclusive = true;  // elastic.mode: "exclusive" (default; host homes partition the units) | "inclusive"
  // config: the allocation-affecting options the document was computed for
  int64_t max_context_tokens = 0, max_sequences = 0, max_rows = 0, chunk_rows = -1;
  std::string cache;  // config.cache: "tierstream" (TierSite, device-gated) | "engine" (the engine's own cache) | ""
  bool device_gated() const { return cache == "tierstream"; }
  Json config;
  Json cache_config;  // config.cache_config (null = absent; required in placement mode, §3.9 cache_sha256)
  // envelope (rev 18): everything the context observations depend on {shapes, limits, pools, allocator, dispatch,
  // cache_config}; a context receipt admits only when its envelope_sha256 equals envelope_sha256 below
  Json envelope;
  Json elastic_doc;  // the elastic object as written (cache_sha256 hashes it, golden/calib_ref.py)
  Json stack;  // stack (rev 18): the build's {stack_bytes, dynamic_stack} (kept as written)
  std::string envelope_sha256;  // sha256 of the canonical envelope ("" = no envelope)
  std::vector<std::string> capabilities;  // sorted, distinct
  // per arch: the engine's declared allowance (static per build; never a measured value: context-v1 carries those)
  std::map<std::string, uint64_t> context, context_growth;
  // context_receipts (rev 15/16): one per complete key (arch, device name, driver, platform, build, signature) the allowance was
  // qualified on; a key field missing or empty and a duplicate key are PL_E_SCHEMA; max_growth_bytes = the residual
  // growth measured after the checkpoint (a receipt above context_growth[arch] does not admit, §2.4)
  struct ContextReceipt {
    std::string arch, device_name, driver, platform, build_sha256, signature, method, receipt_sha256;
    uint64_t allowance_bytes = 0, max_observed_bytes = 0, max_growth_bytes = 0, runs = 0;
    uint64_t alloc_granularity_bytes = 0;  // G (rev 17): the qualified per-allocation overhead bound
    std::string envelope_sha256;           // the configuration envelope the receipt was qualified on (rev 18)
    uint64_t host_alloc_granularity_bytes = 4096;  // qualified pinned overhead per allocation (required, rev 19)
    // warm_bytes (rev 19): the qualified warm-up materialisation per component on a device of this key; every driver
    // charge placed there must equal its entry
    std::map<std::string, uint64_t> warm_bytes;
  };
  std::vector<ContextReceipt> context_receipts;  // canonical order: (arch, device_name, driver, platform, build, signature)
  bool has_spare_billing = false, spare_billing_per_layer = false;  // elastic.spare_billing: global | per_layer
  std::vector<ComponentDecl> components;  // sorted by priority
  std::string receipts_sha256;            // the tierstream/receipts-v1 file the receipts were copied from ("" = none)
  std::string sha256;                     // components_sha256: sha256 of the canonical JSON

  const ComponentDecl* find(const std::string& id) const;
  bool has(const std::string& capability) const;
};

// Strict parse (unknown keys, types, ranges, duplicate ids / arches / capabilities / identical charges, empty phase
// lists, 64-bit overflow of any sum are refused: SS_E_RECIPE "components: PL_E_SCHEMA ...").
Components components_from_json(const Json& doc);
// Canonical form (§2.4): components by priority, charges by (at, phases, bytes, peak_extra, name; no name first),
// qualified by arch,
// capabilities sorted, phases in lifetime order, every default written; components_sha256 hashes json_write of it.
Json components_to_json(const Components& c);

// tierstream/context-v1: attributable per-process context bytes, per {arch, device_name, driver, platform,
// build_sha256} ({"schema", "entries": [{arch, device_name, driver, platform, build_sha256, samples: [{bytes,
// method}]}]}, samples oldest first). S2 charges max(the qualified allowance, the aggregate) + context_growth[arch];
// the aggregate is the median of the latest 5 samples (lower middle for an even count), ignored with fewer than 3,
// refused above 2 x the allowance (PL_E_STALE "context outlier").
struct ContextSample {
  uint64_t bytes = 0;
  std::string method;  // the attributable method (NVML per-process, PDH process dedicated, amdsmi)
};
struct ContextEntry {
  std::string build_sha256, arch, device_name, driver, platform;
  std::vector<ContextSample> samples;
};
int64_t context_aggregate(const ContextEntry& e);  // -1 = fewer than 3 samples
struct ContextDoc {
  std::vector<ContextEntry> entries;  // sorted, distinct keys
  std::string sha256;                 // context_sha256 of the canonical form
  // The entry whose five key fields all equal these (exact match); nullptr = none.
  const ContextEntry* find(const std::string& build, const std::string& arch, const std::string& device_name,
                           const std::string& driver, const std::string& platform) const;
};
ContextDoc context_from_json(const Json& doc);

// tierstream/usage-v1 (contract §3.1, §2.7): the producer's shape (strix-stream-adaptive adapt_usage.cpp):
// {"schema", "model", "sources": [{path, sha256, kind, filter, tokens}], "decay": {factor_q32, half_life_tokens},
// "tick", "tokens", "order", "layers": [{index, n_units, count[], use[], recency[]}]}; use = micro-selections per token
// (rdiv(count x 10^6, tokens), checked), recency = the decayed Q32 value. Rank: use desc, recency desc, layer asc,
// unit asc. Dense layers carry n_units 0; `layer` below is the ordinal among the layers with units (the streamable MoE
// layers), `model_layer` the file's index. A per-language table has the same shape with sources[].filter = <lang>.
struct UsageUnit {
  int64_t layer = 0, model_layer = 0, unit = 0, count = 0, selections = 0, recency = 0;
};
struct UsageDoc {
  std::string model, filter;          // filter: the sources' common filter ("" = pooled or mixed)
  std::vector<std::string> sources;   // the sources' sha256 (trace identities; §3.9 FILE-rate source rules)
  int64_t tokens = 0;
  std::vector<int64_t> layer_units;   // n_units per file layer (0 = dense)
  std::vector<UsageUnit> units;       // every unit, layer-major, unit ascending
  std::string sha256;                 // usage_sha256: sha256 of the document as the producer writes it (json_write)
  std::vector<size_t> rank() const;   // indices into units, hottest first (the producer's order)
};
UsageDoc usage_from_json(const Json& doc);

}  // namespace ts
