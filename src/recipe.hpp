// Recipes strix-stream/recipe-v1 (docs/recipes.md): how one model runs on one hardware set per workload class.
// recipe.cpp: parse + strict validation + canonical form. recipe_resolve.cpp: paths, profile / override layers,
// explain. recipe_run.cpp: hardware match, admission (route selection), compilation to ss_budget / ss_options /
// plan-v1.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "desc.hpp"
#include "json.hpp"
#include "plan.hpp"
#include "strix_stream/ss.h"

namespace ts {

enum class Phase { Prefill, Decode };
enum class Execute { Single, Tp, Replicas, Hetero };
enum class Rank { Model, List, Frequency };
enum class Rest { Disk, Pinned, Peer };

constexpr int64_t kFill = -1;     // resident / pinned = "fill"
constexpr int64_t kNoLimit = -1;  // when.*_max absent

struct RDevice {
  std::string id;
  ss_backend_kind kind = SS_BACKEND_CPU;
  int count = 1;
  std::string name;  // empty = any
  double memory_min_gib = 0, host_ram_min_gib = 0, disk_read_gbytes_ps_min = 0, h2d_gbytes_ps_min = 0;
  bool has_internal_link = false;
  std::string internal_kind = "other";
  double internal_gbps_min = 0;
  // Placement selectors (docs/recipes.md §3.2), each written to the canonical form only when set. An optional set
  // is bound after the others and may stay unbound; uuid / pci equal the probe's normalized identity; index is the
  // position among the host's probed devices of the same kind in PCI order (-1 = not set).
  bool optional = false;
  std::string uuid, pci;
  int64_t index = -1;
  bool unified() const { return kind != SS_BACKEND_CUDA; }
};

struct RLink {
  std::string id, a, b, kind = "other";
  double gbps_min = 0;
};

struct RGroup {
  std::string id;
  LayerKind cls = LayerKind::Moe;
  std::vector<std::pair<int, int>> layers;  // inclusive ranges; empty = every layer of the class
  Rank rank = Rank::Model;
  std::vector<int> list;
  int64_t resident = kFill, pinned = 0;
  Rest rest = Rest::Disk;
  std::string peer;
};

// Scalar knobs of a placement (docs/recipes.md §5.1), table-driven: parse, canonical form and settable paths share it.
enum class FieldType { Gib, Float, Int, Bool, Policy, TierMode, WeightDirs, WeightVerify };
constexpr double kTierInclusive = 0, kTierExclusive = 1;
struct PlacementField {
  const char* key;
  FieldType type;
  double lo, hi, dflt;
};
const std::vector<PlacementField>& placement_fields();

struct RPlacement {
  std::string set;
  std::vector<double> v;  // per placement_fields(); bools 0/1, policy = ss_policy
  bool has_workspace = false;
  std::vector<RGroup> groups;
  std::string weight_dirs;
  double get(const char* key) const;
};

struct RWhen {
  int64_t prompt_tokens_min = 0, prompt_tokens_max = kNoLimit, batch_min = 1, batch_max = kNoLimit;
  bool always() const {
    return prompt_tokens_min == 0 && prompt_tokens_max == kNoLimit && batch_min == 1 && batch_max == kNoLimit;
  }
};

// Placement mode (docs/recipes.md §5.4, contract placement-v1): per-route assists and component destinations.
struct RAssist {  // routes.<id>.assist.<set>
  std::string set;
  double device_gib = 0, margin_gib = 0.5, margin_fraction = 0.03;  // device_gib 0 = disabled
};
struct RComponent {  // routes.<id>.components.<component>
  std::string id;
  std::string device = "auto";  // a set id (anchor or assist) | host_pinned | host_pageable | file | auto
  bool has_gib = false;
  double gib = 0;    // elastic ids only
  int64_t units = -1;  // elastic ids only: an exact unit count (exclusive with gib; -1 = not set)
  bool has_layers = false;  // reserved: refused by the planner (PL_E_CAPABILITY)
  std::vector<std::pair<int, int>> layers;
};

// The closed list of component ids, in priority order (contract §1.1): ties and output orders follow it.
struct ComponentInfo {
  const char* id;
  const char* dflt;  // "anchor" | "host_pinned" | "host_pageable" | "file" | "" (no default: capability only)
  bool elastic;      // takes `gib` / `units`
};
const std::vector<ComponentInfo>& component_table();
int component_priority(const std::string& id);  // 1-based; 0 = not a component id
bool reserved_destination(const std::string& word);  // auto, host_pinned, host_pageable, file
// Probe identities, normalized: "GPU-" + lowercase hex; dddd:bb:dd.f lowercase ("" when not a PCI address).
std::string normalize_uuid(const std::string& s);
std::string normalize_pci(const std::string& s);

struct RRoute {
  std::string id;
  Phase phase = Phase::Prefill;
  std::vector<std::string> devices;
  Execute execute = Execute::Single;
  std::string handoff_route;  // empty = none
  std::vector<std::string> links;
  RWhen when;
  std::string fallback;  // empty = none
  bool enabled = true;   // false: admits no request (docs/recipes.md §5)
  std::vector<RPlacement> placements;
  // Placement mode: 0 = not set (max_rows defaults to 1, max_sequences to the anchor placement's max_batch).
  int64_t max_rows = 0, max_sequences = 0;
  std::vector<RAssist> assists;        // sorted by set id
  std::vector<RComponent> components;  // sorted by priority
  const RPlacement* placement(const std::string& set) const;
  const RAssist* assist(const std::string& set) const;
  const RComponent* component(const std::string& cid) const;
  // A route with an assist or components table plans through ts_place (the engine's --placement opts any route in).
  bool placement_mode() const { return !assists.empty() || !components.empty(); }
};

struct Recipe {
  std::string name, description, generated_by = "human";
  std::vector<std::string> pin;
  std::string model, model_sha256, engine;
  std::vector<RDevice> devices;
  std::vector<RLink> links;
  double host_free_ram_floor_gib = 8.0;
  double host_free_commit_floor_gib = 0;  // 0 = commit not checked
  std::vector<RRoute> routes;

  const RDevice* device(const std::string& id) const;
  const RLink* link(const std::string& id) const;
  const RRoute* route(const std::string& id) const;
};

// Text -> document tree: JSON when the first non-blank character is '{', else TOML. Throws SS_E_RECIPE.
Json recipe_doc_parse(const std::string& text);

// Strict parse of a recipe document. `note(path)` is appended to an error about that path (where its value came
// from); may be null. Throws SS_E_RECIPE.
struct NoteFn {
  std::string (*fn)(const void* user, const std::string& path) = nullptr;
  const void* user = nullptr;
};
Recipe recipe_from_json(const Json& doc, NoteFn note = {});
// The recipe with every default written out, canonical (ss_recipe_to_json, explain).
Json recipe_to_json(const Recipe& r);

uint64_t gib_bytes(double gib);  // floor(gib x 2^30 + 0.5), gib validated in [0, 2^20]

// ---- paths and layers (recipe_resolve.cpp) -------------------------------------------------------------------
// Throws unless `path` is a settable path (docs/recipes.md §6.2) whose ids exist in r; with allow_prefix a segment
// prefix of one is accepted (pin entries).
void check_path(const Recipe& r, const std::string& path, bool allow_prefix);
// Writes value at a settable path of a recipe document (creates a placement table when needed).
void set_path(Json& doc, const std::string& path, const Json& value);
// True when pin entry `pin` covers `path` (equal, or a segment prefix).
bool pin_covers(const std::string& pin, const std::string& path);

struct RecipeLayer {
  enum Kind { Profile, Override } kind = Override;
  std::string origin, source;
  std::vector<std::pair<std::string, Json>> set;
};

// A recipe plus its profile / override layers and the effective recipe they make (every change is atomic).
class RecipeStack {
 public:
  RecipeStack(const std::string& text, const std::string& origin);
  void add_profile(const std::string& text, const std::string& origin, const Json& probe);
  void add_overrides(const std::string& text, const std::string& origin);
  void set(const std::string& path, const std::string& value_text, const std::string& origin);

  const Recipe& effective() const { return eff_; }
  Json effective_json() const { return recipe_to_json(eff_); }
  // sha256 of the canonical effective recipe: binds compiled requirements and running credits to it (§7.3).
  const std::string& effective_sha256() const { return eff_sha_; }
  Json explain() const;
  const std::string& sha256() const { return sha_; }

 private:
  std::string origin_, sha_, eff_sha_;
  Json doc_;     // the recipe as written
  Recipe base_;  // the recipe alone
  std::vector<RecipeLayer> layers_;
  Recipe eff_;
  void rebuild(const std::vector<RecipeLayer>& layers);
  RecipeLayer read_layer(const std::string& text, const std::string& origin, RecipeLayer::Kind kind,
                         const Json* probe) const;
};

// ---- request time (recipe_run.cpp) ----------------------------------------------------------------------------
struct BoundMember {
  std::string id, host, driver;
  int64_t ordinal = -1;
  double disk_read_gbytes_ps = -1, h2d_gbytes_ps = -1;  // -1 = not probed
  // Physical identity (placement): phys = PCI address if probed, else UUID, else the probe id.
  std::string kind, name, uuid, pci, phys, arch;
  std::string platform;  // probe row "platform", else the report's (context-v1 key), "" = not probed
  double memory_gib = 0;
  int64_t memory_bytes = -1;  // probed exact bytes, -1 = not probed (gib_bytes(memory_gib) then)
};
struct Binding {
  // Recipe set order; an unbound optional set has no members.
  std::vector<std::pair<std::string, std::vector<BoundMember>>> sets;
  const std::vector<BoundMember>* of(const std::string& set) const;
};

// Hardware check: throws SS_E_RECIPE listing every unmet requirement; returns the binding.
Binding recipe_bind(const Recipe& r, const Json& probe);
Json binding_json(const Binding& b);
// Admission: strix-stream/recipe-select-v1. Throws SS_E_RECIPE when no route can be admitted.
Json recipe_select(const Recipe& r, const std::string& effective_sha256, const Json& state);

struct RecipePlan {
  ss_budget budget{};
  ss_options options{};  // weight_dirs borrows the recipe placement string until open copies it
  Plan plan;
  Json doc;  // strix-stream/recipe-plan-v1
};
struct RefInput {  // member 0 of one route set, for hetero ownership
  const Desc* desc = nullptr;
  std::vector<double> benefit;
};
// desc / benefit: the member being planned (benefit per global unit of desc, or empty). refs: indexed like the
// route's devices; required in hetero routes, ignored otherwise.
RecipePlan recipe_plan(const Recipe& r, const std::string& effective_sha256, const std::string& route,
                       const std::string& set, int member, const std::string& device_id, const Desc& desc,
                       const std::vector<double>& benefit, const std::vector<RefInput>& refs);

}  // namespace ts
