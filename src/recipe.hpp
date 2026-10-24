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

namespace ss {

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
enum class FieldType { Gib, Float, Int, Bool, Policy, TierMode };
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
  double get(const char* key) const;
};

struct RWhen {
  int64_t prompt_tokens_min = 0, prompt_tokens_max = kNoLimit, batch_min = 1, batch_max = kNoLimit;
  bool always() const {
    return prompt_tokens_min == 0 && prompt_tokens_max == kNoLimit && batch_min == 1 && batch_max == kNoLimit;
  }
};

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
  const RPlacement* placement(const std::string& set) const;
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
};
struct Binding {
  std::vector<std::pair<std::string, std::vector<BoundMember>>> sets;  // recipe set order
  const std::vector<BoundMember>* of(const std::string& set) const;
};

// Hardware check: throws SS_E_RECIPE listing every unmet requirement; returns the binding.
Binding recipe_bind(const Recipe& r, const Json& probe);
Json binding_json(const Binding& b);
// Admission: strix-stream/recipe-select-v1. Throws SS_E_RECIPE when no route can be admitted.
Json recipe_select(const Recipe& r, const std::string& effective_sha256, const Json& state);

struct RecipePlan {
  ss_budget budget{};
  ss_options options{};
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

}  // namespace ss
