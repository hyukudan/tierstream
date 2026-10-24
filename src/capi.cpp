// C ABI: every entry point catches, maps to ss_status and keeps the message per thread.
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>

#include "desc.hpp"
#include "errors.hpp"
#include "plan.hpp"
#include "recipe.hpp"
#include "stream_ctx.hpp"
#include "strix_stream/ss.h"

struct ss_desc {
  ss::Desc d;
};
struct ss_plan {
  ss::Plan p;
  const ss_desc* desc;
};
struct ss_ctx {
  std::unique_ptr<ss::Ctx> c;
};
struct ss_recipe {
  ss::RecipeStack s;
};

namespace {

thread_local std::string g_err;

template <class F>
ss_status guard(F&& f) {
  try {
    f();
    return SS_OK;
  } catch (const ss::Error& e) {
    g_err = e.what();
    return e.status();
  } catch (const std::bad_alloc&) {
    g_err = "out of host memory";
    return SS_E_NOMEM;
  } catch (const std::exception& e) {
    g_err = e.what();
    return SS_E_BACKEND;
  }
}

ss_status give_text(const std::string& s, char** out, size_t* len) {
  if (!out) {
    g_err = "null output pointer";
    return SS_E_INVALID;
  }
  char* p = static_cast<char*>(std::malloc(s.size() + 1));
  if (!p) return SS_E_NOMEM;
  std::memcpy(p, s.data(), s.size());
  p[s.size()] = 0;
  *out = p;
  if (len) *len = s.size();
  return SS_OK;
}

void need(bool ok, const char* what) { ss::require(ok, SS_E_INVALID, what); }

std::string read_text(const char* path, const char* what) {
  std::ifstream f(path, std::ios::binary);
  ss::require(bool(f), SS_E_IO, std::string("cannot open ") + what + " " + path);
  std::stringstream ss_;
  ss_ << f.rdbuf();
  return ss_.str();
}

// JSON inputs of the recipe calls (probe report, identity, request state).
ss::Json parse_input(const char* text, size_t len, const char* what) {
  try {
    return ss::json_parse(std::string(text, len));
  } catch (const ss::Error& e) {
    ss::fail(SS_E_RECIPE, std::string(what) + ": " + e.what());
  }
}

void budget_defaults(ss_budget* b) {
  std::memset(b, 0, sizeof *b);  // tier_mode INCLUSIVE, host_spare_units auto
  b->struct_size = uint32_t(sizeof *b);
  b->margin_gib = 2.0;
  b->margin_fraction = 0.03;
}

void options_defaults(ss_options* o) {
  std::memset(o, 0, sizeof *o);
  o->struct_size = uint32_t(sizeof *o);
  o->backend = SS_BACKEND_CPU;
  o->policy = SS_POLICY_LRU;
  o->lookahead_budget = 2;
  o->reader_threads = 2;
  o->staging_buffers = 4;
  o->staging_bytes = uint64_t(256) << 20;
  o->copy_piece_bytes = uint64_t(64) << 20;
  o->fill_residents = 1;
  o->score_window = 16;
  o->pin_planned = 1;
  o->direct_io = 1;
}

// Bytes that were padding in older headers carry whatever the caller's compiler left there: never read.
void clear_reserved(ss_budget& b) { b.reserved0 = 0; }
void clear_reserved(ss_options&) {}

// Versioned structs: start from this build's defaults and take min(struct_size, sizeof) bytes from the caller
// (an older caller's missing fields keep their defaults; a newer caller's extra fields are not read).
template <class T>
T versioned(const T* in, void (*defaults)(T*), const char* what) {
  T v;
  defaults(&v);
  if (!in) return v;
  ss::require(in->struct_size >= 2 * sizeof(uint32_t), SS_E_INVALID,
              std::string(what) + ": struct_size not set (fill the struct with its *_defaults function)");
  std::memcpy(static_cast<void*>(&v), in, std::min<size_t>(in->struct_size, sizeof(T)));
  v.struct_size = uint32_t(sizeof(T));
  clear_reserved(v);
  return v;
}

// Defaults into the caller's first `size` bytes only.
template <class T>
void sized_defaults(T* out, uint32_t size, void (*defaults)(T*)) {
  if (!out || size < sizeof(uint32_t)) return;
  T v;
  defaults(&v);
  v.struct_size = std::min<uint32_t>(size, uint32_t(sizeof(T)));
  std::memcpy(static_cast<void*>(out), &v, v.struct_size);
}

}  // namespace

extern "C" {

const char* ss_last_error(void) { return g_err.c_str(); }
int ss_abi_version(void) { return SS_ABI_VERSION; }
void ss_free(void* p) { std::free(p); }

ss_status ss_desc_from_json(const char* json_text, size_t len, ss_desc** out) {
  return guard([&] {
    need(json_text && out, "ss_desc_from_json: null argument");
    auto* d = new ss_desc{ss::desc_from_json(ss::json_parse(std::string(json_text, len)))};
    *out = d;
  });
}

ss_status ss_desc_from_file(const char* path, ss_desc** out) {
  return guard([&] {
    need(path && out, "ss_desc_from_file: null argument");
    std::ifstream f(path, std::ios::binary);
    ss::require(bool(f), SS_E_IO, std::string("cannot open descriptor ") + path);
    std::stringstream ss_;
    ss_ << f.rdbuf();
    auto* d = new ss_desc{ss::desc_from_json(ss::json_parse(ss_.str()))};
    *out = d;
  });
}

ss_status ss_desc_from_model(const char* manifest_path, ss_desc** out) { return ss_desc_from_file(manifest_path, out); }

void ss_desc_free(ss_desc* d) { delete d; }

int ss_desc_n_layers(const ss_desc* d) { return d ? int(d->d.layers.size()) : 0; }

int ss_desc_layer_units(const ss_desc* d, int layer) {
  if (!d || layer < 0 || size_t(layer) >= d->d.layers.size()) return -1;
  return d->d.layers[size_t(layer)].n_units;
}

uint64_t ss_desc_unit_bytes(const ss_desc* d, int layer) {
  if (!d || layer < 0 || size_t(layer) >= d->d.layers.size()) return 0;
  return d->d.layers[size_t(layer)].unit_bytes;
}

int64_t ss_desc_unit_index(const ss_desc* d, int layer, int unit) {
  if (!d || layer < 0 || size_t(layer) >= d->d.layers.size()) return -1;
  if (unit < 0 || unit >= d->d.layers[size_t(layer)].n_units) return -1;
  return d->d.unit_base[size_t(layer)] + unit;
}

int64_t ss_desc_total_units(const ss_desc* d) { return d ? d->d.total_units : 0; }

ss_status ss_desc_to_json(const ss_desc* d, char** out_json, size_t* out_len) {
  std::string s;
  ss_status st = guard([&] {
    need(d != nullptr, "ss_desc_to_json: null descriptor");
    s = ss::json_write(ss::desc_to_json(d->d));
  });
  return st != SS_OK ? st : give_text(s, out_json, out_len);
}

void ss_budget_defaults_sized(ss_budget* b, uint32_t size) { sized_defaults(b, size, budget_defaults); }

ss_status ss_plan_make(const ss_desc* d, const ss_budget* b, const double* benefit, size_t n_benefit, ss_plan** out) {
  return guard([&] {
    need(d && b && out, "ss_plan_make: null argument");
    ss_budget bb = versioned(b, budget_defaults, "ss_plan_make");
    std::vector<double> ben;
    if (benefit) ben.assign(benefit, benefit + n_benefit);
    *out = new ss_plan{ss::plan_make(d->d, bb, ben), d};
  });
}

ss_status ss_plan_from_tiers(const ss_desc* d, const ss_budget* b, const uint8_t* tiers, size_t n, ss_plan** out) {
  return guard([&] {
    need(d && b && tiers && out, "ss_plan_from_tiers: null argument");
    ss_budget bb = versioned(b, budget_defaults, "ss_plan_from_tiers");
    *out = new ss_plan{ss::plan_from_tiers(d->d, bb, std::vector<uint8_t>(tiers, tiers + n)), d};
  });
}

ss_status ss_plan_from_json(const ss_desc* d, const ss_budget* b, const char* plan_json, size_t len, ss_plan** out) {
  return guard([&] {
    need(d && b && plan_json && out, "ss_plan_from_json: null argument");
    auto t = ss::tiers_from_json(d->d, ss::json_parse(std::string(plan_json, len)));
    ss_budget bb = versioned(b, budget_defaults, "ss_plan_from_json");
    *out = new ss_plan{ss::plan_from_tiers(d->d, bb, t), d};
  });
}

void ss_plan_free(ss_plan* p) { delete p; }

ss_tier ss_plan_unit_tier(const ss_plan* p, int layer, int unit) {
  int64_t g = p ? ss_desc_unit_index(p->desc, layer, unit) : -1;
  return g < 0 ? SS_TIER_RESIDENT : ss_tier(p->p.tier[size_t(g)]);
}

void ss_plan_get_totals(const ss_plan* p, ss_plan_totals* out) {
  if (!p || !out) return;
  out->fixed_bytes = p->p.fixed;
  out->margin_bytes = p->p.margin;
  out->resident_bytes = p->p.resident;
  out->dynamic_bytes = p->p.dynamic;
  out->ring_bytes = p->p.ring;
  out->peak_device_bytes = p->p.peak;
  out->pinned_bytes = p->p.pinned;
  out->disk_bytes = p->p.disk;
  out->n_resident = p->p.n_resident;
  out->n_pinned = p->p.n_pinned;
  out->n_disk = p->p.n_disk;
}

ss_status ss_plan_to_json(const ss_plan* p, char** out_json, size_t* out_len) {
  std::string s;
  ss_status st = guard([&] {
    need(p != nullptr, "ss_plan_to_json: null plan");
    s = ss::json_write(ss::plan_to_json(p->desc->d, p->p));
  });
  return st != SS_OK ? st : give_text(s, out_json, out_len);
}

void ss_options_defaults_sized(ss_options* o, uint32_t size) { sized_defaults(o, size, options_defaults); }

ss_status ss_open(const ss_desc* d, const ss_budget* budget, const ss_plan* plan, const ss_options* opt, ss_ctx** out) {
  return guard([&] {
    need(d && budget && out, "ss_open: null argument");
    need(!plan || plan->desc == d, "ss_open: plan was made from another descriptor");
    ss_options o = versioned(opt, options_defaults, "ss_open options");
    ss_budget bb = versioned(budget, budget_defaults, "ss_open budget");
    need(!plan || ss::same_budget(plan->p.budget, bb), "ss_open: the plan was made with another budget");
    ss::Plan p = plan ? plan->p : ss::plan_make(d->d, bb, {});
    *out = new ss_ctx{std::make_unique<ss::Ctx>(d->d, bb, p, o)};
  });
}

void ss_close(ss_ctx* c) { delete c; }

int ss_source_direct(const ss_ctx* c, int source) {
  if (!c || source < 0 || source >= c->c->n_sources()) return -1;
  return c->c->source_direct(source) ? 1 : 0;
}

// Context calls: a poisoned context refuses everything; an allocation failure or a non-library exception poisons it
// (library errors - bad arguments, IO, hash, capacity - leave the context consistent and usable).
extern "C++" {
template <class F>
ss_status ctx_guard(ss_ctx* c, F&& f) {
  if (!c) {
    g_err = "null context";
    return SS_E_INVALID;
  }
  if (c->c->poisoned()) {
    g_err = "context poisoned by an earlier out-of-memory or internal failure: close it";
    return SS_E_BACKEND;
  }
  try {
    f();
    return SS_OK;
  } catch (const ss::Error& e) {
    // A device or allocation error mid-call may leave half-updated bookkeeping (and CUDA errors are sticky).
    if (e.status() == SS_E_BACKEND || e.status() == SS_E_NOMEM) c->c->poison();
    g_err = e.what();
    return e.status();
  } catch (const std::bad_alloc&) {
    c->c->poison();
    g_err = "out of host memory (context poisoned)";
    return SS_E_NOMEM;
  } catch (const std::exception& e) {
    c->c->poison();
    g_err = e.what();
    return SS_E_BACKEND;
  }
}

}  // extern "C++"

#define CTX_CALL(expr) return ctx_guard(c, [&] { c->c->expr; })

ss_status ss_seed(ss_ctx* c, int layer, const int32_t* units, int n) { CTX_CALL(seed(layer, units, n)); }
ss_status ss_begin(ss_ctx* c, const uint32_t* rows, int n_layers) { CTX_CALL(begin(rows, n_layers)); }
ss_status ss_layer_routes(ss_ctx* c, int layer, const int32_t* ids, int n) { CTX_CALL(routes(layer, ids, n)); }
ss_status ss_layer_routes_dev(ss_ctx* c, int layer, const int32_t* ids_dev, int n, void* s) {
  CTX_CALL(routes_dev(layer, ids_dev, n, s));
}
ss_status ss_layer_all(ss_ctx* c, int layer) { CTX_CALL(routes_all(layer)); }
ss_status ss_prefetch(ss_ctx* c, int layer, const int32_t* ids, int n, ss_prefetch_kind kind) {
  CTX_CALL(prefetch(layer, ids, n, kind == SS_PREFETCH_SPECULATIVE));
}
ss_status ss_prefetch_layer(ss_ctx* c, int layer) { CTX_CALL(prefetch_layer(layer)); }
ss_status ss_acquire(ss_ctx* c, int layer, void* s, ss_layer_view* out) {
  return ctx_guard(c, [&] {
    need(out != nullptr, "ss_acquire: null view");
    c->c->acquire(layer, s, true, out);
  });
}
ss_status ss_acquire_groups(ss_ctx* c, int layer, void* s, ss_layer_view* out) {
  return ctx_guard(c, [&] {
    need(out != nullptr, "ss_acquire_groups: null view");
    c->c->acquire(layer, s, false, out);
  });
}
ss_status ss_group_wait(ss_ctx* c, int layer, int group, void* s) { CTX_CALL(group_wait(layer, group, s)); }
ss_status ss_release(ss_ctx* c, int layer, void* s) { CTX_CALL(release(layer, s)); }
ss_status ss_end(ss_ctx* c) { CTX_CALL(end()); }
ss_status ss_check_unit(ss_ctx* c, int layer, int unit) { CTX_CALL(check_unit(layer, unit)); }

ss_status ss_lookahead_select(const float* scores, int rows, int n_experts, int top_k, int32_t* out, int max_out,
                              int* n_out) {
  return guard([&] {
    need(scores && out && n_out && rows >= 0 && n_experts > 0 && top_k > 0 && top_k <= n_experts && max_out >= 0,
         "ss_lookahead_select: bad argument");
    // Per row: top_k by (score desc, id asc). Union ordered by (best rank, best score, id).
    std::vector<int> best_rank(size_t(n_experts), top_k);
    std::vector<float> best_score(size_t(n_experts), 0.0f);
    std::vector<int> idx(static_cast<size_t>(n_experts));
    for (int r = 0; r < rows; ++r) {
      const float* s = scores + size_t(r) * size_t(n_experts);
      for (int e = 0; e < n_experts; ++e) idx[size_t(e)] = e;
      std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b) {
        if (s[a] != s[b]) return s[a] > s[b];
        return a < b;
      });
      for (int k = 0; k < top_k; ++k) {
        int e = idx[size_t(k)];
        if (k < best_rank[size_t(e)] || (k == best_rank[size_t(e)] && s[e] > best_score[size_t(e)])) {
          best_rank[size_t(e)] = k;
          best_score[size_t(e)] = s[e];
        }
      }
    }
    std::vector<int> u;
    for (int e = 0; e < n_experts; ++e)
      if (best_rank[size_t(e)] < top_k) u.push_back(e);
    std::sort(u.begin(), u.end(), [&](int a, int b) {
      if (best_rank[size_t(a)] != best_rank[size_t(b)]) return best_rank[size_t(a)] < best_rank[size_t(b)];
      if (best_score[size_t(a)] != best_score[size_t(b)]) return best_score[size_t(a)] > best_score[size_t(b)];
      return a < b;
    });
    int n = std::min(int(u.size()), max_out);
    for (int i = 0; i < n; ++i) out[i] = u[size_t(i)];
    *n_out = n;
  });
}

ss_status ss_stats_layer(const ss_ctx* c, int layer, ss_layer_stats* out) {
  return guard([&] {
    need(c && out && layer >= 0 && layer < c->c->n_layers(), "ss_stats_layer: bad argument");
    *out = c->c->stats(layer);
  });
}

ss_status ss_stats_total(const ss_ctx* c, ss_layer_stats* out) {
  return guard([&] {
    need(c && out, "ss_stats_total: null argument");
    *out = c->c->total();
  });
}

void ss_stats_reset(ss_ctx* c) {
  if (c) c->c->stats_reset();
}

void ss_set_stats_hook(ss_ctx* c, ss_stats_hook hook, void* user) {
  if (c) c->c->set_hook(hook, user);
}

ss_status ss_stats_to_json(const ss_ctx* c, char** out_json, size_t* out_len) {
  std::string s;
  ss_status st = guard([&] {
    need(c != nullptr, "ss_stats_to_json: null context");
    s = ss::json_write(c->c->stats_json());
  });
  return st != SS_OK ? st : give_text(s, out_json, out_len);
}

// Test hook (tests/test_exclusive.cpp, CUDA test): not in ss.h.
ss_status ss_test_check_exclusive(ss_ctx* c) {
  return guard([&] {
    need(c != nullptr, "ss_test_check_exclusive: null context");
    c->c->check_exclusive();
  });
}

ss_status ss_stats_exclusive_sized(const ss_ctx* c, ss_exclusive_stats* out, uint32_t size) {
  return guard([&] {
    need(c && out && size >= sizeof(uint32_t), "ss_stats_exclusive: bad argument");
    ss_exclusive_stats x = c->c->exclusive_stats();
    x.struct_size = std::min<uint32_t>(size, uint32_t(sizeof x));
    std::memcpy(static_cast<void*>(out), &x, x.struct_size);
  });
}

ss_status ss_recipe_from_text(const char* text, size_t len, const char* origin, ss_recipe** out) {
  return guard([&] {
    need(text && out, "ss_recipe_from_text: null argument");
    *out = new ss_recipe{ss::RecipeStack(std::string(text, len), origin ? origin : "<text>")};
  });
}

ss_status ss_recipe_from_file(const char* path, ss_recipe** out) {
  return guard([&] {
    need(path && out, "ss_recipe_from_file: null argument");
    *out = new ss_recipe{ss::RecipeStack(read_text(path, "recipe"), path)};
  });
}

void ss_recipe_free(ss_recipe* r) { delete r; }

ss_status ss_recipe_add_profile(ss_recipe* r, const char* text, size_t len, const char* origin,
                                const char* identity_json, size_t identity_len) {
  return guard([&] {
    need(r && text && identity_json, "ss_recipe_add_profile: null argument");
    ss::Json id = parse_input(identity_json, identity_len, "identity");
    r->s.add_profile(std::string(text, len), origin ? origin : "<profile>", id);
  });
}

ss_status ss_recipe_add_overrides(ss_recipe* r, const char* text, size_t len, const char* origin) {
  return guard([&] {
    need(r && text, "ss_recipe_add_overrides: null argument");
    r->s.add_overrides(std::string(text, len), origin ? origin : "<overrides>");
  });
}

ss_status ss_recipe_set(ss_recipe* r, const char* path, const char* value, const char* origin) {
  return guard([&] {
    need(r && path && value, "ss_recipe_set: null argument");
    r->s.set(path, value, origin ? origin : "--set");
  });
}

ss_status ss_recipe_to_json(const ss_recipe* r, char** out, size_t* len) {
  std::string s;
  ss_status st = guard([&] {
    need(r != nullptr, "ss_recipe_to_json: null recipe");
    s = ss::json_write(r->s.effective_json());
  });
  return st != SS_OK ? st : give_text(s, out, len);
}

ss_status ss_recipe_explain(const ss_recipe* r, char** out, size_t* len) {
  std::string s;
  ss_status st = guard([&] {
    need(r != nullptr, "ss_recipe_explain: null recipe");
    s = ss::json_write(r->s.explain());
  });
  return st != SS_OK ? st : give_text(s, out, len);
}

ss_status ss_recipe_match(const ss_recipe* r, const char* probe_json, size_t len, char** out, size_t* olen) {
  std::string s;
  ss_status st = guard([&] {
    need(r && probe_json, "ss_recipe_match: null argument");
    ss::Binding b = ss::recipe_bind(r->s.effective(), parse_input(probe_json, len, "probe"));
    ss::Json j = ss::Json::object();
    j["schema"] = ss::Json::string("strix-stream/recipe-match-v1");
    j["recipe"] = ss::Json::string(r->s.effective().name);
    j["effective_sha256"] = ss::Json::string(r->s.effective_sha256());
    j["binding"] = ss::binding_json(b);
    s = ss::json_write(j);
  });
  return st != SS_OK ? st : give_text(s, out, olen);
}

ss_status ss_recipe_select(const ss_recipe* r, const char* state_json, size_t len, char** out, size_t* olen) {
  std::string s;
  ss_status st = guard([&] {
    need(r && state_json, "ss_recipe_select: null argument");
    s = ss::json_write(
        ss::recipe_select(r->s.effective(), r->s.effective_sha256(), parse_input(state_json, len, "state")));
  });
  return st != SS_OK ? st : give_text(s, out, olen);
}

ss_status ss_recipe_plan(const ss_recipe* r, const char* route, const char* set, int member, const char* device_id,
                         const ss_desc* desc, const double* benefit, size_t n_benefit, const ss_desc* const* ref_descs,
                         const double* const* ref_benefits, const size_t* ref_n_benefits, size_t n_refs,
                         ss_budget* budget, ss_options* options, ss_plan** plan, char** out, size_t* olen) {
  std::string s;
  ss_plan* made = nullptr;
  ss_budget bout{};
  ss_options oout{};
  ss_status st = guard([&] {
    need(r && route && set && device_id && desc && budget && options, "ss_recipe_plan: null argument");
    std::vector<double> ben;
    if (benefit) ben.assign(benefit, benefit + n_benefit);
    std::vector<ss::RefInput> refs;
    need(n_refs == 0 || ref_descs, "ss_recipe_plan: n_refs > 0 needs ref_descs");
    for (size_t k = 0; k < n_refs; ++k) {
      ss::RefInput ref;
      ref.desc = ref_descs[k] ? &ref_descs[k]->d : nullptr;
      if (ref_benefits && ref_benefits[k]) {
        need(ref_n_benefits != nullptr, "ss_recipe_plan: ref_benefits needs ref_n_benefits");
        ref.benefit.assign(ref_benefits[k], ref_benefits[k] + ref_n_benefits[k]);
      }
      refs.push_back(std::move(ref));
    }
    ss::RecipePlan rp = ss::recipe_plan(r->s.effective(), r->s.effective_sha256(), route, set, member, device_id,
                                        desc->d, ben, refs);
    // Versioned in/out structs: start from the caller's (device ordinal kept), recipe fields from the plan.
    bout = rp.budget;
    oout = versioned(options, options_defaults, "ss_recipe_plan");
    const int device = oout.device;
    oout = rp.options;
    oout.device = device;
    bout.struct_size = uint32_t(sizeof bout);
    oout.struct_size = uint32_t(sizeof oout);
    need(budget->struct_size >= 2 * sizeof(uint32_t), "ss_recipe_plan: budget struct_size not set (ss_budget_defaults)");
    // A truncated copy of an exclusive budget would read back as inclusive: refuse rather than hand out a budget
    // that no longer matches the plan.
    need(bout.tier_mode == SS_TIER_MODE_INCLUSIVE ||
             budget->struct_size >= offsetof(ss_budget, host_spare_units) + sizeof(bout.host_spare_units),
         "ss_recipe_plan: the route's placement is tier_mode = exclusive, which this caller's ss_budget (an older "
         "header) cannot hold");
    if (plan) made = new ss_plan{std::move(rp.plan), desc};
    s = ss::json_write(rp.doc);
  });
  if (st != SS_OK) return st;
  if (out) {
    ss_status t = give_text(s, out, olen);
    if (t != SS_OK) {
      delete made;
      return t;
    }
  }
  const uint32_t bsz = budget->struct_size;
  std::memcpy(static_cast<void*>(budget), &bout, std::min<size_t>(bsz, sizeof bout));
  budget->struct_size = std::min<uint32_t>(bsz, uint32_t(sizeof bout));
  const uint32_t osz = options->struct_size;
  std::memcpy(static_cast<void*>(options), &oout, std::min<size_t>(osz, sizeof oout));
  options->struct_size = std::min<uint32_t>(osz, uint32_t(sizeof oout));
  if (plan) *plan = made;
  return SS_OK;
}

}  // extern "C"
