// C ABI of the adaptive memory plan (include/tierstream/ts_adapt.h).
#include <cstring>
#include <fstream>
#include <sstream>

#include "adapt_solve.hpp"
#include "adapt_usage.hpp"
#include "capi_internal.hpp"
#include "desc.hpp"
#include "errors.hpp"
#include "stream_ctx.hpp"
#include "tierstream/ts_adapt.h"

struct ts_usage {
  ts::adapt::Usage u;
};

namespace {

void need(bool ok, const char* what) { ts::require(ok, SS_E_INVALID, what); }

}  // namespace

extern "C" {

ts_status ts_usage_new(const char* model, const uint32_t* units_per_layer, int n_layers, uint32_t half_life_tokens,
                       ts_usage** out) {
  return ts_capi_guard([&] {
    need(out && units_per_layer && n_layers > 0, "ts_usage_new: bad arguments");
    std::vector<uint32_t> u(units_per_layer, units_per_layer + n_layers);
    *out = new ts_usage{ts::adapt::Usage(model ? model : "", u, half_life_tokens)};
  });
}

ts_status ts_usage_new_for(const ts_desc* d, uint32_t half_life_tokens, ts_usage** out) {
  return ts_capi_guard([&] {
    const ts::Desc* dd = ts_capi_desc(d);
    need(out && dd, "ts_usage_new_for: bad arguments");
    std::vector<uint32_t> u;
    for (const ts::Layer& l : dd->layers) u.push_back(uint32_t(l.n_units));
    *out = new ts_usage{ts::adapt::Usage(dd->model, u, half_life_tokens)};
  });
}

ts_status ts_usage_from_json(const char* text, size_t len, ts_usage** out) {
  return ts_capi_guard([&] {
    need(out && (text || len == 0), "ts_usage_from_json: bad arguments");
    *out = new ts_usage{ts::adapt::Usage::from_text(std::string(text, len))};
  });
}

ts_status ts_usage_from_file(const char* path, ts_usage** out) {
  return ts_capi_guard([&] {
    need(out && path, "ts_usage_from_file: bad arguments");
    std::ifstream f(path, std::ios::binary);
    ts::require(bool(f), SS_E_IO, std::string("cannot open usage table ") + path);
    std::stringstream ss;
    ss << f.rdbuf();
    *out = new ts_usage{ts::adapt::Usage::from_text(ss.str())};
  });
}

void ts_usage_free(ts_usage* u) { delete u; }

ts_status ts_usage_to_json(const ts_usage* u, char** out_json, size_t* out_len) {
  std::string s;
  ts_status st = ts_capi_guard([&] {
    need(u != nullptr, "ts_usage_to_json: null table");
    s = u->u.to_text();
  });
  return st != SS_OK ? st : ts_capi_text(s, out_json, out_len);
}

ts_status ts_usage_sha256(const ts_usage* u, char out_hex[65]) {
  return ts_capi_guard([&] {
    need(u && out_hex, "ts_usage_sha256: bad arguments");
    const std::string h = u->u.sha256();
    std::memcpy(out_hex, h.c_str(), 65);
  });
}

ts_status ts_usage_observe(ts_usage* u, int layer, const int32_t* ids, int n) {
  return ts_capi_guard([&] {
    need(u != nullptr, "ts_usage_observe: null table");
    u->u.observe(layer, ids, n);
  });
}

ts_status ts_usage_tick(ts_usage* u, uint32_t rows) {
  return ts_capi_guard([&] {
    need(u != nullptr, "ts_usage_tick: null table");
    u->u.tick(rows);
  });
}

ts_status ts_usage_attach(ts_ctx* c, ts_usage* u) {
  return ts_capi_guard([&] {
    ts::Ctx* cc = ts_capi_ctx(c);
    need(cc != nullptr, "ts_usage_attach: needs a host-mode context (device-gated mode: the engine feeds ts_usage_observe / ts_usage_tick)");
    ts::require(!cc->poisoned(), SS_E_INVALID, "ts_usage_attach: the context is poisoned");
    cc->set_usage(u ? &u->u : nullptr);
  });
}

int64_t ts_usage_total_units(const ts_usage* u) { return u ? u->u.total_units() : -1; }
uint64_t ts_usage_tokens(const ts_usage* u) { return u ? u->u.tokens() : 0; }

ts_status ts_usage_unit(const ts_usage* u, int64_t g, uint64_t* use_micro, uint64_t* recency_micro, uint64_t* count) {
  return ts_capi_guard([&] {
    need(u && g >= 0 && g < u->u.total_units(), "ts_usage_unit: bad unit");
    if (use_micro) *use_micro = u->u.use(g);
    if (recency_micro) *recency_micro = u->u.recency(g);
    if (count) *count = u->u.count(g);
  });
}

ts_status ts_usage_rank(const ts_usage* u, int64_t* order, size_t n) {
  return ts_capi_guard([&] {
    need(u && order && int64_t(n) == u->u.total_units(), "ts_usage_rank: n must be the table's unit count");
    const std::vector<int64_t> r = u->u.rank();
    std::memcpy(order, r.data(), sizeof(int64_t) * n);
  });
}

ts_status ts_usage_tiers(const ts_usage* u, int64_t active, int64_t host, uint8_t* tiers, size_t n) {
  return ts_capi_guard([&] {
    need(u && tiers && int64_t(n) == u->u.total_units(), "ts_usage_tiers: n must be the table's unit count");
    const std::vector<uint8_t> t = ts::adapt::tiers_from_rank(u->u.rank(), int64_t(n), active, host);
    std::memcpy(tiers, t.data(), n);
  });
}

ts_status ts_usage_merge(ts_usage* a, const ts_usage* b) {
  return ts_capi_guard([&] {
    need(a && b, "ts_usage_merge: null table");
    a->u.merge(b->u);
  });
}

ts_status ts_adapt_solve(const char* input_json, size_t len, const char* base_dir, char** out_json, size_t* out_len) {
  std::string s;
  ts_status st = ts_capi_guard([&] {
    need(input_json != nullptr, "ts_adapt_solve: null input");
    s = ts::json_write(ts::adapt::solve(ts::json_parse(std::string(input_json, len)), base_dir ? base_dir : ""));
  });
  return st != SS_OK ? st : ts_capi_text(s, out_json, out_len);
}

}  // extern "C"
