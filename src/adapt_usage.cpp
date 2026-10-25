#include "adapt_usage.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "errors.hpp"
#include "sha256.hpp"
#include "strix_stream/ss.h"

namespace ts {
namespace adapt {

namespace {

// Full 64 x 64 -> 128 product as (hi, lo).
void mul128(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
  const uint64_t a0 = a & 0xffffffffu, a1 = a >> 32, b0 = b & 0xffffffffu, b1 = b >> 32;
  const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const uint64_t mid = (p00 >> 32) + (p01 & 0xffffffffu) + (p10 & 0xffffffffu);
  lo = (mid << 32) | (p00 & 0xffffffffu);
  hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
}

const char* kSchema = "tierstream/usage-v1";

}  // namespace

uint64_t mul_q32(uint64_t a, uint64_t q32) {
  uint64_t hi, lo;
  mul128(a, q32, hi, lo);
  // + 2^31 then >> 32 (round half up)
  const uint64_t add = uint64_t(1) << 31;
  const uint64_t lo2 = lo + add;
  if (lo2 < lo) ++hi;
  require(hi < (uint64_t(1) << 32), SS_E_INVALID, "usage: fixed-point overflow");
  return (hi << 32) | (lo2 >> 32);
}

uint64_t pow_q32(uint64_t q32, uint64_t e) {
  uint64_t r = uint64_t(1) << 32, b = q32;
  while (e) {
    if (e & 1) r = mul_q32(r, b);
    e >>= 1;
    if (e) b = mul_q32(b, b);
    if (r == 0) break;
  }
  return r;
}

uint64_t decay_q32(uint32_t half_life_tokens) {
  if (half_life_tokens == 0) return uint64_t(1) << 32;
  // Computed once per half life; the integer is stored in the table so every reader uses the same factor.
  const double f = std::exp2(-1.0 / double(half_life_tokens));
  return uint64_t(std::llround(f * 4294967296.0));
}

uint64_t rdiv(uint64_t n, uint64_t d) {
  require(d > 0, SS_E_INVALID, "rdiv: zero divisor");
  // (2n + d) / (2d) without overflow for n, d < 2^62
  require(n < (uint64_t(1) << 62) && d < (uint64_t(1) << 62), SS_E_INVALID, "rdiv: operand too large");
  return (2 * n + d) / (2 * d);
}

Usage::Usage(std::string model, std::vector<uint32_t> units_per_layer, uint32_t half_life_tokens)
    : model_(std::move(model)), units_(std::move(units_per_layer)), half_life_(half_life_tokens) {
  require(!units_.empty(), SS_E_INVALID, "usage: no layers");
  require(half_life_tokens < (1u << 30), SS_E_INVALID, "usage: half life out of range");
  q32_ = decay_q32(half_life_);
  int64_t b = 0;
  for (uint32_t n : units_) {
    base_.push_back(b);
    b += n;
  }
  count_.assign(size_t(b), 0);
  rec_.assign(size_t(b), 0);
  last_.assign(size_t(b), 0);
}

void Usage::touch(int64_t g) {
  const size_t i = size_t(g);
  if (last_[i] != tick_) {
    if (rec_[i]) rec_[i] = mul_q32(rec_[i], pow_q32(q32_, tick_ - last_[i]));
    last_[i] = tick_;
  }
}

void Usage::observe(int layer, const int32_t* ids, int n) {
  require(layer >= 0 && layer < n_layers(), SS_E_INVALID, "usage: layer out of range");
  require(n >= 0 && (ids || n == 0), SS_E_INVALID, "usage: bad ids");
  const uint32_t nu = units_[size_t(layer)];
  for (int i = 0; i < n; ++i) {
    const int32_t u = ids[i];
    if (u < 0) continue;
    require(uint32_t(u) < nu, SS_E_INVALID, "usage: unit id " + std::to_string(u) + " out of range");
    const int64_t g = base_[size_t(layer)] + u;
    ++count_[size_t(g)];
    touch(g);
    rec_[size_t(g)] += kMicro;
  }
}

void Usage::tick(uint32_t rows) {
  tokens_ += rows;
  tick_ += rows;
}

uint64_t Usage::use(int64_t g) const {
  if (!tokens_) return 0;
  const uint64_t c = count_[size_t(g)];
  // c * 10^6 fits: c < 2^43 for any realistic trace (checked by rdiv's bound).
  return rdiv(c * kMicro, tokens_);
}

uint64_t Usage::recency(int64_t g) const {
  const size_t i = size_t(g);
  if (!rec_[i] || last_[i] == tick_) return rec_[i];
  return mul_q32(rec_[i], pow_q32(q32_, tick_ - last_[i]));
}

std::vector<int64_t> Usage::rank() const {
  const int64_t n = total_units();
  std::vector<uint64_t> u(static_cast<size_t>(n)), r(static_cast<size_t>(n));
  for (int64_t g = 0; g < n; ++g) {
    u[size_t(g)] = use(g);
    r[size_t(g)] = recency(g);
  }
  std::vector<int64_t> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), int64_t(0));
  // Global index is layer-major, ascending unit: comparing it is comparing (layer, unit).
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
    if (u[size_t(a)] != u[size_t(b)]) return u[size_t(a)] > u[size_t(b)];
    if (r[size_t(a)] != r[size_t(b)]) return r[size_t(a)] > r[size_t(b)];
    return a < b;
  });
  return order;
}

void Usage::merge(const Usage& o) {
  require(units_ == o.units_, SS_E_INVALID, "usage merge: different shapes");
  require(half_life_ == o.half_life_, SS_E_INVALID, "usage merge: different half lives");
  const uint64_t t = std::max(tick_, o.tick_);
  for (int64_t g = 0; g < total_units(); ++g) {
    const size_t i = size_t(g);
    const uint64_t a = rec_[i] ? mul_q32(rec_[i], pow_q32(q32_, t - last_[i])) : 0;
    const uint64_t b = o.rec_[i] ? mul_q32(o.rec_[i], pow_q32(q32_, t - o.last_[i])) : 0;
    rec_[i] = a + b;
    last_[i] = t;
    count_[i] += o.count_[i];
  }
  tick_ = t;
  tokens_ += o.tokens_;
  for (const Source& s : o.sources_) sources_.push_back(s);
}

Json Usage::to_json() const {
  Json j = Json::object();
  j["schema"] = Json::string(kSchema);
  j["model"] = Json::string(model_);
  j["tokens"] = Json::number(tokens_);
  j["tick"] = Json::number(tick_);
  Json dec = Json::object();
  dec["half_life_tokens"] = Json::number(uint64_t(half_life_));
  dec["factor_q32"] = Json::number(q32_);
  j["decay"] = dec;
  j["order"] = Json::string("use desc, recency desc, layer asc, unit asc");
  Json src = Json::array();
  for (const Source& s : sources_) {
    Json o = Json::object();
    o["kind"] = Json::string(s.kind);
    o["path"] = Json::string(s.path);
    o["sha256"] = Json::string(s.sha256);
    o["filter"] = Json::string(s.filter);
    o["tokens"] = Json::number(s.tokens);
    src.push(o);
  }
  j["sources"] = src;
  Json layers = Json::array();
  for (int L = 0; L < n_layers(); ++L) {
    Json l = Json::object();
    l["index"] = Json::number(int64_t(L));
    l["n_units"] = Json::number(uint64_t(units_[size_t(L)]));
    Json c = Json::array(), u = Json::array(), r = Json::array();
    for (uint32_t k = 0; k < units_[size_t(L)]; ++k) {
      const int64_t g = base_[size_t(L)] + k;
      c.push(Json::number(count_[size_t(g)]));
      u.push(Json::number(use(g)));
      r.push(Json::number(recency(g)));
    }
    l["count"] = c;
    l["use"] = u;
    l["recency"] = r;
    layers.push(l);
  }
  j["layers"] = layers;
  return j;
}

std::string Usage::to_text() const { return json_write(to_json()); }

std::string Usage::sha256() const {
  const std::string t = to_text();
  return sha256_hex(t.data(), t.size());
}

Usage Usage::from_json(const Json& j) {
  require(j.is_object(), SS_E_INVALID, "usage: not an object");
  const Json* sch = j.find("schema");
  require(sch && sch->is_string() && sch->text == kSchema, SS_E_INVALID,
          std::string("usage: schema must be ") + kSchema);
  const Json* layers = j.find("layers");
  require(layers && layers->is_array() && !layers->arr.empty(), SS_E_INVALID, "usage: layers missing");
  const Json* dec = j.find("decay");
  require(dec && dec->is_object(), SS_E_INVALID, "usage: decay missing");
  std::vector<uint32_t> units;
  for (size_t L = 0; L < layers->arr.size(); ++L) {
    const Json& l = layers->arr[L];
    const std::string w = "usage.layers[" + std::to_string(L) + "]";
    require(l.is_object() && l.find("index") && json_u64(*l.find("index"), w + ".index") == L, SS_E_INVALID,
            w + ": index must equal its position");
    const Json* nu = l.find("n_units");
    require(nu != nullptr, SS_E_INVALID, w + ".n_units missing");
    const uint64_t n = json_u64(*nu, w + ".n_units");
    require(n < (uint64_t(1) << 20), SS_E_INVALID, w + ".n_units out of range");  // 0 = a dense layer
    units.push_back(uint32_t(n));
  }
  const Json* hl = dec->find("half_life_tokens");
  require(hl != nullptr, SS_E_INVALID, "usage.decay.half_life_tokens missing");
  const Json* mdl = j.find("model");
  Usage u(mdl && mdl->is_string() ? mdl->text : "", units, uint32_t(json_u64(*hl, "usage.decay.half_life_tokens")));
  const Json* fq = dec->find("factor_q32");
  // The stored factor is authoritative (a reader never recomputes it with its own libm).
  require(fq != nullptr, SS_E_INVALID, "usage.decay.factor_q32 missing");
  u.q32_ = json_u64(*fq, "usage.decay.factor_q32");
  require(u.q32_ > 0 && u.q32_ <= (uint64_t(1) << 32), SS_E_INVALID, "usage.decay.factor_q32 out of range");
  const Json* tok = j.find("tokens");
  const Json* tk = j.find("tick");
  require(tok && tk, SS_E_INVALID, "usage: tokens / tick missing");
  u.tokens_ = json_u64(*tok, "usage.tokens");
  u.tick_ = json_u64(*tk, "usage.tick");
  for (size_t L = 0; L < layers->arr.size(); ++L) {
    const Json& l = layers->arr[L];
    const std::string w = "usage.layers[" + std::to_string(L) + "]";
    const Json* c = l.find("count");
    const Json* r = l.find("recency");
    require(c && c->is_array() && c->arr.size() == units[L], SS_E_INVALID, w + ".count: one per unit");
    require(r && r->is_array() && r->arr.size() == units[L], SS_E_INVALID, w + ".recency: one per unit");
    for (uint32_t k = 0; k < units[L]; ++k) {
      const size_t g = size_t(u.base_[L] + k);
      u.count_[g] = json_u64(c->arr[k], w + ".count");
      u.rec_[g] = json_u64(r->arr[k], w + ".recency");
      u.last_[g] = u.tick_;
    }
    // "use" is derived; a mismatch means the file was edited by hand or written by another rule.
    const Json* us = l.find("use");
    if (us) {
      require(us->is_array() && us->arr.size() == units[L], SS_E_INVALID, w + ".use: one per unit");
      for (uint32_t k = 0; k < units[L]; ++k)
        require(json_u64(us->arr[k], w + ".use") == u.use(u.base_[L] + k), SS_E_INVALID,
                w + ".use[" + std::to_string(k) + "] is not rdiv(count * 10^6, tokens)");
    }
  }
  if (const Json* src = j.find("sources")) {
    require(src->is_array(), SS_E_INVALID, "usage.sources: expected an array");
    for (const Json& o : src->arr) {
      Source s;
      auto str = [&](const char* k) {
        const Json* v = o.find(k);
        return v && v->is_string() ? v->text : std::string();
      };
      s.kind = str("kind");
      s.path = str("path");
      s.sha256 = str("sha256");
      s.filter = str("filter");
      if (const Json* t = o.find("tokens")) s.tokens = json_u64(*t, "usage.sources.tokens");
      u.sources_.push_back(s);
    }
  }
  return u;
}

Usage Usage::from_text(const std::string& text) { return from_json(json_parse(text)); }

std::vector<uint8_t> tiers_from_rank(const std::vector<int64_t>& order, int64_t n_total, int64_t active, int64_t host) {
  require(int64_t(order.size()) == n_total, SS_E_INVALID, "tiers: the ranking must list every unit");
  require(active >= 0 && host >= 0 && active + host <= n_total, SS_E_INVALID, "tiers: active + host > units");
  std::vector<uint8_t> t(static_cast<size_t>(n_total), uint8_t(SS_TIER_DISK));
  std::vector<char> seen(static_cast<size_t>(n_total), 0);
  for (int64_t i = 0; i < n_total; ++i) {
    const int64_t g = order[size_t(i)];
    require(g >= 0 && g < n_total && !seen[size_t(g)], SS_E_INVALID, "tiers: the ranking is not a permutation");
    seen[size_t(g)] = 1;
    t[size_t(g)] = uint8_t(i < active ? SS_TIER_RESIDENT : i < active + host ? SS_TIER_PINNED : SS_TIER_DISK);
  }
  return t;
}

}  // namespace adapt
}  // namespace ts
