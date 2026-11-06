#include "adapt_sim.hpp"

#include <algorithm>
#include <deque>
#include <fstream>
#include <sstream>
#include <string_view>

#include "errors.hpp"
#include "sha256.hpp"
#include "strix_stream/ss.h"

namespace ts {
namespace adapt {

namespace {

bool is_digit(char c) { return c >= '0' && c <= '9'; }

}  // namespace

Trace trace_parse(const std::string& text, int n_units_hint) {
  Trace t;
  t.sha256 = sha256_hex(text.data(), text.size());
  size_t pos = 0;
  int max_id = -1;
  std::vector<int32_t> row;
  Trace::Seq* cur = nullptr;
  uint64_t line_no = 0;
  while (pos < text.size()) {
    size_t eol = text.find('\n', pos);
    if (eol == std::string::npos) eol = text.size();
    std::string_view line(text.data() + pos, eol - pos);
    pos = eol + 1;
    ++line_no;
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (line.empty()) continue;
    if (line[0] == '#') {
      // "# seq <name> rows <n>"
      std::istringstream is{std::string(line.substr(1))};
      std::string kw, name;
      is >> kw >> name;
      require(kw == "seq" && !name.empty(), SS_E_INVALID, "trace line " + std::to_string(line_no) + ": bad header");
      Trace::Seq s;
      s.name = name;
      s.first = uint32_t(t.n_layers && t.top_k ? t.n_tokens() : 0);
      t.seqs.push_back(s);
      cur = &t.seqs.back();
      continue;
    }
    require(cur != nullptr, SS_E_INVALID, "trace: token line before any '# seq' header");
    // Parse "l:a,b,c l:a,b,c ..."
    struct Item {
      int layer;
      std::vector<int32_t> ids;
    };
    std::vector<Item> items;
    size_t i = 0;
    while (i < line.size()) {
      while (i < line.size() && line[i] == ' ') ++i;
      if (i >= line.size()) break;
      int l = 0;
      require(is_digit(line[i]), SS_E_INVALID, "trace line " + std::to_string(line_no) + ": expected a layer");
      while (i < line.size() && is_digit(line[i])) l = l * 10 + (line[i++] - '0');
      require(i < line.size() && line[i] == ':', SS_E_INVALID, "trace line " + std::to_string(line_no) + ": expected ':'");
      ++i;
      Item it{l, {}};
      while (i < line.size() && line[i] != ' ') {
        require(is_digit(line[i]), SS_E_INVALID, "trace line " + std::to_string(line_no) + ": expected an id");
        int v = 0;
        while (i < line.size() && is_digit(line[i])) v = v * 10 + (line[i++] - '0');
        it.ids.push_back(v);
        max_id = std::max(max_id, v);
        if (i < line.size() && line[i] == ',') ++i;
      }
      items.push_back(std::move(it));
    }
    if (items.empty()) continue;
    if (t.n_layers == 0) {
      int ml = 0;
      size_t mk = 0;
      for (const Item& it : items) {
        ml = std::max(ml, it.layer + 1);
        mk = std::max(mk, it.ids.size());
      }
      t.n_layers = ml;
      t.top_k = int(mk);
      require(t.top_k > 0, SS_E_INVALID, "trace: no ids on the first token line");
    }
    row.assign(size_t(t.n_layers) * size_t(t.top_k), -1);
    for (const Item& it : items) {
      require(it.layer < t.n_layers, SS_E_INVALID, "trace line " + std::to_string(line_no) + ": layer out of range");
      require(int(it.ids.size()) <= t.top_k, SS_E_INVALID, "trace line " + std::to_string(line_no) + ": too many ids");
      for (size_t k = 0; k < it.ids.size(); ++k) row[size_t(it.layer) * size_t(t.top_k) + k] = it.ids[k];
    }
    t.ids.insert(t.ids.end(), row.begin(), row.end());
    ++cur->tokens;
  }
  require(t.n_layers > 0, SS_E_INVALID, "trace: no tokens");
  t.n_units = n_units_hint > 0 ? n_units_hint : max_id + 1;
  require(max_id < t.n_units, SS_E_INVALID, "trace: an id is >= the unit count");
  return t;
}

Trace trace_load(const std::string& path, int n_units_hint) {
  std::ifstream f(path, std::ios::binary);
  require(bool(f), SS_E_IO, "cannot open trace " + path);
  std::stringstream ss;
  ss << f.rdbuf();
  Trace t = trace_parse(ss.str(), n_units_hint);
  t.path = path;
  return t;
}

namespace {

bool match_or(const std::string& name, const std::string& f) {
  bool neg = !f.empty() && f[0] == '!';
  const std::string g = neg ? f.substr(1) : f;
  bool hit = false;
  size_t p = 0;
  while (p <= g.size()) {
    size_t q = g.find(',', p);
    if (q == std::string::npos) q = g.size();
    const std::string part = g.substr(p, q - p);
    if (!part.empty() && name.find(part) != std::string::npos) hit = true;
    p = q + 1;
  }
  return neg ? !hit : hit;
}

}  // namespace

bool seq_match(const std::string& name, const std::string& filter) {
  if (filter.empty()) return true;
  size_t p = 0;
  while (p <= filter.size()) {
    size_t q = filter.find('&', p);
    if (q == std::string::npos) q = filter.size();
    const std::string part = filter.substr(p, q - p);
    if (!part.empty() && !match_or(name, part)) return false;
    p = q + 1;
  }
  return true;
}

Usage usage_from_trace(const Trace& t, const std::string& filter, const std::string& model, uint32_t half_life) {
  Usage u(model, std::vector<uint32_t>(size_t(t.n_layers), uint32_t(t.n_units)), half_life);
  uint64_t tokens = 0;
  for (const Trace::Seq& s : t.seqs) {
    if (!seq_match(s.name, filter)) continue;
    for (uint32_t k = 0; k < s.tokens; ++k) {
      for (int L = 0; L < t.n_layers; ++L) u.observe(L, t.at(s.first + k, L), t.top_k);
      u.tick(1);
      ++tokens;
    }
  }
  Source src;
  src.kind = "routes-trace";
  src.path = t.path;
  src.sha256 = t.sha256;
  src.filter = filter;
  src.tokens = tokens;
  u.sources().push_back(src);
  return u;
}

namespace {

// Intrusive LRU over global keys: head = least recent.
struct Lru {
  std::vector<int64_t> prev, next;
  std::vector<char> in;
  int64_t head = -1, tail = -1, size = 0;
  explicit Lru(int64_t n) : prev(size_t(n), -1), next(size_t(n), -1), in(size_t(n), 0) {}
  void push_mru(int64_t g) {
    prev[size_t(g)] = tail;
    next[size_t(g)] = -1;
    if (tail >= 0) next[size_t(tail)] = g;
    else head = g;
    tail = g;
    in[size_t(g)] = 1;
    ++size;
  }
  void remove(int64_t g) {
    const int64_t p = prev[size_t(g)], n = next[size_t(g)];
    if (p >= 0) next[size_t(p)] = n;
    else head = n;
    if (n >= 0) prev[size_t(n)] = p;
    else tail = p;
    in[size_t(g)] = 0;
    --size;
  }
  void touch(int64_t g) {
    if (g == tail) return;
    remove(g);
    push_mru(g);
  }
};

}  // namespace

SimResult simulate(const Trace& t, const std::string& filter, const std::vector<int64_t>& order, const SimConfig& c) {
  const int64_t n = int64_t(t.n_layers) * t.n_units;
  require(int64_t(order.size()) == n, SS_E_INVALID, "simulate: the ranking does not match the trace's units");
  require(c.active >= 0 && c.spares >= 0 && c.file_units >= 0 && c.active + c.file_units <= n, SS_E_INVALID,
          "simulate: active + file_units exceeds the units");
  require(c.rows >= 1, SS_E_INVALID, "simulate: rows >= 1");
  std::vector<char> is_file(size_t(n), 0);
  for (int64_t i = n - c.file_units; i < n; ++i) is_file[size_t(order[size_t(i)])] = 1;
  Lru lru(n);
  for (int64_t i = c.active - 1; i >= 0; --i) lru.push_mru(order[size_t(i)]);  // hottest = most recent
  int64_t free = c.spares;
  std::deque<std::pair<uint64_t, int64_t>> pending;
  std::vector<uint64_t> stamp(size_t(n), 0);
  SimResult r;
  uint64_t w = 0;
  std::vector<int64_t> keys;
  std::vector<int64_t> victims;
  for (const Trace::Seq& s : t.seqs) {
    if (!seq_match(s.name, filter)) continue;
    for (uint32_t k0 = 0; k0 < s.tokens; k0 += c.rows) {
      ++w;
      const uint32_t k1 = std::min(s.tokens, k0 + c.rows);
      r.tokens += k1 - k0;
      while (!pending.empty() && pending.front().first <= w) {
        free += pending.front().second;
        pending.pop_front();
      }
      for (int L = 0; L < t.n_layers; ++L) {
        keys.clear();
        for (uint32_t k = k0; k < k1; ++k) {
          const int32_t* ids = t.at(s.first + k, L);
          for (int j = 0; j < t.top_k; ++j) {
            if (ids[j] < 0) continue;
            const int64_t g = int64_t(L) * t.n_units + ids[j];
            if (stamp[size_t(g)] == w) continue;
            stamp[size_t(g)] = w;
            keys.push_back(g);
          }
        }
        for (int64_t g : keys) {
          ++r.lookups;
          if (is_file[size_t(g)]) ++r.file_uses;
          if (lru.in[size_t(g)]) {
            lru.touch(g);
            continue;
          }
          if (is_file[size_t(g)]) ++r.miss_file;
          else ++r.miss_pinned;
          if (free > 0) {
            --free;
            lru.push_mru(g);
          } else {
            ++r.bank;
          }
        }
      }
      // Boundary.
      int64_t inflight = 0;
      for (const auto& p : pending) inflight += p.second;
      int64_t need = c.spares - free - inflight;
      int64_t wb = 0;
      if (need > 0 && c.evict == Evict::FileLast) {
        // Pinned homes among the `window` least recent go first (cheap to re-fetch), then plain LRU.
        victims.clear();
        int64_t g = lru.head;
        for (uint32_t i = 0; i < c.file_last_window && g >= 0 && int64_t(victims.size()) < need; ++i) {
          if (!is_file[size_t(g)]) victims.push_back(g);
          g = lru.next[size_t(g)];
        }
        for (int64_t v : victims) {
          lru.remove(v);
          ++wb;
          ++r.evicted;
        }
        need -= int64_t(victims.size());
      }
      while (need > 0 && lru.head >= 0) {
        const int64_t v = lru.head;
        lru.remove(v);
        if (is_file[size_t(v)]) {
          ++r.discarded;
          ++free;
        } else {
          ++wb;
          ++r.evicted;
        }
        --need;
      }
      if (wb) pending.emplace_back(w + c.wb_windows, wb);
    }
  }
  r.windows = w;
  return r;
}

void SimResult::add(const SimResult& r) {
  windows += r.windows;
  tokens += r.tokens;
  lookups += r.lookups;
  miss_pinned += r.miss_pinned;
  miss_file += r.miss_file;
  bank += r.bank;
  evicted += r.evicted;
  discarded += r.discarded;
  file_uses += r.file_uses;
}

SimResult simulate_xval(const Trace& t, const std::string& filter, const SimConfig& c, int folds, uint32_t half_life) {
  require(folds >= 2, SS_E_INVALID, "xval: folds >= 2");
  Trace tf = t;
  for (size_t i = 0; i < tf.seqs.size(); ++i)
    tf.seqs[i].name = "#f" + std::to_string(int(i) % folds) + "#" + tf.seqs[i].name;
  SimResult tot;
  for (int f = 0; f < folds; ++f) {
    const std::string self = "#f" + std::to_string(f) + "#";
    const Usage u = usage_from_trace(tf, "!" + self, "xval", half_life);
    tot.add(simulate(tf, filter.empty() ? self : self + "&" + filter, u.rank(), c));
  }
  return tot;
}

Json miss_curve(const Trace& t, const std::vector<std::pair<std::string, std::string>>& labels, const Usage& u,
                const std::vector<int64_t>& actives, const SimConfig& base, int folds) {
  const std::vector<int64_t> order = u.rank();
  Json j = Json::object();
  j["schema"] = Json::string("tierstream/miss-curve-v1");
  Json tr = Json::object();
  tr["path"] = Json::string(t.path);
  tr["sha256"] = Json::string(t.sha256);
  tr["tokens"] = Json::number(t.n_tokens());
  j["trace"] = tr;
  Json pol = Json::object();
  pol["model"] = Json::string(
      "device-gated global arena: a demand miss takes a free slot else the bank; at each window boundary LRU victims "
      "until free + in-flight write-backs reach the spare target; FILE victims discarded");
  pol["evict"] = Json::string(base.evict == Evict::Lru ? "lru" : "file-last");
  if (base.evict == Evict::FileLast) pol["file_last_window"] = Json::number(uint64_t(base.file_last_window));
  pol["spares"] = Json::number(int64_t(base.spares));
  pol["wb_windows"] = Json::number(uint64_t(base.wb_windows));
  pol["file_units"] = Json::number(int64_t(base.file_units));
  pol["warm_start"] = Json::string("usage rank");
  pol["window"] = Json::string(base.rows == 1 ? "one token"
                                              : "union of " + std::to_string(base.rows) +
                                                    " consecutive committed tokens (rejected draft rows not modelled)");
  if (folds >= 2) {
    pol["ranking"] = Json::string(std::to_string(folds) + "-fold: sequence i in fold i % " + std::to_string(folds) +
                                  ", ranked by a usage table of the other folds (as the offline reference implementation)");
  } else {
    pol["ranking"] = Json::string("usage table " + u.sha256());
    j["usage_sha256"] = Json::string(u.sha256());
  }
  j["policy"] = pol;
  Json curves = Json::object(), detail = Json::object();
  for (const auto& lf : labels) {
    Json knots = Json::array(), split = Json::array();
    for (int64_t a : actives) {
      SimConfig c = base;
      c.active = a;
      const SimResult r = folds >= 2 ? simulate_xval(t, lf.second, c, folds, u.half_life()) : simulate(t, lf.second, order, c);
      Json k = Json::array();
      k.push(Json::number(int64_t(a)));
      k.push(Json::number(r.milli_per_window(r.miss_pinned + r.miss_file)));
      knots.push(k);
      Json s = Json::object();
      s["active_units"] = Json::number(int64_t(a));
      s["windows"] = Json::number(r.windows);
      s["tokens"] = Json::number(r.tokens);
      s["milli_pinned"] = Json::number(r.milli_per_window(r.miss_pinned));
      s["milli_file"] = Json::number(r.milli_per_window(r.miss_file));
      s["milli_file_uses"] = Json::number(r.milli_per_window(r.file_uses));
      s["milli_bank"] = Json::number(r.milli_per_window(r.bank));
      s["milli_writebacks"] = Json::number(r.milli_per_window(r.evicted));
      s["milli_discards"] = Json::number(r.milli_per_window(r.discarded));
      split.push(s);
    }
    const std::string key = lf.first + "/R" + std::to_string(base.rows);
    curves[key] = knots;  // the contract's shape: [[active_units, milli-misses per window], ...]
    Json cv = Json::object();
    cv["filter"] = Json::string(lf.second);
    cv["split"] = split;
    detail[key] = cv;
  }
  j["curves"] = curves;
  j["detail"] = detail;
  return j;
}

}  // namespace adapt
}  // namespace ts
