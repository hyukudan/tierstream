// ts-adapt: the adaptive memory plan's tools (usage table, miss curves, tiers, solver).
//
//   ts-adapt usage TRACE [--filter F] [--half-life H] [--model M] [--units N] [-o OUT]
//   ts-adapt rank USAGE [--top N]
//   ts-adapt tiers USAGE --active A --host H [-o OUT.json]
//   ts-adapt sim TRACE USAGE --active A [--file F] [--spares S] [--rows R] [--wb W] [--filter F] [--evict lru|file-last]
//   ts-adapt curve TRACE USAGE --actives A,B,.. [--labels code=:code:,all=] [--eval val-] [--file F] [--spares S]
//                  [--rows R] [--evict lru|file-last] [-o OUT]
//   ts-adapt xval TRACE --active A [--folds 5] [--file F] [--spares S] [--rows R] [--half-life H] [--evict E]
//                  k-fold cross-validation as wstream's g0impl.py: fold f = sequences i % folds == f, ranked by a
//                  usage table of the other folds, one arena per fold
//   ts-adapt solve INPUT.json [-o OUT] [--table]
//   ts-adapt ref FIXTURE.json            (placement contract §3.9 arithmetic alone, = golden/solver_ref.py)
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "adapt_sim.hpp"
#include "adapt_solve.hpp"
#include "adapt_usage.hpp"
#include "errors.hpp"
#include "json.hpp"

using namespace ts;
using namespace ts::adapt;

namespace {

int usage_text() {
  std::fprintf(stderr,
               "usage: ts-adapt usage|rank|tiers|sim|curve|solve|ref ... (see tools/ts_adapt.cpp header)\n");
  return 2;
}

struct Args {
  std::vector<std::string> pos;
  std::map<std::string, std::string> opt;
  bool has(const std::string& k) const { return opt.count(k) != 0; }
  std::string get(const std::string& k, const std::string& d = "") const {
    auto it = opt.find(k);
    return it == opt.end() ? d : it->second;
  }
  int64_t num(const std::string& k, int64_t d) const {
    auto it = opt.find(k);
    if (it == opt.end()) return d;
    char* e = nullptr;
    long long v = std::strtoll(it->second.c_str(), &e, 10);
    require(e && *e == 0, SS_E_INVALID, "--" + k + ": expected an integer");
    return v;
  }
};

Args parse(int argc, char** argv, int from) {
  Args a;
  for (int i = from; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "-o") s = "--out";
    if (s.rfind("--", 0) == 0) {
      std::string k = s.substr(2);
      if (k == "table") {
        a.opt[k] = "1";
        continue;
      }
      require(i + 1 < argc, SS_E_INVALID, s + ": missing value");
      a.opt[k] = argv[++i];
    } else {
      a.pos.push_back(s);
    }
  }
  return a;
}

std::string read_file(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  require(bool(f), SS_E_IO, "cannot open " + p);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

void emit(const Args& a, const std::string& text) {
  if (a.has("out")) {
    std::ofstream f(a.get("out"), std::ios::binary);
    require(bool(f), SS_E_IO, "cannot write " + a.get("out"));
    f << text;
  } else {
    std::fwrite(text.data(), 1, text.size(), stdout);
    std::fputc('\n', stdout);
  }
}

std::vector<int64_t> list(const std::string& s) {
  std::vector<int64_t> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ','))
    if (!t.empty()) v.push_back(std::stoll(t));
  return v;
}

Evict evict_of(const std::string& s) {
  if (s.empty() || s == "lru") return Evict::Lru;
  require(s == "file-last", SS_E_INVALID, "--evict: lru | file-last");
  return Evict::FileLast;
}

double gib(uint64_t b) { return double(b) / double(uint64_t(1) << 30); }

void print_table(const Json& out) {
  std::printf("%-22s %7s %5s %9s %6s %9s %6s %8s %6s  %-6s %9s %9s %7s %7s %-12s\n", "choice", "ctx", "mode",
              "state GiB", "active", "arena GiB", "host", "host GiB", "file", "lang", "miss/tok", "file/tok", "tok/s",
              "%floor", "source");
  for (const Json& r : out.find("rows")->arr) {
    const std::string st = r.find("status")->text;
    auto u = [&](const char* k) { return r.find(k) ? std::strtoull(r.find(k)->text.c_str(), nullptr, 10) : 0ull; };
    if (st != "ok") {
      std::printf("%-22s %7llu %5s %9.3f %s (overflow %.3f GiB)\n", r.find("id")->text.c_str(), u("ctx"),
                  r.find("mode")->text.c_str(), gib(u("state_bytes")), st.c_str(), gib(u("overflow_bytes")));
      continue;
    }
    for (const auto& kv : r.find("predicted")->obj) {
      for (const char* src : {"simulated", "contract"}) {
        const Json* p = kv.second.find(src);
        if (!p || !p->is_object()) continue;
        auto pu = [&](const char* k) { return p->find(k) ? std::strtoull(p->find(k)->text.c_str(), nullptr, 10) : 0ull; };
        const uint64_t total = p->find("milli_misses_total") ? pu("milli_misses_total")
                                                              : pu("milli_misses_pinned") + pu("milli_misses_file");
        const uint64_t file = p->find("milli_misses_file") ? pu("milli_misses_file") : pu("milli_misses_file_bound");
        if (!p->find("milli_tok_s")) {
          std::printf("%-22s %7llu %5s %9.3f %6llu %9.2f %6llu %8.2f %6llu  %-6s %9.3f %9.3f %7s %7s %s (unpredicted)\n",
                      r.find("id")->text.c_str(), u("ctx"), r.find("mode")->text.c_str(), gib(u("state_bytes")),
                      u("active_units"), gib(u("arena_bytes")), u("host_units"), gib(u("host_pinned_bytes")),
                      u("file_units"), kv.first.c_str(), total / 1000.0, file / 1000.0, "-", "-", src);
          continue;
        }
        std::printf("%-22s %7llu %5s %9.3f %6llu %9.2f %6llu %8.2f %6llu  %-6s %9.3f %9.3f %7.2f %7.1f %-12s\n",
                    r.find("id")->text.c_str(), u("ctx"), r.find("mode")->text.c_str(), gib(u("state_bytes")),
                    u("active_units"), gib(u("arena_bytes")), u("host_units"), gib(u("host_pinned_bytes")),
                    u("file_units"), kv.first.c_str(), total / 1000.0, file / 1000.0, pu("milli_tok_s") / 1000.0,
                    pu("floor_pct_milli") / 1000.0,
                    (std::string(src) + (p->find("tag") ? "/" + p->find("tag")->text : "")).c_str());
      }
    }
  }
}

int run(int argc, char** argv) {
  if (argc < 2) return usage_text();
  const std::string cmd = argv[1];
  Args a = parse(argc, argv, 2);
  if (cmd == "usage") {
    require(a.pos.size() == 1, SS_E_INVALID, "usage: one trace");
    Trace t = trace_load(a.pos[0], int(a.num("units", 0)));
    Usage u = usage_from_trace(t, a.get("filter"), a.get("model", "unknown"), uint32_t(a.num("half-life", 256)));
    emit(a, u.to_text());
    std::fprintf(stderr, "usage: %llu tokens, %d layers x %d units, sha256 %s\n",
                 static_cast<unsigned long long>(u.tokens()), u.n_layers(), t.n_units, u.sha256().c_str());
    return 0;
  }
  if (cmd == "rank") {
    require(a.pos.size() == 1, SS_E_INVALID, "rank: one usage table");
    Usage u = Usage::from_text(read_file(a.pos[0]));
    const std::vector<int64_t> r = u.rank();
    const int64_t top = a.num("top", int64_t(r.size()));
    std::string s;
    for (int64_t i = 0; i < top && i < int64_t(r.size()); ++i) {
      const int64_t g = r[size_t(i)];
      int L = 0;
      while (L + 1 < u.n_layers() && u.base(L + 1) <= g) ++L;
      s += std::to_string(i) + " " + std::to_string(L) + ":" + std::to_string(g - u.base(L)) + " use " +
           std::to_string(u.use(g)) + " recency " + std::to_string(u.recency(g)) + "\n";
    }
    std::fwrite(s.data(), 1, s.size(), stdout);
    return 0;
  }
  if (cmd == "tiers") {
    require(a.pos.size() == 1, SS_E_INVALID, "tiers: one usage table");
    Usage u = Usage::from_text(read_file(a.pos[0]));
    const std::vector<uint8_t> t = tiers_from_rank(u.rank(), u.total_units(), a.num("active", 0), a.num("host", 0));
    Json j = Json::object();
    j["schema"] = Json::string("tierstream/tiers-v1");
    j["usage_sha256"] = Json::string(u.sha256());
    Json arr = Json::array();
    for (uint8_t x : t) arr.push(Json::number(uint64_t(x)));
    j["tiers"] = arr;
    emit(a, json_write(j));
    return 0;
  }
  if (cmd == "sim") {
    require(a.pos.size() == 2, SS_E_INVALID, "sim: TRACE USAGE");
    Usage u = Usage::from_text(read_file(a.pos[1]));
    Trace t = trace_load(a.pos[0], int(u.layer_units(0)));
    SimConfig c;
    c.active = a.num("active", 0);
    c.file_units = a.num("file", 0);
    c.spares = a.num("spares", 96);
    c.rows = uint32_t(a.num("rows", 1));
    c.wb_windows = uint32_t(a.num("wb", 1));
    c.evict = evict_of(a.get("evict"));
    c.file_last_window = uint32_t(a.num("window", 64));
    SimResult r = simulate(t, a.get("filter"), u.rank(), c);
    std::printf("windows %llu tokens %llu misses/window %.3f (pinned %.3f file %.3f) file-uses %.3f bank %.3f "
                "writebacks %.3f discards %.3f\n",
                static_cast<unsigned long long>(r.windows), static_cast<unsigned long long>(r.tokens),
                r.milli_per_window(r.miss_pinned + r.miss_file) / 1000.0, r.milli_per_window(r.miss_pinned) / 1000.0,
                r.milli_per_window(r.miss_file) / 1000.0, r.milli_per_window(r.file_uses) / 1000.0,
                r.milli_per_window(r.bank) / 1000.0, r.milli_per_window(r.evicted) / 1000.0,
                r.milli_per_window(r.discarded) / 1000.0);
    return 0;
  }
  if (cmd == "xval") {
    require(a.pos.size() == 1, SS_E_INVALID, "xval: TRACE");
    Trace t = trace_load(a.pos[0], int(a.num("units", 384)));
    const int folds = int(a.num("folds", 5));
    SimConfig c;
    c.active = a.num("active", 0);
    c.file_units = a.num("file", 0);
    c.spares = a.num("spares", 96);
    c.rows = uint32_t(a.num("rows", 1));
    c.wb_windows = uint32_t(a.num("wb", 1));
    c.evict = evict_of(a.get("evict"));
    c.file_last_window = uint32_t(a.num("window", 64));
    SimResult tot;
    for (int f = 0; f < folds; ++f) {
      // Tag every sequence with its fold so the name filters select it.
      Trace tf = t;
      for (size_t i = 0; i < tf.seqs.size(); ++i) tf.seqs[i].name = "#f" + std::to_string(int(i) % folds) + "#" + tf.seqs[i].name;
      const std::string self = "#f" + std::to_string(f) + "#";
      Usage u = usage_from_trace(tf, "!" + self, "xval", uint32_t(a.num("half-life", 256)));
      std::string filt = self;
      if (a.has("filter")) filt += "&" + a.get("filter");
      SimResult r = simulate(tf, filt, u.rank(), c);
      tot.windows += r.windows;
      tot.tokens += r.tokens;
      tot.lookups += r.lookups;
      tot.miss_pinned += r.miss_pinned;
      tot.miss_file += r.miss_file;
      tot.bank += r.bank;
      tot.evicted += r.evicted;
      tot.discarded += r.discarded;
      tot.file_uses += r.file_uses;
    }
    std::printf("xval folds %d active %lld file %lld: windows %llu misses/window %.3f (pinned %.3f file %.3f) "
                "file-uses %.3f bank %.3f writebacks %.3f discards %.3f\n",
                folds, static_cast<long long>(c.active), static_cast<long long>(c.file_units),
                static_cast<unsigned long long>(tot.windows), tot.milli_per_window(tot.miss_pinned + tot.miss_file) / 1000.0,
                tot.milli_per_window(tot.miss_pinned) / 1000.0, tot.milli_per_window(tot.miss_file) / 1000.0,
                tot.milli_per_window(tot.file_uses) / 1000.0, tot.milli_per_window(tot.bank) / 1000.0,
                tot.milli_per_window(tot.evicted) / 1000.0, tot.milli_per_window(tot.discarded) / 1000.0);
    return 0;
  }
  if (cmd == "curve") {
    require(a.pos.size() == 2, SS_E_INVALID, "curve: TRACE USAGE");
    Usage u = Usage::from_text(read_file(a.pos[1]));
    Trace t = trace_load(a.pos[0], int(u.layer_units(0)));
    std::vector<std::pair<std::string, std::string>> labels;
    const std::string ev = a.get("eval");
    std::stringstream ls(a.get("labels", "all="));
    std::string item;
    while (std::getline(ls, item, ',')) {
      const size_t eq = item.find('=');
      require(eq != std::string::npos, SS_E_INVALID, "--labels: name=filter,...");
      std::string f = item.substr(eq + 1);
      if (!ev.empty()) f = f.empty() ? ev : ev + "&" + f;
      labels.emplace_back(item.substr(0, eq), f);
    }
    SimConfig c;
    c.file_units = a.num("file", 0);
    c.spares = a.num("spares", 96);
    c.rows = uint32_t(a.num("rows", 1));
    c.wb_windows = uint32_t(a.num("wb", 1));
    c.evict = evict_of(a.get("evict"));
    c.file_last_window = uint32_t(a.num("window", 64));
    emit(a, json_write(miss_curve(t, labels, u, list(a.get("actives")), c, int(a.num("folds", 0)))));
    return 0;
  }
  if (cmd == "solve") {
    require(a.pos.size() == 1, SS_E_INVALID, "solve: one input");
    const std::string base = std::filesystem::path(a.pos[0]).parent_path().string();
    Json out = solve(json_parse(read_file(a.pos[0])), base);
    if (a.has("out")) emit(a, json_write(out));
    if (a.has("table") || !a.has("out")) print_table(out);
    return 0;
  }
  if (cmd == "ref") {
    require(a.pos.size() == 1, SS_E_INVALID, "ref: one fixture");
    emit(a, json_write(solve_ref(json_parse(read_file(a.pos[0])))));
    return 0;
  }
  return usage_text();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ts-adapt: %s\n", e.what());
    return 1;
  }
}
