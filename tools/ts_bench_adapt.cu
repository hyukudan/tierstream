// ts-bench-adapt: replay recorded routing traces through tierstream on a CUDA device with a usage-ranked
// VRAM / pinned / NVMe placement, a compute stub per layer and optional next-layer lookahead, and report ms/token,
// misses per tier and a bitwise sample check (landed bytes == file bytes, i.e. == the all-resident bytes).
//
//   ts-bench-adapt --desc D.json --trace T.txt --usage U.json --active A --host H [--filter val-&:code:]
//                  [--tokens N] [--warmup W] [--layer-us 154.5] [--ahead K] [--recall 0.79] [--hostpf 1]
//                  [--specpf 0] [--readahead 8] [--readers 4] [--dyn 2] [--ring 48] [--spares 96]
//                  [--mode exclusive|inclusive] [--check-every 50] --device-gib X --host-gib Y [--out R.json]
//                  [--max-seconds 600] [--force 1] [--dry 1]
//
// Shared-GPU safety: --device-gib / --host-gib are required (no defaults); the device must be an RTX PRO 6000 unless
// --force 1; the plan's device peak + 1 GiB must fit the device's free memory and the plan's pinned bytes must fit
// --host-gib, else nothing is allocated; the replay stops after --max-seconds (default 600). Run it only inside a
// PC-BIG slot.
//
// --active: units placed RESIDENT (top of the usage ranking), --host: next units PINNED, the rest DISK (file homes).
// Lookahead: after layer L's routes, layer L+1..L+K's ids are hinted (the true ids, each replaced by a wrong one with
// probability 1 - recall, deterministic): disk units are read ahead into pinned buffers (--hostpf), and with --specpf
// the hinted units are also copied speculatively into ring slots.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "adapt_sim.hpp"
#include "adapt_usage.hpp"
#include "json.hpp"
#include "tierstream/ts.h"
#include "tierstream/ts_adapt.h"

namespace {

__global__ void spin_ns(unsigned long long ns) {
  unsigned long long t0, t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  do {
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  } while (t1 - t0 < ns);
}

void ok(ss_status s, const char* what) {
  if (s != SS_OK) {
    std::fprintf(stderr, "ts-bench-adapt: %s: %s\n", what, ss_last_error());
    std::exit(1);
  }
}

void cu(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "ts-bench-adapt: %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

std::string slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", p.c_str());
    std::exit(1);
  }
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Args {
  std::map<std::string, std::string> m;
  std::string s(const std::string& k, const std::string& d = "") const {
    auto it = m.find(k);
    return it == m.end() ? d : it->second;
  }
  double f(const std::string& k, double d) const {
    auto it = m.find(k);
    if (it == m.end()) return d;
    char* e = nullptr;
    const double v = std::strtod(it->second.c_str(), &e);
    if (!e || *e || !std::isfinite(v) || v < 0) bad(k);
    return v;
  }
  long long i(const std::string& k, long long d) const {
    auto it = m.find(k);
    if (it == m.end()) return d;
    char* e = nullptr;
    const long long v = std::strtoll(it->second.c_str(), &e, 10);
    if (!e || *e || v < 0) bad(k);
    return v;
  }
  [[noreturn]] static void bad(const std::string& k) {
    std::fprintf(stderr, "ts-bench-adapt: --%s: expected a non-negative number\n", k.c_str());
    std::exit(2);
  }
};

double pct(std::vector<double> v, double q) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  size_t k = size_t(std::min<double>(double(v.size() - 1), std::floor(q * double(v.size() - 1) + 0.5)));
  return v[k];
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (argc % 2 == 0) {
    std::fprintf(stderr, "ts-bench-adapt: options come in --name value pairs\n");
    return 2;
  }
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i];
    if (k.rfind("--", 0) != 0 || k.size() < 3) {
      std::fprintf(stderr, "ts-bench-adapt: unexpected argument %s\n", k.c_str());
      return 2;
    }
    a.m[k.substr(2)] = argv[i + 1];
  }
  for (const char* req : {"desc", "trace", "usage", "device-gib", "host-gib"})
    if (!a.m.count(req)) {
      std::fprintf(stderr, "ts-bench-adapt: --%s is required\n", req);
      return 2;
    }
  const std::string desc_text = slurp(a.s("desc"));
  ss_desc* d = nullptr;
  ok(ss_desc_from_json(desc_text.data(), desc_text.size(), &d), "descriptor");
  const int L = ss_desc_n_layers(d);
  const int64_t n = ss_desc_total_units(d);
  const uint64_t ub = ss_desc_unit_bytes(d, 0);
  ts::adapt::Usage usage = ts::adapt::Usage::from_text(slurp(a.s("usage")));
  if (usage.total_units() != n) {
    std::fprintf(stderr, "usage table shape differs from the descriptor\n");
    return 1;
  }
  ts::adapt::Trace trace = ts::adapt::trace_load(a.s("trace"), ss_desc_layer_units(d, 0));
  const int64_t active = a.i("active", 0), host = a.i("host", 0);
  const std::vector<uint8_t> tiers = ts::adapt::tiers_from_rank(usage.rank(), n, active, host);
  const bool excl = a.s("mode", "exclusive") == "exclusive";

  ss_budget b;
  ss_budget_defaults(&b);
  b.margin_gib = 0.5;
  b.margin_fraction = 0;
  b.device_bytes = uint64_t(a.f("device-gib", 0) * double(uint64_t(1) << 30));
  b.host_pinned_bytes = uint64_t(a.f("host-gib", 0) * double(uint64_t(1) << 30));
  b.dynamic_slots_per_layer = uint32_t(a.i("dyn", 2));
  b.ring_slots = uint32_t(a.i("ring", 48));
  b.tier_mode = excl ? SS_TIER_MODE_EXCLUSIVE : SS_TIER_MODE_INCLUSIVE;
  b.host_spare_units = uint32_t(a.i("spares", 96));
  b.pinned_backing = 0;
  ss_plan* p = nullptr;
  ok(ss_plan_from_tiers(d, &b, tiers.data(), tiers.size(), &p), "plan");
  ss_plan_totals tot{};
  ss_plan_get_totals(p, &tot);
  std::printf("plan: resident %llu (%.2f GiB) pinned %llu (%.2f GiB) disk %llu (%.2f GiB) device peak %.2f GiB ring %.2f "
              "dyn %.2f\n",
              (unsigned long long)tot.n_resident, tot.resident_bytes / 1073741824.0, (unsigned long long)tot.n_pinned,
              tot.pinned_bytes / 1073741824.0, (unsigned long long)tot.n_disk, tot.disk_bytes / 1073741824.0,
              tot.peak_device_bytes / 1073741824.0, tot.ring_bytes / 1073741824.0, tot.dynamic_bytes / 1073741824.0);
  if (a.s("dry", "0") == "1") return 0;

  ss_options o;
  ss_options_defaults(&o);
  o.backend = SS_BACKEND_CUDA;
  o.device = int(a.i("device", 0));
  o.pin_planned = 0;
  o.reader_threads = uint32_t(a.i("readers", 4));
  o.readahead_units = uint32_t(a.i("readahead", 8));
  o.lookahead_budget = uint32_t(a.i("spec-budget", 6));
  o.staging_buffers = uint32_t(a.i("staging", 4));
  o.staging_bytes = uint64_t(a.i("staging-mib", 32)) << 20;
  o.verify_hash = 0;  // no descriptor hashes for the GGUF: the sample check below compares with the file bytes
  cu(cudaSetDevice(o.device), "set device");
  {
    cudaDeviceProp prop{};
    cu(cudaGetDeviceProperties(&prop, o.device), "device properties");
    if (std::string(prop.name).find("RTX PRO 6000") == std::string::npos && a.s("force", "0") != "1") {
      std::fprintf(stderr, "ts-bench-adapt: device %d is '%s', not an RTX PRO 6000 (--force 1 to override)\n", o.device,
                   prop.name);
      return 3;
    }
    size_t free_b = 0, total_b = 0;
    cu(cudaMemGetInfo(&free_b, &total_b), "memory info");
    const uint64_t need = tot.peak_device_bytes + (uint64_t(1) << 30);
    if (free_b < need) {
      std::fprintf(stderr, "ts-bench-adapt: device %d has %.2f GiB free, the plan needs %.2f GiB (+1 GiB margin)\n",
                   o.device, free_b / 1073741824.0, need / 1073741824.0);
      return 3;
    }
    if (tot.pinned_bytes > b.host_pinned_bytes) {
      std::fprintf(stderr, "ts-bench-adapt: the plan pins %.2f GiB > --host-gib\n", tot.pinned_bytes / 1073741824.0);
      return 3;
    }
  }
  cudaStream_t s;
  cu(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
  spin_ns<<<1, 1, 0, s>>>(1000);  // load the module before any timed step
  cu(cudaStreamSynchronize(s), "warm kernel");

  auto t_open0 = std::chrono::steady_clock::now();
  ss_ctx* c = nullptr;
  ok(ss_open(d, &b, p, &o, &c), "open");
  const double open_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_open0).count();
  std::printf("open %.1f s (direct io: %d)\n", open_s, ss_source_direct(c, 0));
  ts_usage* online = nullptr;
  if (a.s("online", "1") == "1") {
    ok(ts_usage_new_for(d, 256, &online), "usage");
    ok(ts_usage_attach(c, online), "attach");
  }

  const std::string filter = a.s("filter", "val-");
  const int64_t max_tokens = a.i("tokens", 1000), warmup = a.i("warmup", 100);
  const unsigned long long layer_ns = (unsigned long long)(a.f("layer-us", 154.5) * 1000.0);
  const int ahead = int(a.i("ahead", 1));
  const double recall = a.f("recall", 1.0);
  const bool hostpf = a.s("hostpf", "1") == "1", specpf = a.s("specpf", "0") == "1";
  const int check_every = int(a.i("check-every", 50));
  const int E = ss_desc_layer_units(d, 0), K = trace.top_k;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> uni(0, 1);

  std::vector<double> ms;
  ss_layer_stats st0{}, st1{};
  uint64_t checked = 0, check_fail = 0, tokens = 0, hinted = 0;
  std::vector<int32_t> hint;
  auto t_run0 = std::chrono::steady_clock::now();
  const auto t_start = t_run0;
  const double max_seconds = a.f("max-seconds", 600);
  bool timed_out = false;
  for (const auto& sq : trace.seqs) {
    if (timed_out) break;
    if (!ts::adapt::seq_match(sq.name, filter)) continue;
    for (uint32_t k = 0; k < sq.tokens && int64_t(tokens) < max_tokens + warmup; ++k) {
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() > max_seconds) {
        std::fprintf(stderr, "ts-bench-adapt: --max-seconds reached after %llu tokens\n", (unsigned long long)tokens);
        timed_out = true;
        break;
      }
      if (int64_t(tokens) == warmup) {
        ok(ss_stats_total(c, &st0), "stats");
        t_run0 = std::chrono::steady_clock::now();
      }
      const uint64_t tok = sq.first + k;
      auto t0 = std::chrono::steady_clock::now();
      ok(ss_begin(c, nullptr, 0), "begin");
      for (int l = 0; l < L; ++l) {
        ok(ss_layer_routes(c, l, trace.at(tok, l), K), "routes");
        for (int j = 1; j <= ahead && l + j < L; ++j) {
          hint.clear();
          const int32_t* nx = trace.at(tok, l + j);
          for (int q = 0; q < K; ++q) {
            int32_t e = nx[q];
            if (uni(rng) >= recall) e = int32_t(rng() % uint32_t(E));
            if (std::find(hint.begin(), hint.end(), e) == hint.end()) hint.push_back(e);
          }
          hinted += hint.size();
          if (hostpf) ok(ss_host_prefetch(c, l + j, hint.data(), int(hint.size())), "host prefetch");
          if (specpf && j == 1) ok(ss_prefetch(c, l + j, hint.data(), int(hint.size()), SS_PREFETCH_SPECULATIVE), "spec");
        }
        ss_layer_view v{};
        ok(ss_acquire(c, l, s, &v), "acquire");
        spin_ns<<<1, 1, 0, s>>>(layer_ns);
        if (check_every > 0 && tokens % uint64_t(check_every) == 0 && l == int(tokens / check_every) % L) {
          cu(cudaStreamSynchronize(s), "sync check");
          const int32_t* ids = trace.at(tok, l);
          for (int q = 0; q < K; ++q) {
            ++checked;
            if (ss_check_unit(c, l, ids[q]) != SS_OK) {
              ++check_fail;
              std::fprintf(stderr, "check (%d, %d): %s\n", l, ids[q], ss_last_error());
            }
          }
        }
        ok(ss_release(c, l, s), "release");
      }
      ok(ss_end(c), "end");
      cu(cudaStreamSynchronize(s), "sync");
      const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      if (int64_t(tokens) >= warmup) ms.push_back(dt);
      ++tokens;
    }
    if (int64_t(tokens) >= max_tokens + warmup) break;
  }
  const double run_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_run0).count();
  ok(ss_stats_total(c, &st1), "stats");
  const double nt = double(ms.size());
  if (ms.empty()) {
    std::fprintf(stderr, "ts-bench-adapt: no timed token (filter matched too few tokens or warm-up too long)\n");
    ss_close(c);
    return 4;
  }
  double mean = 0;
  for (double x : ms) mean += x;
  mean /= std::max(1.0, nt);
  const double miss_pinned = double(st1.bytes_pinned - st0.bytes_pinned) / double(ub) / nt;
  const double miss_disk = double(st1.bytes_disk - st0.bytes_disk) / double(ub) / nt;
  const double admitted = double(st1.admitted - st0.admitted) / nt, transient = double(st1.transient - st0.transient) / nt;
  const double routed = double(st1.routed - st0.routed) / nt, hits = double(st1.hits - st0.hits) / nt;
  const double floor_ms = a.f("floor-ms", 6.18), w3 = a.f("w3-tok-s", 47.8);
  ss_exclusive_stats xs{};
  ss_stats_exclusive(c, &xs);
  char* js = nullptr;
  ss_stats_to_json(c, &js, nullptr);
  std::printf("tokens %d (warmup %lld) ms/token mean %.3f p50 %.3f p90 %.3f p99 %.3f -> %.2f tok/s = %.1f %% of the "
              "%.2f ms floor, %.1f %% of w3 %.1f tok/s\n",
              int(nt), (long long)warmup, mean, pct(ms, 0.5), pct(ms, 0.9), pct(ms, 0.99), 1000.0 / mean,
              100.0 * floor_ms / mean, floor_ms, 100.0 * (1000.0 / mean) / w3, w3);
  std::printf("per token: routed %.2f hits %.2f admitted %.3f transient %.3f | units from pinned %.3f from NVMe %.3f | "
              "hinted %.2f | exclusive writebacks %llu deferred %llu | check %llu units, %llu failed | run %.1f s\n",
              routed, hits, admitted, transient, miss_pinned, miss_disk, double(hinted) / double(tokens),
              (unsigned long long)xs.writebacks, (unsigned long long)xs.deferred, (unsigned long long)checked,
              (unsigned long long)check_fail, run_s);
  if (a.m.count("out")) {
    ts::Json r = ts::Json::object();
    r["schema"] = ts::Json::string("tierstream/bench-adapt-v1");
    for (const auto& kv : a.m) r["args"][kv.first] = ts::Json::string(kv.second);
    r["tokens"] = ts::Json::number(uint64_t(nt));
    r["ms_mean"] = ts::Json::number(mean);
    r["ms_p50"] = ts::Json::number(pct(ms, 0.5));
    r["ms_p90"] = ts::Json::number(pct(ms, 0.9));
    r["ms_p99"] = ts::Json::number(pct(ms, 0.99));
    r["tok_s"] = ts::Json::number(1000.0 / mean);
    r["floor_pct"] = ts::Json::number(100.0 * floor_ms / mean);
    r["vs_w3_pct"] = ts::Json::number(100.0 * (1000.0 / mean) / w3);
    r["miss_pinned_per_token"] = ts::Json::number(miss_pinned);
    r["miss_nvme_per_token"] = ts::Json::number(miss_disk);
    r["admitted_per_token"] = ts::Json::number(admitted);
    r["transient_per_token"] = ts::Json::number(transient);
    r["routed_per_token"] = ts::Json::number(routed);
    r["hits_per_token"] = ts::Json::number(hits);
    r["checked_units"] = ts::Json::number(checked);
    r["check_failures"] = ts::Json::number(check_fail);
    r["open_s"] = ts::Json::number(open_s);
    r["n_resident"] = ts::Json::number(uint64_t(tot.n_resident));
    r["n_pinned"] = ts::Json::number(uint64_t(tot.n_pinned));
    r["n_disk"] = ts::Json::number(uint64_t(tot.n_disk));
    r["usage_sha256"] = ts::Json::string(usage.sha256());
    r["trace_sha256"] = ts::Json::string(trace.sha256);
    r["stats"] = ts::json_parse(js ? js : "{}");
    std::ofstream f(a.s("out"), std::ios::binary);
    f << ts::json_write(r);
  }
  ss_free(js);
  if (online) {
    ts_usage_attach(c, nullptr);
    if (a.m.count("usage-out")) {
      char* u = nullptr;
      ts_usage_to_json(online, &u, nullptr);
      std::ofstream f(a.s("usage-out"), std::ios::binary);
      f << u;
      ss_free(u);
    }
    ts_usage_free(online);
  }
  ss_close(c);
  ss_plan_free(p);
  ss_desc_free(d);
  return check_fail ? 2 : 0;
}
