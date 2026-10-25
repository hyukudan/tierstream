// Device-gated decode mode: the service thread (dg_service.hpp). Port of wdsrtx's StreamSite::serve (ds41/rtx-replica
// a8cccdc7, src/cuda/ds41_stream_site.cu), with the library's failure protocol instead of the replica's _Exit.

#include "dg_service.hpp"

#include <algorithm>

#include "errors.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <avrt.h>
#include <timeapi.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define SS_PAUSE() _mm_pause()
#else
#define SS_PAUSE() std::this_thread::yield()
#endif

namespace ts {

namespace {

using clock_type = std::chrono::steady_clock;

double us_since(clock_type::time_point a, clock_type::time_point b) {
  return std::chrono::duration<double, std::micro>(b - a).count();
}

#if defined(_WIN32)
// Process-wide facilities (the timer period, the priority class) are shared by every context: the last one to stop
// restores them.
std::mutex g_boost_mu;
int g_timer_refs = 0, g_class_refs = 0;
DWORD g_old_class = 0;
#endif

}  // namespace

DgService::DgService(DgCore* core, const DgMapped& m, const DgServiceConfig& cfg) : core_(core), m_(m), cfg_(cfg) {
  require(cfg_.max_ids >= 1 && cfg_.max_ids <= kDgMaxIds, SS_E_INVALID, "device-gated: max ids outside 1..64");
  ids_.reserve(kDgMaxIds);
  pred_.reserve(kDgMaxIds);
  tl_.assign(core_->gated(), DgLayerTimeline{});
  tl_done_.assign(core_->gated(), DgLayerTimeline{});  // preallocated: publishing it never allocates
}

DgService::~DgService() { stop_join(); }

// The last token handed over before the thread exists: a begin() that races the thread's start is still served.
void DgService::start() {
  const uint32_t seen = host_seq_.load(std::memory_order_acquire);
  th_ = std::thread([this, seen] { run(seen); });
}

void DgService::begin(uint32_t seq) { host_seq_.store(seq, std::memory_order_release); }

DgService::State DgService::wait_terminal(uint32_t seq, std::chrono::milliseconds limit) {
  const auto end = clock_type::now() + limit;
  for (uint32_t spins = 0;; ++spins) {
    const uint64_t t = term_.load(std::memory_order_acquire);
    const State s = State(t >> 32);
    if (s == kFailed || s == kStopped) return s;
    if (s == kServed && uint32_t(t) == seq) return kServed;
    if ((spins & 1023u) == 0 && clock_type::now() > end) return kIdle;
    SS_PAUSE();
  }
}

void DgService::stop_join() {
  stop_.store(true, std::memory_order_release);
  if (m_.host_abort) dg_store_release(m_.host_abort, 1);
  if (th_.joinable()) th_.join();
}

std::string DgService::error() const {
  std::lock_guard<std::mutex> lk(err_mu_);
  return err_;
}

void DgService::publish(State s, uint32_t seq) {
  term_.store(uint64_t(s) << 32 | seq, std::memory_order_release);
}

void DgService::fail_with(const std::string& msg) {
  {
    std::lock_guard<std::mutex> lk(err_mu_);
    if (err_.empty()) err_ = msg;
  }
  // the device stops waiting at once: every later wait of the token sees host_abort
  if (m_.host_abort) dg_store_release(m_.host_abort, 1);
}

void DgService::wait_word(const uint32_t* w, uint32_t seq, const char* what, uint32_t g, bool strict) {
  const auto start = clock_type::now();
  for (uint32_t spins = 0;; ++spins) {
    const uint32_t v = dg_load_acquire(w);
    if (v == seq) return;
    if (strict && v != seq - 1u)
      fail(SS_E_BACKEND, std::string("device-gated: sequence desync: ") + what + " of gated layer " +
                             std::to_string(g) + " holds " + std::to_string(v) + " while serving token " +
                             std::to_string(seq));
    if ((spins & 4095u) == 0) {
      if (stop_.load(std::memory_order_acquire)) throw Stop{};
      if (clock_type::now() - start > cfg_.timeout)
        fail(SS_E_BACKEND, std::string("device-gated: ") + what + " of gated layer " + std::to_string(g) +
                               " never came (token " + std::to_string(seq) + ")");
    }
    SS_PAUSE();
  }
}

void DgService::run(uint32_t seen) {
  try {
    if (cfg_.bind) cfg_.bind();
    boost_on();
  } catch (const std::exception& e) {
    fail_with(e.what());
    publish(kFailed, 0);
    return;
  }
  for (;;) {
    uint32_t seq = 0;
    for (uint32_t spins = 0;; ++spins) {
      seq = host_seq_.load(std::memory_order_acquire);
      if (seq != seen) break;
      if (stop_.load(std::memory_order_acquire)) {
        boost_off();
        publish(kStopped, seen);
        return;
      }
      // idle between tokens: spin briefly (the next token's first export comes within microseconds of ss_begin),
      // then yield the core
      if (spins < 20000) SS_PAUSE();
      else std::this_thread::yield();
    }
    seen = seq;
    try {
      serve_token(seq);
      if (cfg_.diag) std::copy(tl_.begin(), tl_.end(), tl_done_.begin());
    } catch (const Stop&) {
      boost_off();
      publish(kStopped, seq);
      return;
    } catch (const std::exception& e) {
      fail_with(e.what());
      boost_off();
      publish(kFailed, seq);
      return;
    } catch (...) {
      fail_with("device-gated: unknown exception in the service thread");
      boost_off();
      publish(kFailed, seq);
      return;
    }
    publish(kServed, seq);
  }
}

void DgService::serve_token(uint32_t seq) {
  const uint32_t G = core_->gated();
  core_->begin_token(seq);
  auto t_prev = clock_type::now();
  for (uint32_t g = 0; g < G; ++g) {
    DgLayerTimeline t;
    wait_word(&m_.flag[g], seq, "export flag", g, true);
    const auto t_seen = clock_type::now();
    t.seen_us = us_since(t_prev, t_seen);
    const DgRecord& r = m_.rec[g];
    // validate before any key is computed
    require(r.seq == seq, SS_E_BACKEND,
            "device-gated: sequence desync: the record of gated layer " + std::to_string(g) + " is for token " +
                std::to_string(r.seq) + ", serving " + std::to_string(seq));
    const uint32_t n = r.n_ids, n_pred = r.n_pred_promised;
    require(n >= 1 && n <= cfg_.max_ids, SS_E_BACKEND,
            "device-gated: gated layer " + std::to_string(g) + " exported " + std::to_string(n) + " ids");
    require(n_pred <= kDgMaxIds && (n_pred == 0 || g + 1 < G), SS_E_BACKEND,
            "device-gated: gated layer " + std::to_string(g) + " promised " + std::to_string(n_pred) +
                " predicted ids");
    const uint64_t mask = uint64_t(r.mask_lo) | uint64_t(r.mask_hi) << 32;
    ids_.assign(r.ids, r.ids + n);
    const DgStats before = core_->stats();
    core_->serve(g, ids_.data(), n, mask);  // throws on an id outside the layer
    const auto t_demand = clock_type::now();
    t.demand_us = us_since(t_seen, t_demand);
    if (n_pred > 0) {
      wait_word(&m_.pflag[g], seq, "prediction flag", g, false);
      const auto t_pw = clock_type::now();
      t.pred_wait_us = us_since(t_demand, t_pw);
      const DgPredRecord& pr = m_.prec[g];
      require(pr.n_pred == n_pred, SS_E_BACKEND,
              "device-gated: gated layer " + std::to_string(g) + " predicted " + std::to_string(pr.n_pred) +
                  " ids, its export promised " + std::to_string(n_pred));
      pred_.assign(pr.pred, pr.pred + n_pred);
      core_->serve_prediction(g + 1, pred_.data(), n_pred);
      t.prefetch_us = us_since(t_pw, clock_type::now());
    }
    const auto t_end = clock_type::now();
    t.serve_us = us_since(t_seen, t_end);
    if (cfg_.diag) {
      const DgStats& after = core_->stats();
      t.ids = n;
      t.misses = uint32_t(after.misses - before.misses);
      t.covered = uint32_t(after.covered - before.covered);
      t.fetched = uint32_t(after.fetched - before.fetched);
      t.prefetched = uint32_t(after.prefetched - before.prefetched);
      tl_[g] = t;
    }
    t_prev = t_end;
  }
}

void DgService::boost_on() {
#if defined(_WIN32)
  if (cfg_.boost <= 0) return;
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
  DWORD task = 0;
  HANDLE av = AvSetMmThreadCharacteristicsW(L"Pro Audio", &task);
  {
    std::lock_guard<std::mutex> lk(g_boost_mu);
    if (g_timer_refs++ == 0) timeBeginPeriod(1);
    if (cfg_.boost >= 2 && g_class_refs++ == 0) {
      g_old_class = GetPriorityClass(GetCurrentProcess());
      SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }
  }
  av_ = av;
  boosted_ = true;
#endif
}

void DgService::boost_off() {
#if defined(_WIN32)
  if (!boosted_) return;
  if (av_) AvRevertMmThreadCharacteristics(static_cast<HANDLE>(av_));
  av_ = nullptr;
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_NORMAL);
  std::lock_guard<std::mutex> lk(g_boost_mu);
  if (--g_timer_refs == 0) timeEndPeriod(1);
  if (cfg_.boost >= 2 && --g_class_refs == 0) SetPriorityClass(GetCurrentProcess(), g_old_class);
  boosted_ = false;
#endif
}

}  // namespace ts
