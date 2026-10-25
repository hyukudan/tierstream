// Device-gated decode mode: the service thread (docs/design.md §3.7, design rev 3.2 §4). It waits for each gated
// layer's export flag, validates the record, and runs the core's demand and prediction work; the transport puts the
// copies, publications and the ready word on the device. Its terminal word tells the engine thread how each token
// ended (Served / Failed / Stopped); after Failed or Stopped the thread has returned and never touches the core again.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "dg_core.hpp"
#include "dg_proto.hpp"

namespace ts {

struct DgServiceConfig {
  uint32_t max_ids = kDgMaxIds;
  std::chrono::milliseconds timeout{10000};  // per flag / pflag wait
  int boost = 0;                              // 0 off, 1 thread-only, 2 + process priority class (Windows)
  bool diag = false;                          // per-layer timeline of the last served token
  std::function<void()> bind;                 // device binding on the service thread (cudaSetDevice)
};

// The replica's per-layer timeline (StreamSite): microseconds on the service thread.
struct DgLayerTimeline {
  double seen_us = 0, demand_us = 0, pred_wait_us = 0, prefetch_us = 0, serve_us = 0;
  double dma_ms = -1;  // filled by ss_end from the transport's marks (-1 = not measured)
  uint32_t ids = 0, misses = 0, covered = 0, fetched = 0, prefetched = 0, predicted_true = 0, covered_waits = 0;
};

class DgService {
 public:
  enum State : uint32_t { kIdle = 0, kServed = 1, kFailed = 2, kStopped = 3 };

  DgService(DgCore* core, const DgMapped& m, const DgServiceConfig& cfg);
  ~DgService();  // stop_join
  DgService(const DgService&) = delete;
  DgService& operator=(const DgService&) = delete;

  void start();
  // Engine thread: hands token `seq` to the service (release). The service owns the core until it publishes the
  // terminal word for seq.
  void begin(uint32_t seq);
  // Engine thread: the terminal word for `seq`, waiting at most `limit`. kIdle = not yet (deadline passed).
  State wait_terminal(uint32_t seq, std::chrono::milliseconds limit);
  // Requests stop, sets host_abort and joins the thread. Idempotent. After it, the core belongs to the engine thread.
  void stop_join();
  bool running() const { return th_.joinable(); }
  std::string error() const;
  // The timeline of the last Served token (diag), engine thread, between tokens.
  const std::vector<DgLayerTimeline>& timeline() const { return tl_done_; }
  // Engine thread with terminal ownership (or before the first begin). Sequence/protocol words are retained.
  void invalidate_timeline() {
    for (auto& t : tl_) t = DgLayerTimeline{};
    for (auto& t : tl_done_) t = DgLayerTimeline{};
    new_origin_.store(true, std::memory_order_release);
  }
  uint32_t gated() const { return core_->gated(); }

 private:
  struct Stop {};
  void run(uint32_t seen);
  void serve_token(uint32_t seq);
  // Waits until *w == seq (acquire); the deadline and the stop request end the wait. strict (the export flag, written
  // every token): any value other than seq - 1 or seq is a desync. The prediction flag is written only on tokens that
  // predict, so it may hold any older token.
  void wait_word(const uint32_t* w, uint32_t seq, const char* what, uint32_t g, bool strict);
  void wait_began(uint32_t seq);
  void publish(State s, uint32_t seq);
  void fail_with(const std::string& msg);
  void boost_on();
  void boost_off();

  DgCore* core_;
  DgMapped m_;
  DgServiceConfig cfg_;
  std::thread th_;
  std::chrono::steady_clock::time_point t_served_{};  // the previous token's end (seen_us[0]'s origin)
  std::atomic<uint32_t> host_seq_{0};
  std::atomic<uint64_t> term_{0};  // (state << 32) | seq
  std::atomic<bool> stop_{false};
  mutable std::mutex err_mu_;
  std::string err_;
  std::vector<uint32_t> ids_, pred_;
  std::vector<DgLayerTimeline> tl_, tl_done_;
  std::atomic<bool> new_origin_{false};
  bool boosted_ = false;
  void* av_ = nullptr;  // the MMCSS task handle (Windows)
};

}  // namespace ts
