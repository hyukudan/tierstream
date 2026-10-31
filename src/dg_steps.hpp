// Device-gated decode mode: the context's per-layer step bookkeeping (dg_ctx.cpp), free of any device API so the CPU
// tests drive it with the core and the service: the keep sets (ss_dg_keep / ss_dg_make_resident), which layers hold a
// resident set (never exported), the layers each step exports (ss_begin's rows), and the export flag value each layer
// must hold between steps (the last sequence that exported it).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ts {

class DgSteps {
 public:
  void init(uint32_t gated) {
    keep_.assign(gated, {});
    resident_.assign(gated, 0);
    run_.assign(gated, 1);
    flag_seq_.assign(gated, 0);
  }
  uint32_t gated() const { return uint32_t(keep_.size()); }

  // ss_begin, before the desync check: the layers the step exports. rows (NULL = every layer) is indexed by
  // descriptor layer, `layer_of[g]` the descriptor layer of gated layer g. A held resident set is never exported.
  const std::vector<uint8_t>& plan(const uint32_t* rows, const std::vector<int>& layer_of) {
    for (size_t g = 0; g < run_.size(); ++g)
      run_[g] = uint8_t((!rows || rows[size_t(layer_of[g])] > 0) && !resident_[g]);
    return run_;
  }
  // Every export flag holds the last sequence that exported its layer (a stray export shows here).
  bool flags_clean(const uint32_t* flag, uint32_t (*load)(const uint32_t*)) const {
    for (size_t g = 0; g < flag_seq_.size(); ++g)
      if (load(flag + g) != flag_seq_[g]) return false;
    return true;
  }
  // ss_begin, after the desync check: the step `seq` exports the planned layers.
  void begin(uint32_t seq) {
    for (size_t g = 0; g < run_.size(); ++g)
      if (run_[g]) flag_seq_[g] = seq;
  }
  const std::vector<uint8_t>& run() const { return run_; }
  bool exported(uint32_t g) const { return run_[g] != 0; }

  // ss_dg_keep: a plain keep (g is exported again); empty = the release.
  void keep(uint32_t g, std::vector<uint32_t> keys) {
    keep_[g] = std::move(keys);
    resident_[g] = 0;
  }
  // ss_dg_make_resident: the keys the call must not evict, every layer's keep including g's current one (it stays
  // protected until the new set is in place: a failed call leaves g's keep, and its units, as they were).
  std::vector<uint32_t> protect() const {
    std::vector<uint32_t> v;
    for (const auto& k : keep_) v.insert(v.end(), k.begin(), k.end());
    return v;
  }
  // ss_dg_make_resident succeeded: g holds exactly `keys`, never exported while held.
  void hold(uint32_t g, std::vector<uint32_t> keys) {
    keep_[g] = std::move(keys);
    resident_[g] = 1;
  }
  bool resident(uint32_t g) const { return resident_[g] != 0; }
  const std::vector<uint32_t>& kept(uint32_t g) const { return keep_[g]; }
  // The boundary's keep list.
  std::vector<uint32_t> all_kept() const { return protect(); }
  // ss_dg_reset_open: every keep dropped (the sequence words are retained).
  void reset() {
    for (auto& k : keep_) k.clear();
    std::fill(resident_.begin(), resident_.end(), uint8_t(0));
  }

 private:
  std::vector<std::vector<uint32_t>> keep_;
  std::vector<uint8_t> resident_, run_;
  std::vector<uint32_t> flag_seq_;
};

}  // namespace ts
