// Adaptive memory plan, part 1: the expert usage table (tierstream/usage-v1).
//
// Per (layer, unit): how often the router picked it (count of selections and micro-selections per token) and a
// decayed recency (each selection adds 10^6, decaying by half every `half_life` tokens). Integers only, so a table
// built on any machine from the same routes is byte-identical (canonical JSON, sha256 of those bytes).
//
// Ranking (contract placement-v1 rev 4 §3.1): use descending, then recency descending, then (layer, unit) ascending.
// The planner and the tier solver give the top of this order the VRAM arena, the next the pinned host tier and the
// tail the NVMe (FILE) tier.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "json.hpp"

namespace ts {
namespace adapt {

constexpr uint64_t kMicro = 1000000;

// 64 x 64 -> 128 bit helpers (portable; MSVC has no unsigned __int128).
uint64_t mul_q32(uint64_t a, uint64_t q32);       // round(a * q32 / 2^32), q32 <= 2^32
uint64_t pow_q32(uint64_t q32, uint64_t e);       // q32^e in Q32, rounded at each step (deterministic)
uint64_t decay_q32(uint32_t half_life_tokens);    // round(2^32 * 2^(-1 / half_life)), 0 = no decay (2^32)
uint64_t rdiv(uint64_t n, uint64_t d);            // round half up; d > 0

struct Source {
  std::string kind;    // "routes-trace" | "online" | "merge"
  std::string path;    // as given (informational)
  std::string sha256;  // of the source bytes ("" when online)
  std::string filter;  // sequence filter used ("" = all)
  uint64_t tokens = 0;
};

class Usage {
 public:
  Usage() = default;
  Usage(std::string model, std::vector<uint32_t> units_per_layer, uint32_t half_life_tokens);

  // One routed selection set of one layer for the current token (ids < 0 ignored; out of range throws).
  void observe(int layer, const int32_t* ids, int n);
  // Token boundary: `rows` tokens were decoded by the observations since the last tick (1 for a plain decode step).
  void tick(uint32_t rows = 1);

  int n_layers() const { return int(units_.size()); }
  uint32_t layer_units(int layer) const { return units_[size_t(layer)]; }
  int64_t total_units() const { return int64_t(count_.size()); }
  int64_t base(int layer) const { return base_[size_t(layer)]; }
  uint64_t tokens() const { return tokens_; }
  uint64_t tick_now() const { return tick_; }
  uint32_t half_life() const { return half_life_; }
  const std::string& model() const { return model_; }
  std::vector<Source>& sources() { return sources_; }
  const std::vector<Source>& sources() const { return sources_; }

  uint64_t count(int64_t g) const { return count_[size_t(g)]; }
  // Micro-selections per token: rdiv(count * 10^6, tokens) (0 before the first tick).
  uint64_t use(int64_t g) const;
  // Decayed recency at the current tick, in micro-selections.
  uint64_t recency(int64_t g) const;

  // Global unit order, hottest first (the ranking above). Deterministic.
  std::vector<int64_t> rank() const;

  // Adds another table of the same shape (counts and tokens add; recency is aligned to the later tick).
  void merge(const Usage& o);

  Json to_json() const;  // canonical
  std::string to_text() const;
  std::string sha256() const;
  static Usage from_json(const Json& j);
  static Usage from_text(const std::string& text);

 private:
  void touch(int64_t g);
  std::string model_;
  std::vector<uint32_t> units_;
  std::vector<int64_t> base_;
  uint32_t half_life_ = 0;
  uint64_t q32_ = uint64_t(1) << 32;
  uint64_t tokens_ = 0, tick_ = 0;
  std::vector<uint64_t> count_;
  std::vector<uint64_t> rec_;   // decayed recency as of last_[g]
  std::vector<uint64_t> last_;  // tick of the last update of rec_[g]
  std::vector<Source> sources_;
};

// Tiers from a ranking: the first `active` units RESIDENT, the next `host` PINNED, the rest DISK (FILE homes).
// order: global indices hottest first (Usage::rank); n_total units. Throws when active + host > n_total.
std::vector<uint8_t> tiers_from_rank(const std::vector<int64_t>& order, int64_t n_total, int64_t active, int64_t host);

}  // namespace adapt
}  // namespace ts
