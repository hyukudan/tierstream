// Adaptive memory plan, part 2: recorded routing traces and the tier simulator that turns them into miss curves
// (tierstream/miss-curve-v1).
//
// Trace format (the replica's route dump, e.g. hetero/w1/data/w1-routes-r0.txt):
//   # seq <name> rows <n>
//   0:e,e,e,e,e,e 1:e,... 39:e,...      one line per token, "layer:ids" for every routed layer
//
// The simulator models the device-gated cache (one global arena, devgate / the DS replica's StreamCache): `active`
// persistent slots warm-started with the top of the usage ranking plus `spares` free slots; a demand miss takes a free
// slot (else it is served from the demand bank and not admitted); at each window boundary the least recently used
// units are evicted until free + in-flight write-backs reach `spares`. A unit's home is pinned RAM or, for the `file`
// set (the coldest units of the ranking), its file: a FILE victim is discarded (its slot is free at once), a pinned
// victim is written back and its slot is free `wb_windows` windows later. Every miss is counted by the home it is
// fetched from.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "adapt_usage.hpp"
#include "json.hpp"

namespace ts {
namespace adapt {

struct Trace {
  struct Seq {
    std::string name;
    uint32_t first = 0, tokens = 0;  // token range in `ids`
  };
  int n_layers = 0, top_k = 0, n_units = 0;  // n_units: per layer (uniform); max id + 1 unless given
  std::vector<Seq> seqs;
  std::vector<int32_t> ids;  // token-major: [token][layer][top_k], -1 = missing
  std::string sha256;
  std::string path;

  uint64_t n_tokens() const { return ids.size() / (size_t(n_layers) * size_t(top_k)); }
  const int32_t* at(uint64_t token, int layer) const {
    return ids.data() + (token * uint64_t(n_layers) + uint64_t(layer)) * uint64_t(top_k);
  }
};

Trace trace_parse(const std::string& text, int n_units_hint = 0);
Trace trace_load(const std::string& path, int n_units_hint = 0);
// Sequence filter: "" = all; else '&'-separated groups that must all match; a group is comma-separated substrings and
// matches when the name contains any of them (e.g. "train-", "val-&:code:", ":en:,:es:"); a leading '!' negates a group.
bool seq_match(const std::string& name, const std::string& filter);

// Usage table from the matching sequences of a trace (one tick per token).
Usage usage_from_trace(const Trace& t, const std::string& filter, const std::string& model, uint32_t half_life);

enum class Evict { Lru = 0, FileLast = 1 };

struct SimConfig {
  int64_t active = 0;          // persistent slots warm-started from the ranking
  int64_t spares = 96;         // free-slot target at each boundary
  int64_t file_units = 0;      // coldest units of the ranking whose home is their file
  uint32_t rows = 1;           // tokens per window (R): misses are counted on the union of the window's routes
  uint32_t wb_windows = 1;     // write-back latency in windows before a pinned victim's slot is free
  Evict evict = Evict::Lru;
  uint32_t file_last_window = 64;  // FileLast: victims are taken from the `window` least recent, pinned homes first
};

struct SimResult {
  uint64_t windows = 0, tokens = 0;
  uint64_t lookups = 0;                     // distinct (layer, unit) per window
  uint64_t miss_pinned = 0, miss_file = 0;  // demand misses by home
  uint64_t bank = 0;                        // misses served without admission (no free slot)
  uint64_t evicted = 0, discarded = 0;      // write-backs / FILE discards
  uint64_t file_uses = 0;                   // lookups of file-home units (resident or not)
  uint64_t milli_per_window(uint64_t v) const { return windows ? rdiv(v * 1000, windows) : 0; }
  void add(const SimResult& r);
};

SimResult simulate(const Trace& t, const std::string& filter, const std::vector<int64_t>& order, const SimConfig& c);

// k-fold cross-validation (as wstream's g0impl.py): sequence i is in fold i % folds; each fold is simulated with its
// own arena, ranked by a usage table of the other folds; results summed.
SimResult simulate_xval(const Trace& t, const std::string& filter, const SimConfig& c, int folds, uint32_t half_life);

// miss-curve-v1 for a list of arena sizes and a fixed file set, one curve per filter label ("code" -> ":code:"):
// curves = {"<label>/R<rows>": [[active_units, milli-misses per window], ...]} (contract shape), detail = the split
// by home. folds >= 2: cross-validated (u gives only the half life); else ranked by u.
Json miss_curve(const Trace& t, const std::vector<std::pair<std::string, std::string>>& labels,
                const Usage& u, const std::vector<int64_t>& actives, const SimConfig& base, int folds = 0);

}  // namespace adapt
}  // namespace ts
