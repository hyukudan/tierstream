// Multi-location weight file (docs/design.md §4.4): one primary file plus stripe copies of some of its chunks on other
// drives, read in parallel with one reader pool per drive. Every byte a secondary location serves belongs to a chunk
// whose xxh64 matched the stripe manifest; any failure falls back to the primary for that chunk.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "raw_file.hpp"
#include "stripe_manifest.hpp"
#include "strix_stream/ss.h"

namespace ts {

class MFile {
 public:
  struct Options {
    bool direct = true;
    uint32_t drive_readers = 4;
    int verify_mode = SS_MFILE_VERIFY_CACHED;
    std::string primary_sha256;
  };
  static constexpr uint32_t kAlign = 4096;

  // Lookup by (primary, secondary dirs, direct); reuse checks identities, manifests, reader count and verification.
  // Pools are shared by (volume, reader count). No secondary dir after canonicalisation means multi() == false.
  static std::shared_ptr<MFile> open(const std::string& primary, const std::string& weight_dirs, const Options& o);
  // Secondary dirs of a weight_dirs list after canonicalisation (primary's dir and duplicates dropped).
  static std::vector<std::string> secondary_dirs(const std::string& primary, const std::string& weight_dirs);
  ~MFile();

  uint64_t size() const;
  bool multi() const;
  bool direct() const;
  // Aligned contract: offset, n and dst multiples of kAlign; a read may extend past EOF to the aligned end.
  uint64_t pread(void* dst, uint64_t n, uint64_t offset);
  void preadv(const ss_mfile_io* ios, uint32_t n);
  void counters(ss_mfile_counters* out) const;
  std::string location_path(int loc) const;
  int locations() const;
  std::string reasons() const;  // why secondary locations were disabled at open (empty when none)

  // test hook: make reads of location loc fail (1 = error, 2 = short read, 3 = corrupt bytes) after `after` reads
  void test_inject(int loc, int kind, uint64_t after);
  // test hook: every request may use the stripe (not only large requests and long batches)
  void test_prefer_stripe(bool on);
  // test hook: every piece served by location loc takes `microseconds` longer (a slow drive)
  void test_delay(int loc, uint64_t microseconds);
  // Test observation: total sessions constructed, including sessions already closed.
  static uint64_t test_sessions_created();
  static uint64_t test_pools_created();
  double test_rate(int loc) const;

  struct Impl;

 private:
  explicit MFile(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace ts
