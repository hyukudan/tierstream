// Stripe manifest tierstream/stripe-v1 (docs/design.md §4.4): which chunks of a primary file a secondary location holds,
// where, and their xxh64. Parsing validates every rule before allocating anything large.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ts {

struct StripeRun {
  uint64_t first_chunk = 0, count = 0, first_slot = 0;
};

struct StripeManifest {
  uint64_t chunk_bytes = 0;
  std::string primary_basename;
  uint64_t primary_size = 0;
  std::string primary_sha256;  // 64 lowercase hex digits
  std::string data_name;
  uint64_t data_size = 0;
  std::vector<StripeRun> runs;
  std::vector<uint64_t> hashes;  // xxh64 of each placed chunk's logical bytes, in slot order
  uint64_t slots() const { return hashes.size(); }
};

// Canonical text with its digest field; parse validates every rule (throws SS_E_DESC with the reason).
std::string stripe_manifest_write(const StripeManifest& m);
StripeManifest stripe_manifest_parse(const std::string& text);
std::string stripe_manifest_name(const std::string& primary_basename);  // "<base>.stripe.json"
void stripe_data_name_validate(const std::string& primary_basename, const std::string& data_name);

// Limits checked before allocation.
constexpr uint64_t kStripeMaxChunks = uint64_t(1) << 24;
constexpr uint64_t kStripeMaxRuns = uint64_t(1) << 20;
constexpr uint64_t kStripeMaxPrimary = uint64_t(1) << 50;

}  // namespace ts
