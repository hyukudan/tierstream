// Host read-ahead (wave4/gapfix/wstream-readahead/design.md §7-11) on the CPU backend: byte identity against
// readahead_units = 0, both completion / landing orders, failures with and without a waiting landing, cancellation,
// shutdown without ss_end, deduplication, one reader, ABI and the R = 0 boundary.
#include <cstddef>
#include <cstring>
#include <string>
#include <vector>

#include "check.hpp"
#include "fixture.hpp"
#include "strix_stream/ss.h"

using sstest::Fixture;

// Test hooks (capi.cpp, not in ss.h).
extern "C" ss_status ss_test_hold_readers(ss_ctx* c, int hold);
extern "C" ss_status ss_test_inject(ss_ctx* c, int what);
extern "C" ss_status ss_test_ra_quiesce(ss_ctx* c);
extern "C" uint64_t ss_test_ra_started(ss_ctx* c);

namespace {

struct Run {
  ss_desc* d = nullptr;
  ss_plan* p = nullptr;
  ss_ctx* c = nullptr;
  ss_budget b;
  ss_options o;
  ~Run() {
    ss_close(c);
    ss_plan_free(p);
    ss_desc_free(d);
  }
};

const char* const kAllDisk[] = {"", "DDDDDDDDDDDDDDDD", "DDDDDDDD", "D", "DDDDDDDDDDDDDDDD"};

bool open_run(Run& r, const Fixture& fx, uint32_t ra, uint32_t ring = 16, uint32_t readers = 2,
              const char* const* tiers = kAllDisk) {
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return false;
  ss_budget_defaults(&r.b);
  r.b.margin_gib = 1.0 / 1024;
  r.b.margin_fraction = 0;
  r.b.device_bytes = uint64_t(256) << 20;
  r.b.ring_slots = ring;
  r.b.host_pinned_bytes = 4 << 20;
  std::string js = "{\"tiers\":[";
  for (int L = 0; L < 5; ++L) js += std::string(L ? ",\"" : "\"") + tiers[L] + "\"";
  js += "]}";
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js.data(), js.size(), &r.p));
  if (!r.p) return false;
  ss_options_defaults(&r.o);
  r.o.verify_hash = 1;
  r.o.staging_bytes = 64 * 1024;
  r.o.reader_threads = readers;
  r.o.readahead_units = ra;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  return r.c != nullptr;
}

bool same(const ss_layer_view& v, const Fixture& fx, int layer, int unit) {
  const auto& want = fx.expected[size_t(layer)][size_t(unit)];
  const void* a = v.table_host ? reinterpret_cast<const void*>(uintptr_t(v.table_host[unit])) : nullptr;
  if (!a) std::printf("  unit (%d, %d): not in the table\n", layer, unit);
  else if (std::memcmp(a, want.data(), want.size()) != 0) std::printf("  unit (%d, %d): bytes differ\n", layer, unit);
  return a && std::memcmp(a, want.data(), want.size()) == 0;
}

int units(int L) { return L == 2 ? Fixture::kE2 : L == 3 ? 1 : Fixture::kE1; }

// One whole-layer forward over layers 1-4 with host lookahead H; every unit's bytes checked.
void layer_major(Run& r, const Fixture& fx, int H) {
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  for (int L = 1; L <= H && L <= 4; ++L) CHECK_OK(ss_host_prefetch_layer(r.c, L));
  for (int L = 1; L <= 4; ++L) {
    for (int k = 1; k <= H && L + k <= 4; ++k) CHECK_OK(ss_host_prefetch_layer(r.c, L + k));  // overlapping windows
    CHECK_OK(ss_layer_all(r.c, L));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(r.c, L, nullptr, &v));
    for (int u = 0; u < units(L); ++u) CHECK(same(v, fx, L, u));
    CHECK_OK(ss_release(r.c, L, nullptr));
  }
  CHECK_OK(ss_end(r.c));
}

std::string stats(ss_ctx* c) {
  char* js = nullptr;
  CHECK_OK(ss_stats_to_json(c, &js, nullptr));
  std::string s = js ? js : "";
  ss_free(js);
  return s;
}

uint64_t stat_u64(const std::string& js, const char* key) {
  const std::string k = std::string("\"") + key + "\":";
  const size_t at = js.find("\"readahead\":");
  const size_t p = js.find(k, at == std::string::npos ? 0 : at);
  return p == std::string::npos ? ~uint64_t(0) : std::stoull(js.substr(p + k.size()));
}

uint64_t disk_bytes(ss_ctx* c) {
  ss_layer_stats t;
  CHECK_OK(ss_stats_total(c, &t));
  return t.bytes_disk;
}


}  // namespace

TEST(readahead_identity_and_accounting) {
  // Same bytes and the same file bytes with and without read-ahead; landings happen; JSON only with R > 0.
  Fixture f0("ra-r0"), f4("ra-r4");
  Run a, b;
  if (!open_run(a, f0, 0) || !open_run(b, f4, 4)) return;
  for (int pass = 0; pass < 3; ++pass) {
    layer_major(a, f0, 2);
    layer_major(b, f4, 2);
  }
  CHECK_EQ(disk_bytes(a.c), disk_bytes(b.c));
  const std::string s0 = stats(a.c), s4 = stats(b.c);
  CHECK(s0.find("readahead") == std::string::npos);  // R = 0: today's JSON
  CHECK(stat_u64(s4, "landed") > 0);
  CHECK_EQ(stat_u64(s4, "unused"), uint64_t(0));
  CHECK_EQ(stat_u64(s4, "failed"), uint64_t(0));
  CHECK_EQ(stat_u64(s4, "read"), stat_u64(s4, "landed"));  // every read lands (requests overtaken are never read)
  CHECK_EQ(stat_u64(s4, "buffer_bytes"), uint64_t(1052672));  // the blob: the largest unit that can be read
}

TEST(readahead_one_reader_and_one_buffer) {
  for (uint32_t readers : {1u, 2u}) {
    Fixture fx(readers == 1 ? "ra-1r" : "ra-2r");
    Run r;
    if (!open_run(r, fx, 1, 16, readers)) return;
    for (int pass = 0; pass < 2; ++pass) layer_major(r, fx, 4);
    const std::string s = stats(r.c);
    CHECK(stat_u64(s, "landed") > 0);
  }
}

TEST(readahead_landing_orders) {
  Fixture fx("ra-orders");
  Run r;
  if (!open_run(r, fx, 4, 16)) return;
  // (a) Landing while the read is still queued (readers held): the read is promoted, then lands.
  CHECK_OK(ss_test_hold_readers(r.c, 1));
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(r.c, 3));
  CHECK_OK(ss_layer_all(r.c, 3));
  CHECK_OK(ss_test_hold_readers(r.c, 0));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(r.c, 3, nullptr, &v));
  CHECK(same(v, fx, 3, 0));
  CHECK_OK(ss_release(r.c, 3, nullptr));
  CHECK_OK(ss_end(r.c));
  // (b) Landing after the read finished (Ready).
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(r.c, 3));
  CHECK_OK(ss_test_ra_quiesce(r.c));
  CHECK_OK(ss_layer_all(r.c, 3));
  CHECK_OK(ss_acquire(r.c, 3, nullptr, &v));
  CHECK(same(v, fx, 3, 0));
  CHECK_OK(ss_release(r.c, 3, nullptr));
  CHECK_OK(ss_end(r.c));
  CHECK_EQ(stat_u64(stats(r.c), "landed"), uint64_t(2));
}

TEST(readahead_failures) {
  // Attached failure: the read fails with a landing waiting on it -> the landing fails like a read, next forward OK.
  Fixture bad("ra-bad", /*corrupt_hash=*/true);
  Run r;
  if (!open_run(r, bad, 4, 16)) return;
  CHECK_OK(ss_test_hold_readers(r.c, 1));
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t three[1] = {3};
  CHECK_OK(ss_host_prefetch(r.c, 1, three, 1));
  CHECK_OK(ss_layer_routes(r.c, 1, three, 1));
  CHECK_OK(ss_test_hold_readers(r.c, 0));
  ss_layer_view v{};
  CHECK_EQ(int(ss_acquire(r.c, 1, nullptr, &v)), int(SS_E_HASH));
  CHECK(ss_end(r.c) != SS_OK);
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t two[1] = {2};
  CHECK_OK(ss_host_prefetch(r.c, 1, two, 1));
  CHECK_OK(ss_layer_routes(r.c, 1, two, 1));
  CHECK_OK(ss_acquire(r.c, 1, nullptr, &v));
  CHECK(same(v, bad, 1, 2));
  CHECK_OK(ss_release(r.c, 1, nullptr));
  CHECK_OK(ss_end(r.c));

  // Unattached failure (injected read error): the fetch falls back to the file; ss_end is OK.
  Fixture fx("ra-fall");
  Run q;
  if (!open_run(q, fx, 4, 16)) return;
  CHECK_OK(ss_begin(q.c, nullptr, 0));
  CHECK_OK(ss_test_inject(q.c, 5));
  CHECK_OK(ss_host_prefetch_layer(q.c, 3));
  CHECK_OK(ss_test_ra_quiesce(q.c));
  CHECK_OK(ss_layer_all(q.c, 3));
  CHECK_OK(ss_acquire(q.c, 3, nullptr, &v));
  CHECK(same(v, fx, 3, 0));
  CHECK_OK(ss_release(q.c, 3, nullptr));
  CHECK_OK(ss_end(q.c));
  std::string s = stats(q.c);
  CHECK_EQ(stat_u64(s, "failed"), uint64_t(1));
  CHECK_EQ(stat_u64(s, "fallback"), uint64_t(1));
  // Queueing a read fails (injected): the producer is terminal at once; the fetch reads the file; ss_end OK.
  CHECK_OK(ss_begin(q.c, nullptr, 0));
  CHECK_OK(ss_test_inject(q.c, 1));
  CHECK_OK(ss_host_prefetch_layer(q.c, 3));
  CHECK_OK(ss_layer_all(q.c, 3));
  CHECK_OK(ss_acquire(q.c, 3, nullptr, &v));
  CHECK(same(v, fx, 3, 0));
  CHECK_OK(ss_release(q.c, 3, nullptr));
  CHECK_OK(ss_end(q.c));
  // Landing copy fails before any copy: reported, the buffer comes back (4 buffers still serve 4 reads).
  CHECK_OK(ss_begin(q.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(q.c, 3));
  CHECK_OK(ss_test_ra_quiesce(q.c));
  CHECK_OK(ss_test_inject(q.c, 3));
  CHECK_OK(ss_layer_all(q.c, 3));
  CHECK_EQ(int(ss_acquire(q.c, 3, nullptr, &v)), int(SS_E_IO));
  CHECK(ss_end(q.c) != SS_OK);
  // Landing copy fails after its first piece: that buffer is quarantined; the others keep working.
  CHECK_OK(ss_begin(q.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(q.c, 3));
  CHECK_OK(ss_test_ra_quiesce(q.c));
  CHECK_OK(ss_test_inject(q.c, 4));
  CHECK_OK(ss_layer_all(q.c, 3));
  CHECK_EQ(int(ss_acquire(q.c, 3, nullptr, &v)), int(SS_E_IO));
  CHECK(ss_end(q.c) != SS_OK);
  for (int pass = 0; pass < 2; ++pass) layer_major(q, fx, 3);  // 3 buffers left
}

// Queueing a landing fails (injected), in both orders: the landing fails terminally with SS_E_NOMEM (it poisons the
// context, per the context-call rule), and the poisoned context closes with its buffers back (ASan: no leak).
TEST(readahead_landing_enqueue_failure_both_orders) {
  for (int order = 0; order < 2; ++order) {
    Fixture fx(order ? "ra-enq-reading" : "ra-enq-ready");
    Run r;
    if (!open_run(r, fx, 4, 16)) return;
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    if (order == 0) {  // the read is Ready when the slot is given
      CHECK_OK(ss_host_prefetch_layer(r.c, 3));
      CHECK_OK(ss_test_ra_quiesce(r.c));
      CHECK_OK(ss_test_inject(r.c, 2));
      CHECK_OK(ss_layer_all(r.c, 3));
    } else {  // the read is still queued: the landing is parked, the read's epilogue queues it
      CHECK_OK(ss_test_hold_readers(r.c, 1));
      CHECK_OK(ss_host_prefetch_layer(r.c, 3));
      CHECK_OK(ss_layer_all(r.c, 3));
      CHECK_OK(ss_test_inject(r.c, 2));
      CHECK_OK(ss_test_hold_readers(r.c, 0));
    }
    ss_layer_view v{};
    CHECK_EQ(int(ss_acquire(r.c, 3, nullptr, &v)), int(SS_E_NOMEM));
    CHECK_EQ(int(ss_end(r.c)), int(SS_E_BACKEND));  // poisoned: only close works
  }
}

// The MiniMax prefill pattern: a full ring (layer 1 holds all 16 slots of its size) and the next layer queued by a
// certain whole-layer prefetch (k = 1) or ss_layer_all: its units wait for slots, and their reads start at once.
TEST(readahead_k1_full_ring_starts_reads) {
  for (int how = 0; how < 2; ++how) {
    Fixture fx(how ? "ra-k1-all" : "ra-k1-pf");
    Run r;
    if (!open_run(r, fx, 4, 16)) return;
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    CHECK_OK(ss_layer_all(r.c, 1));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(r.c, 1, nullptr, &v));
    const uint64_t before = ss_test_ra_started(r.c);
    if (how == 0) CHECK_OK(ss_prefetch_layer(r.c, 4));
    else CHECK_OK(ss_layer_all(r.c, 4));
    CHECK_EQ(ss_test_ra_started(r.c) - before, uint64_t(4));  // one read per buffer, before layer 1 releases
    CHECK_OK(ss_release(r.c, 1, nullptr));
    if (how == 0) CHECK_OK(ss_layer_all(r.c, 4));
    CHECK_OK(ss_acquire(r.c, 4, nullptr, &v));
    for (int u = 0; u < Fixture::kE1; ++u) CHECK(same(v, fx, 4, u));
    CHECK_OK(ss_release(r.c, 4, nullptr));
    CHECK_OK(ss_end(r.c));
    CHECK(stat_u64(stats(r.c), "landed") >= 4);
  }
}

// File bytes: a read whose hash fails still read its bytes (counted once), and its ordinary fallback counts its own.
TEST(readahead_byte_accounting) {
  Fixture bad("ra-bytes", /*corrupt_hash=*/true);
  Run r;
  if (!open_run(r, bad, 4, 16)) return;
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t three[1] = {3};
  CHECK_OK(ss_host_prefetch(r.c, 1, three, 1));
  CHECK_OK(ss_test_ra_quiesce(r.c));  // the read failed its hash
  CHECK_OK(ss_layer_routes(r.c, 1, three, 1));  // fallback: an ordinary read, which fails its hash too
  ss_layer_view v{};
  CHECK_EQ(int(ss_acquire(r.c, 1, nullptr, &v)), int(SS_E_HASH));
  CHECK(ss_end(r.c) != SS_OK);
  CHECK_EQ(disk_bytes(r.c), uint64_t(2 * 11000));
  const std::string s = stats(r.c);
  CHECK_EQ(stat_u64(s, "bytes"), uint64_t(11000));
  CHECK_EQ(stat_u64(s, "failed"), uint64_t(1));
  CHECK_EQ(stat_u64(s, "fallback"), uint64_t(1));
}

TEST(readahead_cancellation) {
  Fixture fx("ra-cancel");
  Run r;
  if (!open_run(r, fx, 4, 16)) return;
  ss_layer_view v{};
  // A landing parked on a held read, released without waiting its group: retire_ring cancels it while it is parked.
  CHECK_OK(ss_test_hold_readers(r.c, 1));
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(r.c, 3));
  CHECK_OK(ss_layer_all(r.c, 3));
  CHECK_OK(ss_acquire_groups(r.c, 3, nullptr, &v));
  CHECK_OK(ss_release(r.c, 3, nullptr));
  CHECK_OK(ss_test_hold_readers(r.c, 0));
  CHECK_OK(ss_end(r.c));
  // A landing already queued (read Ready, readers held) and cancelled before it ran.
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(r.c, 3));
  CHECK_OK(ss_test_ra_quiesce(r.c));
  CHECK_OK(ss_test_hold_readers(r.c, 1));
  CHECK_OK(ss_layer_all(r.c, 3));
  CHECK_OK(ss_acquire_groups(r.c, 3, nullptr, &v));
  CHECK_OK(ss_release(r.c, 3, nullptr));
  CHECK_OK(ss_test_hold_readers(r.c, 0));
  CHECK_OK(ss_end(r.c));
  // Unused reads at ss_end: counted, buffers back; the context keeps working with all 4.
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(r.c, 1));
  CHECK_OK(ss_host_prefetch_layer(r.c, 4));
  CHECK_OK(ss_test_ra_quiesce(r.c));
  CHECK_OK(ss_end(r.c));
  CHECK(stat_u64(stats(r.c), "unused") > 0);
  for (int pass = 0; pass < 2; ++pass) layer_major(r, fx, 2);
}

TEST(readahead_dedup) {
  // Overlapping windows with one buffer: each unit is requested once per forward, nothing is stuck.
  Fixture fx("ra-dedup");
  Run r;
  if (!open_run(r, fx, 1, 16)) return;
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  for (int k = 0; k < 3; ++k) CHECK_OK(ss_host_prefetch_layer(r.c, 2));
  CHECK_OK(ss_layer_all(r.c, 2));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
  for (int u = 0; u < Fixture::kE2; ++u) CHECK(same(v, fx, 2, u));
  CHECK_OK(ss_release(r.c, 2, nullptr));
  CHECK_OK(ss_end(r.c));
  CHECK_EQ(stat_u64(stats(r.c), "requested"), uint64_t(Fixture::kE2));
}

TEST(readahead_close_without_end) {
  // Between forwards (no begin): reads Ready and requests Waiting, then close. And in a forward, a landing parked
  // on a held read, then close without ss_end. ASan / TSan builds check for leaks and races.
  {
    Fixture fx("ra-close1");
    Run r;
    if (!open_run(r, fx, 1, 16)) return;
    CHECK_OK(ss_host_prefetch_layer(r.c, 1));
    CHECK_OK(ss_host_prefetch_layer(r.c, 3));
    CHECK_OK(ss_test_ra_quiesce(r.c));
  }
  {
    Fixture fx("ra-close2");
    Run r;
    if (!open_run(r, fx, 2, 16)) return;
    CHECK_OK(ss_test_hold_readers(r.c, 1));
    CHECK_OK(ss_begin(r.c, nullptr, 0));
    CHECK_OK(ss_host_prefetch_layer(r.c, 3));
    CHECK_OK(ss_layer_all(r.c, 3));
    CHECK_OK(ss_test_hold_readers(r.c, 0));
  }
}

TEST(readahead_exclusive_persistent_admission) {
  // Exclusive decode: a unit with a read-ahead request admitted into a persistent slot abandons the request (read-ahead
  // lands only in ring slots) and takes the ordinary path, with its victim's write-back.
  Fixture fx("ra-excl");
  Run r;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return;
  ss_budget_defaults(&r.b);
  r.b.margin_gib = 1.0 / 1024;
  r.b.margin_fraction = 0;
  r.b.device_bytes = uint64_t(256) << 20;
  r.b.ring_slots = 16;
  r.b.host_pinned_bytes = 4 << 20;
  r.b.tier_mode = SS_TIER_MODE_EXCLUSIVE;
  const char* js = "{\"tiers\":[\"\",\"PPPPPPPPPPPPPPPP\",\"RRRRDDDD\",\"P\",\"PPPPPPPPPPPPPPPP\"]}";
  CHECK_OK(ss_plan_from_json(r.d, &r.b, js, std::strlen(js), &r.p));
  if (!r.p) return;
  ss_options_defaults(&r.o);
  r.o.verify_hash = 1;
  r.o.pin_planned = 0;
  r.o.staging_bytes = 64 * 1024;
  r.o.readahead_units = 2;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  if (!r.c) return;
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t ids[2] = {5, 6};
  CHECK_OK(ss_host_prefetch(r.c, 2, ids, 2));
  CHECK_OK(ss_test_ra_quiesce(r.c));
  CHECK_OK(ss_layer_routes(r.c, 2, ids, 2));  // both admitted (evicting residents 0, 1): their requests are abandoned
  ss_layer_view v{};
  CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
  CHECK(same(v, fx, 2, 5) && same(v, fx, 2, 6));
  CHECK_OK(ss_check_unit(r.c, 2, 5));
  CHECK_OK(ss_release(r.c, 2, nullptr));
  CHECK_OK(ss_end(r.c));
  const std::string s = stats(r.c);
  CHECK_EQ(stat_u64(s, "landed"), uint64_t(0));
  CHECK_EQ(stat_u64(s, "unused"), uint64_t(2));
  ss_exclusive_stats x{};
  CHECK_OK(ss_stats_exclusive(r.c, &x));
  CHECK_EQ(x.writebacks, uint64_t(2));
  // The written-back residents come back from their blocks.
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  int32_t back[2] = {0, 1};
  CHECK_OK(ss_layer_routes(r.c, 2, back, 2));
  CHECK_OK(ss_acquire(r.c, 2, nullptr, &v));
  CHECK(same(v, fx, 2, 0) && same(v, fx, 2, 1));
  CHECK_OK(ss_release(r.c, 2, nullptr));
  CHECK_OK(ss_end(r.c));
}

TEST(readahead_off_and_abi) {
  // R = 0: host prefetch is a no-op.
  Fixture fx("ra-off");
  Run r;
  if (!open_run(r, fx, 0, 16)) return;
  CHECK_OK(ss_host_prefetch_layer(r.c, 1));
  layer_major(r, fx, 2);
  CHECK(stats(r.c).find("readahead") == std::string::npos);
  // Over the limit.
  Fixture f2("ra-limit");
  Run q;
  CHECK_OK(ss_desc_from_json(f2.json.data(), f2.json.size(), &q.d));
  if (!q.d) return;
  ss_budget_defaults(&q.b);
  q.b.margin_gib = 1.0 / 1024;
  q.b.device_bytes = uint64_t(256) << 20;
  ss_options_defaults(&q.o);
  q.o.readahead_units = 65;
  CHECK_EQ(int(ss_open(q.d, &q.b, nullptr, &q.o, &q.c)), int(SS_E_INVALID));
  // An old caller's ss_options (struct_size 80, before reserved0) with garbage in its padding: read-ahead stays off.
  static_assert(sizeof(void*) != 8 || offsetof(ss_options, reserved0) == 76, "old struct end");
  static_assert(sizeof(void*) != 8 || offsetof(ss_options, readahead_units) == 80, "first new field");
  unsigned char raw[sizeof(ss_options)];
  std::memset(raw, 0xA5, sizeof raw);
  ss_options_defaults_sized(reinterpret_cast<ss_options*>(raw), uint32_t(offsetof(ss_options, reserved0) + 4));
  ss_options* old = reinterpret_cast<ss_options*>(raw);
  old->reserved0 = 3;
  CHECK_OK(ss_open(q.d, &q.b, nullptr, old, &q.c));
  if (q.c) {
    CHECK_OK(ss_host_prefetch_layer(q.c, 1));  // a no-op: R = 0
    CHECK(stats(q.c).find("readahead") == std::string::npos);
  }
}

int main() { return sstest::run_all(); }
