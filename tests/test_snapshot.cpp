// Stats snapshot (design.md §7.1), CPU backend: arming, every publication equal to the live stats at its boundary
// (struct and stats-v1 bytes), what does and does not publish (resets, failures, poison, between-forward work), the
// sized struct ABI, and readers on other threads while forwards run.
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "check.hpp"
#include "fixture.hpp"
#include "strix_stream/ss.h"

using sstest::Fixture;

// Test hooks (capi.cpp, not in ss.h).
extern "C" ss_status ss_test_ra_quiesce(ss_ctx* c);
extern "C" ss_status ss_test_inject(ss_ctx* c, int what);

namespace {

static_assert(offsetof(ss_snapshot_info, flags) == 4, "flags");
static_assert(offsetof(ss_snapshot_info, sequence) == 8, "sequence");
static_assert(offsetof(ss_snapshot_info, forwards) == 16, "forwards");
static_assert(offsetof(ss_snapshot_info, readahead_buffers) == 24, "first read-ahead field");
static_assert(offsetof(ss_snapshot_info, exclusive_writebacks) == 96, "first exclusive field");
static_assert(sizeof(ss_snapshot_info) == 160, "v1 size");

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

struct Cfg {
  const char* tiers = R"({"tiers":["","RRRRPPPPDDDDDDDD","PPPPDDDD","D","DDDDDDDDDDDDDDDD"]})";
  int mode = SS_TIER_MODE_INCLUSIVE;
  uint32_t readahead = 0;
  uint32_t dynamic = 2;
  ss_policy policy = SS_POLICY_SCORED;
};

const char* const kDisk = R"({"tiers":["","DDDDDDDDDDDDDDDD","DDDDDDDD","D","DDDDDDDDDDDDDDDD"]})";
const char* const kExcl = R"({"tiers":["","RRRRPPPPPPPPPPPP","RRPPPPPP","P","PPPPPPPPPPPPPPPP"]})";

bool open_run(Run& r, const Fixture& fx, const Cfg& cfg) {
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &r.d));
  if (!r.d) return false;
  ss_budget_defaults(&r.b);
  r.b.margin_gib = 1.0 / 1024;
  r.b.margin_fraction = 0;
  r.b.device_bytes = uint64_t(256) << 20;
  r.b.ring_slots = 32;
  r.b.host_pinned_bytes = 4 << 20;
  r.b.dynamic_slots_per_layer = cfg.dynamic;
  r.b.tier_mode = cfg.mode;
  r.b.host_spare_units = cfg.mode == SS_TIER_MODE_EXCLUSIVE ? 4 : 0;
  CHECK_OK(ss_plan_from_json(r.d, &r.b, cfg.tiers, std::strlen(cfg.tiers), &r.p));
  if (!r.p) return false;
  ss_options_defaults(&r.o);
  r.o.verify_hash = 1;
  r.o.staging_bytes = 64 * 1024;
  r.o.policy = cfg.policy;
  r.o.pin_planned = 0;
  r.o.readahead_units = cfg.readahead;
  CHECK_OK(ss_open(r.d, &r.b, r.p, &r.o, &r.c));
  return r.c != nullptr;
}

std::vector<int32_t> random_route(std::mt19937& g, int n_units, int top_k, int rows) {
  std::vector<int32_t> ids;
  std::uniform_int_distribution<int> hot(0, 3), any(0, n_units - 1), coin(0, 2);
  for (int r = 0; r < rows; ++r) {
    std::set<int> row;
    while (int(row.size()) < top_k) row.insert(coin(g) ? hot(g) : any(g));
    ids.insert(ids.end(), row.begin(), row.end());
  }
  return ids;
}

// The routed part of one decode forward (layers 1, 2, 4 and the blob layer 3); `inside` runs after layer 1.
template <class F>
void forward_body(ss_ctx* c, std::mt19937& g, int step, bool ra, F&& inside) {
  if (ra) CHECK_OK(ss_host_prefetch_layer(c, 3));
  for (int L : {1, 2, 4}) {
    const int E = L == 2 ? Fixture::kE2 : Fixture::kE1, k = L == 2 ? 2 : 4;
    auto ids = random_route(g, E, k, 1 + step % 3);
    if (L == 1) {
      int32_t pred[2] = {int32_t(step % 8), int32_t((step * 5) % 8)};
      CHECK_OK(ss_prefetch(c, 2, pred, 2, SS_PREFETCH_SPECULATIVE));
    }
    CHECK_OK(ss_layer_routes(c, L, ids.data(), int(ids.size())));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(c, L, nullptr, &v));
    CHECK_OK(ss_release(c, L, nullptr));
    if (L == 1) inside();
  }
  CHECK_OK(ss_layer_all(c, 3));
  ss_layer_view v{};
  CHECK_OK(ss_acquire(c, 3, nullptr, &v));
  CHECK_OK(ss_release(c, 3, nullptr));
}

void forward(ss_ctx* c, std::mt19937& g, int step, bool ra) {
  CHECK_OK(ss_begin(c, nullptr, 0));
  forward_body(c, g, step, ra, [] {});
  CHECK_OK(ss_end(c));
}

std::string live_json(ss_ctx* c) {
  char* js = nullptr;
  CHECK_OK(ss_stats_to_json(c, &js, nullptr));
  std::string s = js ? js : "";
  ss_free(js);
  return s;
}

// ss_stats_snapshot_json: the status, and the document when SS_OK.
ss_status snap_json(const ss_ctx* c, std::string* out) {
  char* js = nullptr;
  size_t n = 0;
  ss_status st = ss_stats_snapshot_json(c, &js, &n);
  if (st == SS_OK) {
    *out = std::string(js, n);
    ss_free(js);
  }
  return st;
}

// A number inside the JSON object `section` ("" = the whole document); ~0 when missing.
uint64_t json_u64(const std::string& js, const char* section, const char* key) {
  size_t at = 0;
  if (*section) {
    at = js.find(std::string("\"") + section + "\":{");
    if (at == std::string::npos) return ~uint64_t(0);
  }
  const std::string k = std::string("\"") + key + "\":";
  const size_t p = js.find(k, at);
  return p == std::string::npos ? ~uint64_t(0) : std::stoull(js.substr(p + k.size()));
}

bool same_stats(const ss_layer_stats& a, const ss_layer_stats& b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// What a publication made now must hold, from the live calls on the engine thread.
struct Expect {
  ss_layer_stats total{};
  ss_snapshot_info info{};
  std::string json;
};

Expect expect_now(ss_ctx* c, uint64_t seq) {
  Expect e;
  CHECK_OK(ss_stats_total(c, &e.total));
  e.json = live_json(c);
  ss_snapshot_info& i = e.info;
  i.struct_size = uint32_t(sizeof i);
  i.sequence = seq;
  i.forwards = json_u64(e.json, "", "forwards");
  if (e.json.find("\"readahead\":{") != std::string::npos) {
    i.flags |= SS_SNAPSHOT_READAHEAD;
    i.readahead_buffers = json_u64(e.json, "readahead", "buffers");
    i.readahead_buffer_bytes = json_u64(e.json, "readahead", "buffer_bytes");
    i.readahead_requested = json_u64(e.json, "readahead", "requested");
    i.readahead_read = json_u64(e.json, "readahead", "read");
    i.readahead_landed = json_u64(e.json, "readahead", "landed");
    i.readahead_unused = json_u64(e.json, "readahead", "unused");
    i.readahead_failed = json_u64(e.json, "readahead", "failed");
    i.readahead_fallback = json_u64(e.json, "readahead", "fallback");
    i.readahead_bytes = json_u64(e.json, "readahead", "bytes");
  }
  if (e.json.find("\"exclusive\":{") != std::string::npos) {
    ss_exclusive_stats x{};
    CHECK_OK(ss_stats_exclusive(c, &x));
    i.flags |= SS_SNAPSHOT_EXCLUSIVE;
    i.exclusive_writebacks = x.writebacks;
    i.exclusive_writeback_bytes = x.writeback_bytes;
    i.exclusive_deferred = x.deferred;
    i.exclusive_promoted = x.promoted;
    i.exclusive_free_blocks = x.free_blocks;
    i.exclusive_min_free_blocks = x.min_free_blocks;
    i.exclusive_rereads = x.rereads;
    i.exclusive_blocks = json_u64(e.json, "exclusive", "blocks");
  }
  return e;
}

bool same_info(const ss_snapshot_info& a, const ss_snapshot_info& b) { return std::memcmp(&a, &b, sizeof a) == 0; }

// Both calls return exactly the expected publication.
void check_published(ss_ctx* c, const Expect& e) {
  ss_layer_stats t;
  ss_snapshot_info i;
  std::memset(&i, 0xCD, sizeof i);
  CHECK_OK(ss_stats_snapshot(c, &t, &i));
  CHECK(same_stats(t, e.total));
  CHECK(same_info(i, e.info));
  if (!same_info(i, e.info))
    std::printf("  seq %llu vs %llu, forwards %llu vs %llu, flags %u vs %u\n", (unsigned long long)i.sequence,
                (unsigned long long)e.info.sequence, (unsigned long long)i.forwards,
                (unsigned long long)e.info.forwards, i.flags, e.info.flags);
  std::string js;
  CHECK_OK(snap_json(c, &js));
  CHECK_EQ(js, e.json);
}

void check_unpublished(ss_ctx* c) {
  ss_layer_stats t;
  ss_snapshot_info i;
  std::memset(&t, 0xAB, sizeof t);
  std::memset(&i, 0xAB, sizeof i);
  ss_layer_stats t0 = t;
  ss_snapshot_info i0 = i;
  CHECK_EQ(int(ss_stats_snapshot(c, &t, &i)), int(SS_E_INVALID));
  CHECK(std::strstr(ss_last_error(), "no stats snapshot published yet") != nullptr);
  CHECK(same_stats(t, t0));
  CHECK(same_info(i, i0));
  char* js = reinterpret_cast<char*>(uintptr_t(1));
  size_t n = 7;
  CHECK_EQ(int(ss_stats_snapshot_json(c, &js, &n)), int(SS_E_INVALID));
  CHECK(js == reinterpret_cast<char*>(uintptr_t(1)));
  CHECK_EQ(n, size_t(7));
}

}  // namespace

// The first call arms and finds nothing; the next boundary publishes; a reset while unarmed publishes nothing.
TEST(snapshot_arming) {
  Fixture fx("snap-arm");
  Run r;
  if (!open_run(r, fx, Cfg{})) return;
  std::mt19937 g(1);
  forward(r.c, g, 0, false);
  ss_stats_reset(r.c);  // unarmed: no publication
  forward(r.c, g, 1, false);
  check_unpublished(r.c);  // arms
  check_unpublished(r.c);
  forward(r.c, g, 2, false);
  Expect e = expect_now(r.c, 1);
  CHECK_EQ(e.info.forwards, uint64_t(3));
  check_published(r.c, e);

  // Armed by the JSON call; the first publication is a reset between forwards.
  Fixture fy("snap-arm2");
  Run q;
  if (!open_run(q, fy, Cfg{})) return;
  forward(q.c, g, 0, false);
  std::string js;
  CHECK_EQ(int(snap_json(q.c, &js)), int(SS_E_INVALID));
  ss_stats_reset(q.c);
  check_published(q.c, expect_now(q.c, 1));
}

// Every publication equals the live stats at its boundary: inclusive with misses and speculation, read-ahead,
// exclusive; a snapshot inside a forward still shows the previous boundary.
TEST(snapshot_matches_live) {
  struct Case {
    const char* tag;
    Cfg cfg;
  };
  Cfg ra;
  ra.tiers = kDisk;
  ra.readahead = 2;
  ra.dynamic = 0;
  ra.policy = SS_POLICY_LRU;
  Cfg ex;
  ex.tiers = kExcl;
  ex.mode = SS_TIER_MODE_EXCLUSIVE;
  ex.policy = SS_POLICY_LRU;
  for (const Case& k : {Case{"snap-incl", Cfg{}}, Case{"snap-ra", ra}, Case{"snap-excl", ex}}) {
    Fixture fx(k.tag);
    Run r;
    if (!open_run(r, fx, k.cfg)) return;
    check_unpublished(r.c);
    std::mt19937 g(7);
    Expect prev;
    for (int step = 0; step < 24; ++step) {
      CHECK_OK(ss_begin(r.c, nullptr, 0));
      forward_body(r.c, g, step, k.cfg.readahead > 0, [&] {
        if (step == 0) check_unpublished(r.c);
        else check_published(r.c, prev);
      });
      CHECK_OK(ss_end(r.c));
      prev = expect_now(r.c, uint64_t(step + 1));
      check_published(r.c, prev);
    }
    const bool want_ra = k.cfg.readahead > 0, want_x = k.cfg.mode == SS_TIER_MODE_EXCLUSIVE;
    CHECK_EQ((prev.info.flags & SS_SNAPSHOT_READAHEAD) != 0, want_ra);
    CHECK_EQ((prev.info.flags & SS_SNAPSHOT_EXCLUSIVE) != 0, want_x);
    if (want_ra) CHECK(prev.info.readahead_requested > 0 && prev.info.readahead_buffers == 2);
    if (want_x) CHECK(prev.info.exclusive_writebacks > 0 && prev.info.exclusive_blocks > 0);
    CHECK(prev.total.routed > 0 && prev.total.transient > 0);
  }
}

// readahead_units > 0 with no unit readable from a file (all pinned) allocates no buffer: no READAHEAD flag, no
// "readahead" object.
TEST(snapshot_readahead_flag_is_effective) {
  Fixture fx("snap-ra-none");
  Cfg cfg;
  cfg.tiers = R"({"tiers":["","PPPPPPPPPPPPPPPP","PPPPPPPP","P","PPPPPPPPPPPPPPPP"]})";
  cfg.dynamic = 0;
  cfg.readahead = 4;
  Run r;
  if (!open_run(r, fx, cfg)) return;
  check_unpublished(r.c);
  std::mt19937 g(3);
  forward(r.c, g, 0, false);
  Expect e = expect_now(r.c, 1);
  CHECK_EQ(e.info.flags, 0u);
  CHECK(e.json.find("\"readahead\"") == std::string::npos);
  check_published(r.c, e);
}

// Resets: between forwards they publish; inside a forward (also with only a between-forward host prefetch open) they
// do not, and the next ss_end publishes the live state.
TEST(snapshot_reset_boundaries) {
  Fixture fx("snap-reset");
  Cfg cfg;
  cfg.tiers = kExcl;
  cfg.mode = SS_TIER_MODE_EXCLUSIVE;
  Run r;
  if (!open_run(r, fx, cfg)) return;
  check_unpublished(r.c);
  std::mt19937 g(11);
  uint64_t seq = 0;
  for (int s = 0; s < 6; ++s) forward(r.c, g, s, false);
  seq = 6;
  ss_stats_reset(r.c);
  Expect e = expect_now(r.c, ++seq);
  CHECK_EQ(e.total.routed, uint64_t(0));
  CHECK(e.info.exclusive_writebacks > 0);  // not reset by ss_stats_reset
  check_published(r.c, e);
  forward(r.c, g, 6, false);
  Expect before = expect_now(r.c, ++seq);
  check_published(r.c, before);
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  forward_body(r.c, g, 7, false, [&] {
    ss_stats_reset(r.c);  // inside a forward: live counters reset, nothing published
    check_published(r.c, before);
  });
  check_published(r.c, before);
  CHECK_OK(ss_end(r.c));
  Expect after = expect_now(r.c, ++seq);
  check_published(r.c, after);

  // Read-ahead: a host prefetch between forwards opens the next one (no ss_begin yet).
  Fixture fy("snap-reset-ra");
  Cfg ra;
  ra.tiers = kDisk;
  ra.readahead = 2;
  ra.dynamic = 0;
  Run q;
  if (!open_run(q, fy, ra)) return;
  check_unpublished(q.c);
  for (int s = 0; s < 3; ++s) forward(q.c, g, s, true);
  Expect base = expect_now(q.c, 3);
  check_published(q.c, base);
  CHECK_OK(ss_host_prefetch_layer(q.c, 3));
  CHECK_OK(ss_test_ra_quiesce(q.c));
  check_published(q.c, base);  // requested between forwards: counted at the next ss_end
  ss_stats_reset(q.c);         // the forward is open: nothing published
  check_published(q.c, base);
  forward(q.c, g, 3, true);
  Expect next = expect_now(q.c, 4);
  CHECK(next.info.readahead_requested >= 1);  // the reset cleared the earlier ones; the between-forward one counts
  check_published(q.c, next);
}

// ss_seed between forwards (exclusive: its promotions free blocks) shows at the next publication only.
TEST(snapshot_seed_visible_at_next_boundary) {
  Fixture fx("snap-seed");
  Cfg cfg;
  cfg.tiers = R"({"tiers":["","PPPPPPPPPPPPPPPP","RRRRPPPP","P","PPPPPPPPPPPPPPPP"]})";
  cfg.mode = SS_TIER_MODE_EXCLUSIVE;
  cfg.policy = SS_POLICY_LRU;
  Run r;
  if (!open_run(r, fx, cfg)) return;
  check_unpublished(r.c);
  std::mt19937 g(5);
  ss_stats_reset(r.c);  // publishes the open state (between forwards)
  Expect base = expect_now(r.c, 1);
  check_published(r.c, base);
  int32_t ids[2] = {11, 13};
  CHECK_OK(ss_seed(r.c, 1, ids, 2));
  ss_exclusive_stats x{};
  CHECK_OK(ss_stats_exclusive(r.c, &x));
  CHECK_EQ(x.promoted, base.info.exclusive_promoted + 2);
  check_published(r.c, base);
  forward(r.c, g, 1, false);
  Expect next = expect_now(r.c, 2);
  CHECK(next.info.exclusive_promoted >= x.promoted);
  check_published(r.c, next);
}

// Failures: an ss_end that fails before folding keeps the publication; a forward whose landing failed is folded and
// published; a poisoned context keeps its last publication (resets publish nothing more).
TEST(snapshot_failures_and_poison) {
  Fixture fx("snap-fail");
  Cfg ra;
  ra.tiers = kDisk;
  ra.readahead = 4;
  ra.dynamic = 0;
  Run r;
  if (!open_run(r, fx, ra)) return;
  check_unpublished(r.c);
  std::mt19937 g(9);
  forward(r.c, g, 0, true);
  Expect e1 = expect_now(r.c, 1);
  CHECK_EQ(int(ss_end(r.c)), int(SS_E_INVALID));  // no forward open
  check_published(r.c, e1);
  // Landing copy fails (injected) before any copy: ss_acquire and ss_end report it; the forward is folded.
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(r.c, 3));
  CHECK_OK(ss_test_ra_quiesce(r.c));
  CHECK_OK(ss_test_inject(r.c, 3));
  CHECK_OK(ss_layer_all(r.c, 3));
  ss_layer_view v{};
  CHECK_EQ(int(ss_acquire(r.c, 3, nullptr, &v)), int(SS_E_IO));
  CHECK(ss_end(r.c) != SS_OK);
  Expect e2 = expect_now(r.c, 2);
  CHECK_EQ(e2.info.forwards, uint64_t(2));
  check_published(r.c, e2);
  forward(r.c, g, 2, true);  // usable after a library error
  Expect e3 = expect_now(r.c, 3);
  check_published(r.c, e3);
  // Queueing a landing fails (injected allocation failure): the context is poisoned.
  CHECK_OK(ss_begin(r.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(r.c, 3));
  CHECK_OK(ss_test_ra_quiesce(r.c));
  CHECK_OK(ss_test_inject(r.c, 2));
  CHECK_OK(ss_layer_all(r.c, 3));
  CHECK_EQ(int(ss_acquire(r.c, 3, nullptr, &v)), int(SS_E_NOMEM));
  CHECK_EQ(int(ss_end(r.c)), int(SS_E_BACKEND));
  check_published(r.c, e3);
  ss_stats_reset(r.c);
  check_published(r.c, e3);

  // Poisoned before any publication: nothing to show, a reset does not publish.
  Fixture fy("snap-poison0");
  Run q;
  if (!open_run(q, fy, ra)) return;
  check_unpublished(q.c);
  CHECK_OK(ss_begin(q.c, nullptr, 0));
  CHECK_OK(ss_host_prefetch_layer(q.c, 3));
  CHECK_OK(ss_test_ra_quiesce(q.c));
  CHECK_OK(ss_test_inject(q.c, 2));
  CHECK_OK(ss_layer_all(q.c, 3));
  CHECK_EQ(int(ss_acquire(q.c, 3, nullptr, &v)), int(SS_E_NOMEM));
  CHECK_EQ(int(ss_end(q.c)), int(SS_E_BACKEND));
  check_unpublished(q.c);
  ss_stats_reset(q.c);
  check_unpublished(q.c);
}

// Sized struct: older (shorter) and newer (longer) callers, partial fields, bad arguments.
TEST(snapshot_abi_sizes) {
  Fixture fx("snap-abi");
  Run r;
  if (!open_run(r, fx, Cfg{})) return;
  check_unpublished(r.c);
  std::mt19937 g(2);
  forward(r.c, g, 0, false);
  Expect e = expect_now(r.c, 1);
  ss_layer_stats t;
  CHECK_EQ(int(ss_stats_snapshot(nullptr, &t, nullptr)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_stats_snapshot_sized(r.c, nullptr, nullptr, 160)), int(SS_E_INVALID));
  char* js = nullptr;
  CHECK_EQ(int(ss_stats_snapshot_json(nullptr, &js, nullptr)), int(SS_E_INVALID));
  CHECK_EQ(int(ss_stats_snapshot_json(r.c, nullptr, nullptr)), int(SS_E_INVALID));
  CHECK_OK(ss_stats_snapshot(r.c, &t, nullptr));
  CHECK(same_stats(t, e.total));
  const size_t big = sizeof(ss_snapshot_info) + 32;
  for (uint32_t size : {0u, 1u, 3u, 4u, 8u, 12u, 16u, 24u, 100u, 160u, 192u}) {
    unsigned char raw[sizeof(ss_snapshot_info) + 32];
    std::memset(raw, 0xEE, big);
    std::memset(&t, 0xEE, sizeof t);
    const ss_layer_stats t0 = t;
    ss_status st = ss_stats_snapshot_sized(r.c, &t, reinterpret_cast<ss_snapshot_info*>(raw), size);
    if (size < 4) {
      CHECK_EQ(int(st), int(SS_E_INVALID));
      CHECK(same_stats(t, t0));
      for (size_t k = 0; k < big; ++k) CHECK_EQ(int(raw[k]), 0xEE);
      continue;
    }
    CHECK_OK(st);
    CHECK(same_stats(t, e.total));
    const uint32_t wrote = size < 160 ? size : 160;
    uint32_t ss = 0;
    std::memcpy(&ss, raw, 4);
    CHECK_EQ(ss, wrote);
    ss_snapshot_info want = e.info;
    want.struct_size = wrote;
    CHECK(std::memcmp(raw + 4, reinterpret_cast<const unsigned char*>(&want) + 4, wrote - 4) == 0);
    for (size_t k = wrote; k < big; ++k) CHECK_EQ(int(raw[k]), 0xEE);
  }
}

// Readers on other threads while the engine runs forwards. The engine pauses inside some forwards until every reader
// has read again (a forced overlap with a half-updated live state) and records each expected publication; readers
// keep their own samples, checked after the join.
TEST(snapshot_concurrent_readers) {
  Cfg ra;
  ra.tiers = kDisk;
  ra.readahead = 2;
  ra.dynamic = 4;
  ra.policy = SS_POLICY_SCORED;
  Cfg ex;
  ex.tiers = kExcl;
  ex.mode = SS_TIER_MODE_EXCLUSIVE;
  ex.policy = SS_POLICY_LRU;
  int scenario = 0;
  for (const Cfg& cfg : {ra, ex}) {
    Fixture fx(scenario++ ? "snap-mt-excl" : "snap-mt-ra");
    Run r;
    if (!open_run(r, fx, cfg)) return;
    check_unpublished(r.c);  // armed: every ss_end publishes, sequence k = forward k
    constexpr int kReaders = 3, kForwards = 400, kPauseEvery = 16;
    constexpr size_t kMaxSamples = 4000;
    struct Sample {
      ss_layer_stats total;
      ss_snapshot_info info;
    };
    struct Reader {
      std::vector<Sample> structs;
      std::vector<std::string> docs;
      std::atomic<uint64_t> reads{0};
      int errors = 0, order = 0, unpublished = 0;
    };
    std::vector<Reader> readers(kReaders);
    std::vector<Expect> expected(size_t(kForwards) + 1);
    std::atomic<int> ready{0};
    std::atomic<bool> go{false}, stop{false};
    std::vector<std::thread> threads;
    for (int k = 0; k < kReaders; ++k)
      threads.emplace_back([&, k] {
        Reader& me = readers[size_t(k)];
        ready.fetch_add(1);
        while (!go.load()) std::this_thread::yield();
        uint64_t last_seq = 0;
        for (uint64_t n = 0; !stop.load(); ++n) {
          if ((n + uint64_t(k)) % 2 == 0) {
            Sample s;
            ss_status st = ss_stats_snapshot(r.c, &s.total, &s.info);
            if (st == SS_E_INVALID) ++me.unpublished;
            else if (st != SS_OK) ++me.errors;
            else {
              if (s.info.sequence < last_seq) ++me.order;
              last_seq = s.info.sequence;
              if (me.structs.size() < kMaxSamples) me.structs.push_back(s);
              else me.structs[size_t(n / 2) % kMaxSamples] = s;  // keep sampling the whole run
            }
          } else {
            std::string js;
            ss_status st = snap_json(r.c, &js);
            if (st == SS_E_INVALID) ++me.unpublished;
            else if (st != SS_OK) ++me.errors;
            else if (me.docs.size() < kMaxSamples) me.docs.push_back(std::move(js));
            else me.docs[size_t(n / 2) % kMaxSamples] = std::move(js);
          }
          me.reads.fetch_add(1);
        }
      });
    while (ready.load() < kReaders) std::this_thread::yield();
    go = true;
    std::mt19937 g(17);
    int stalls = 0;
    for (int step = 0; step < kForwards; ++step) {
      CHECK_OK(ss_begin(r.c, nullptr, 0));
      forward_body(r.c, g, step, cfg.readahead > 0, [&] {
        if (step % kPauseEvery != 0) return;
        std::vector<uint64_t> seen;
        for (auto& rd : readers) seen.push_back(rd.reads.load());
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        for (size_t k = 0; k < readers.size(); ++k)
          while (readers[k].reads.load() < seen[k] + 2) {
            if (std::chrono::steady_clock::now() > until) {
              ++stalls;
              break;
            }
            std::this_thread::yield();
          }
      });
      CHECK_OK(ss_end(r.c));
      expected[size_t(step) + 1] = expect_now(r.c, uint64_t(step) + 1);
    }
    stop = true;
    for (auto& t : threads) t.join();
    CHECK_EQ(stalls, 0);
    size_t n_structs = 0, n_docs = 0, distinct = 0;
    std::set<uint64_t> seqs;
    for (const Reader& rd : readers) {
      CHECK_EQ(rd.errors, 0);
      CHECK_EQ(rd.order, 0);
      for (const Sample& s : rd.structs) {
        ++n_structs;
        seqs.insert(s.info.sequence);
        if (s.info.sequence < 1 || s.info.sequence > uint64_t(kForwards)) {
          CHECK(false);
          continue;
        }
        const Expect& e = expected[size_t(s.info.sequence)];
        CHECK(same_stats(s.total, e.total));
        CHECK(same_info(s.info, e.info));
      }
      for (const std::string& js : rd.docs) {
        ++n_docs;
        const uint64_t f = json_u64(js, "", "forwards");
        if (f < 1 || f > uint64_t(kForwards)) {
          CHECK(false);
          continue;
        }
        CHECK_EQ(js, expected[size_t(f)].json);
      }
    }
    distinct = seqs.size();
    std::printf("  %s: %zu struct and %zu document samples checked, %zu distinct publications seen\n",
                cfg.mode ? "exclusive" : "read-ahead", n_structs, n_docs, distinct);
    CHECK(n_structs > 0 && n_docs > 0 && distinct > 1);
    check_published(r.c, expected[size_t(kForwards)]);
  }
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }
