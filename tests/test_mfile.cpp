#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <thread>

#include "check.hpp"
#include "errors.hpp"
#include "json.hpp"
#include "mfile.hpp"
#include "raw_file.hpp"
#include "stripe_manifest.hpp"
#include "xxh64.hpp"

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

using namespace ts;
namespace fs = std::filesystem;

namespace {
constexpr uint64_t C = uint64_t(1) << 20;
struct Buffer {
  char* p;
  explicit Buffer(size_t n) {
#if defined(_WIN32)
    p = static_cast<char*>(_aligned_malloc(n, 4096));
#else
    void* raw = nullptr;
    if (posix_memalign(&raw, 4096, n)) raw = nullptr;
    p = static_cast<char*>(raw);
#endif
    if (!p) throw std::bad_alloc();
  }
  ~Buffer() {
#if defined(_WIN32)
    _aligned_free(p);
#else
    std::free(p);
#endif
  }
};

void put(const fs::path& p, const void* data, size_t n) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(static_cast<const char*>(data), std::streamsize(n));
  require(bool(f), SS_E_IO, "test file write");
}
void env(const char* name, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  if (value.empty()) unsetenv(name);
  else setenv(name, value.c_str(), 1);
#endif
}
struct Fixture {
  fs::path root, primary, secondary, cache;
  std::string old_cache;
  std::vector<char> bytes;
  StripeManifest manifest;
  Fixture() {
    static std::atomic<unsigned> serial{0};
    root = fs::current_path() / ("mfile-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                                 "-" + std::to_string(serial.fetch_add(1)));
    primary = root / "primary" / "weights.bin";
    secondary = root / "secondary";
    cache = root / "cache";
    fs::create_directories(primary.parent_path());
    fs::create_directories(secondary);
    fs::create_directories(cache);
#if defined(_WIN32)
    const char* cache_env = "LOCALAPPDATA";
#else
    const char* cache_env = "XDG_CACHE_HOME";
#endif
    if (const char* p = std::getenv(cache_env)) old_cache = p;
    env(cache_env, cache.u8string());
    bytes.resize(size_t(39 * C + 123));
    uint32_t r = 17;
    for (size_t i = 0; i < bytes.size(); ++i) {
      r ^= r << 13; r ^= r >> 17; r ^= r << 5;
      bytes[i] = char(r);
    }
    put(primary, bytes.data(), bytes.size());
    manifest.chunk_bytes = C;
    manifest.primary_basename = primary.filename().u8string();
    manifest.primary_size = bytes.size();
    manifest.primary_sha256 = sha256_file(primary.u8string());
    manifest.data_name = "stripe.data";
    std::ofstream data(secondary / manifest.data_name, std::ios::binary);
    std::vector<char> slot(size_t(C), 0);
    for (uint64_t chunk = 1; chunk < 40; chunk += 2) {
      uint64_t n = std::min(C, uint64_t(bytes.size()) - chunk * C);
      std::fill(slot.begin(), slot.end(), char(0));
      std::memcpy(slot.data(), bytes.data() + chunk * C, size_t(n));
      manifest.runs.push_back({chunk, 1, manifest.hashes.size()});
      manifest.hashes.push_back(xxh64(slot.data(), size_t(n)));
      data.write(slot.data(), std::streamsize(C));
    }
    data.close();
    manifest.data_size = manifest.hashes.size() * C;
    publish();
  }
  ~Fixture() {
#if defined(_WIN32)
    env("LOCALAPPDATA", old_cache);
#else
    env("XDG_CACHE_HOME", old_cache);
#endif
    std::error_code ec;
    fs::remove_all(root, ec);
  }
  void publish() {
    write_file_atomic((secondary / stripe_manifest_name(primary.filename().u8string())).u8string(), stripe_manifest_write(manifest));
  }
  MFile::Options options(int verify = SS_MFILE_VERIFY_ALWAYS, bool direct = true) const {
    MFile::Options o;
    o.primary_sha256 = manifest.primary_sha256;
    o.verify_mode = verify;
    o.direct = direct;
    o.drive_readers = 2;
    return o;
  }
  // stripe = true: every request may use the stripe (the failure-path tests read single chunks from it)
  std::shared_ptr<MFile> open(int verify = SS_MFILE_VERIFY_ALWAYS, bool direct = true, bool stripe = true) const {
    auto f = MFile::open(primary.u8string(), secondary.u8string(), options(verify, direct));
    f->test_prefer_stripe(stripe);
    return f;
  }
  void corrupt(uint64_t slot) {
    std::fstream f(secondary / manifest.data_name, std::ios::binary | std::ios::in | std::ios::out);
    f.seekp(std::streamoff(slot * C));
    char b = char(bytes[size_t(manifest.runs[size_t(slot)].first_chunk * C)] ^ 1);
    f.write(&b, 1);
    require(bool(f), SS_E_IO, "test corruption write");
  }
  bool equal(MFile& f, uint64_t off, uint64_t n) const {
    Buffer dst(size_t(n ? n : 4096));
    uint64_t want = off < bytes.size() ? std::min(n, bytes.size() - off) : 0;
    uint64_t got = f.pread(dst.p, n, off);
    return got == want && (!got || std::memcmp(dst.p, bytes.data() + off, size_t(got)) == 0);
  }
};

ss_mfile_counters counters(const MFile& f) {
  ss_mfile_counters c{};
  c.struct_size = sizeof c;
  f.counters(&c);
  return c;
}
template <class F> int status(F fn) {
  try { fn(); return SS_OK; } catch (const Error& e) { return e.status(); }
}

std::string signed_text(Json j) {
  j.obj.erase("digest");
  std::string text = json_write(j);
  char hex[17];
  std::snprintf(hex, sizeof hex, "%016llx", static_cast<unsigned long long>(xxh64(text.data(), text.size())));
  j["digest"] = Json::string(hex);
  return json_write(j);
}

StripeManifest small_manifest() {
  StripeManifest m;
  m.chunk_bytes = C; m.primary_basename = "weights.bin"; m.primary_size = 40 * C;
  m.primary_sha256 = std::string(64, '0'); m.data_name = "stripe.data"; m.data_size = 2 * C;
  m.runs = {{1, 1, 0}, {3, 1, 1}}; m.hashes = {0, UINT64_MAX};
  return m;
}
}  // namespace

TEST(manifest_roundtrip_and_rejections) {
  StripeManifest m = small_manifest();
  std::string text = stripe_manifest_write(m);
  CHECK_EQ(stripe_manifest_write(stripe_manifest_parse(text)), text);
  CHECK(text.find("ffffffffffffffff") != std::string::npos);
  CHECK_EQ(stripe_manifest_name("weights.bin"), std::string("weights.bin.stripe.json"));
  Json original = json_parse(text);
  CHECK_EQ(text, signed_text(original));
  CHECK_EQ(stripe_manifest_write(stripe_manifest_parse(" \n" + text + "\n")), text);
  auto reject = [&](std::function<void(Json&)> edit) {
    Json j = original;
    edit(j);
    CHECK_EQ(status([&] { stripe_manifest_parse(signed_text(j)); }), int(SS_E_DESC));
  };
  reject([](Json& j) { j["schema"] = Json::string("unknown"); });
  reject([](Json& j) { j.obj.erase("primary"); });
  reject([](Json& j) { j["extra"] = Json::number(uint64_t(0)); });
  for (uint64_t c : {uint64_t(0), C / 2, C + 4096, uint64_t(512) << 20})
    reject([&](Json& j) { j["chunk_bytes"] = Json::number(c); });
  reject([](Json& j) { j["primary"]["size"] = Json::number((uint64_t(1) << 50) + 1); });
  reject([](Json& j) { j["primary"]["size"] = Json::number((kStripeMaxChunks + 1) * C); });
  reject([](Json& j) { j["primary"]["size"] = Json::number(uint64_t(1) << 53); });
  reject([](Json& j) { j["primary"]["sha256"] = Json::string(std::string(64, 'A')); });
  reject([](Json& j) { j["primary"]["sha256"] = Json::string("00"); });
  for (const char* name : {"../x", "x..y", "a/b", "a\\b", "a:b", ".", "", "weights.bin"})
    reject([&](Json& j) { j["data"]["name"] = Json::string(name); });
#if defined(_WIN32)
  reject([](Json& j) { j["data"]["name"] = Json::string("WEIGHTS.BIN"); });
#endif
  reject([](Json& j) { j["data"]["size"] = Json::number(C); });
  reject([](Json& j) { j["data"]["size"].text = "9007199254740992"; });
  reject([](Json& j) { j["runs"] = Json::object(); });
  reject([](Json& j) { j["runs"].arr[0].arr.pop_back(); });
  reject([](Json& j) { j["runs"].arr[0].arr[1] = Json::number(uint64_t(0)); });
  reject([](Json& j) { j["runs"].arr[0].arr[0] = Json::number(uint64_t(40)); });
  reject([](Json& j) { j["runs"].arr[0].arr[1] = Json::number(uint64_t(40)); });
  reject([](Json& j) { j["runs"].arr[1].arr[0] = Json::number(uint64_t(1)); });
  reject([](Json& j) { std::swap(j["runs"].arr[0], j["runs"].arr[1]); });
  reject([](Json& j) { j["runs"].arr[1].arr[2] = Json::number(uint64_t(0)); });
  reject([](Json& j) { j["runs"].arr[1].arr[2] = Json::number(uint64_t(2)); });
  for (const char* n : {"-1", "1.0", "1e0", "9007199254740992", "18446744073709551615", "18446744073709551616"})
    reject([&](Json& j) { j["runs"].arr[0].arr[1].text = n; });
  reject([](Json& j) { j["hashes"].arr.pop_back(); });
  reject([](Json& j) { j["hashes"].arr[0] = Json::string("FFFFFFFFFFFFFFFF"); });
  reject([](Json& j) { j["hashes"].arr[0] = Json::string("0"); });
  reject([](Json& j) { j["hashes"].arr[0] = Json::number(uint64_t(0)); });
  Json bad = original;
  bad["digest"] = Json::string(std::string(16, '0'));
  CHECK_EQ(status([&] { stripe_manifest_parse(json_write(bad)); }), int(SS_E_DESC));
  CHECK_EQ(status([&] { stripe_manifest_parse(text + "x"); }), int(SS_E_DESC));
  CHECK_EQ(status([&] { stripe_manifest_parse("{\"schema\":0,\"schema\":0}"); }), int(SS_E_DESC));
  // Cardinality is rejected without building a million Json arrays.
  std::string excessive = "{\"runs\":[";
  for (uint64_t i = 0; i <= kStripeMaxRuns; ++i) excessive += i ? ",[0,1,0]" : "[0,1,0]";
  excessive += "]}";
  CHECK_EQ(status([&] { stripe_manifest_parse(excessive); }), int(SS_E_DESC));
}

TEST(mfile_reads_and_capi) {
  Fixture t;
  for (bool direct : {false, true}) {
    auto f = t.open(SS_MFILE_VERIFY_ALWAYS, direct);
    CHECK(f->multi());
    auto c = counters(*f);
    CHECK_EQ(c.locations, uint32_t(2));
    CHECK_EQ(c.verify_mode, int(SS_MFILE_VERIFY_ALWAYS));
    CHECK_EQ(c.verified_bytes, uint64_t(19 * C + 123));
    CHECK_EQ(c.disabled_mask, uint32_t(0));
    if (!direct) CHECK_EQ(c.direct_mask, uint32_t(0));
    CHECK(t.equal(*f, C - 4096, 8192));
    CHECK(t.equal(*f, 0, 8 * C));
    CHECK(t.equal(*f, 39 * C, 4096));
    CHECK(t.equal(*f, 38 * C, C + 4096));
    CHECK(t.equal(*f, 39 * C + 4096, 0));
    std::mt19937 rng(27);
    for (int i = 0; i < 100; ++i) {
      uint64_t off = uint64_t(rng() % (38 * C / 4096)) * 4096;
      uint64_t n = uint64_t(1 + rng() % 256) * 4096;
      CHECK(t.equal(*f, off, n));
    }
    Buffer data(size_t(8 * C));
    std::vector<ss_mfile_io> ios = {{data.p, C - 4096, 8192}, {data.p + 2 * C, 2 * C, 2 * C},
                                    {data.p + 4 * C, 39 * C, 4096}, {data.p + 5 * C, 5 * C, C}};
    f->preadv(ios.data(), uint32_t(ios.size()));
    for (const auto& io : ios)
      CHECK(std::memcmp(io.dst, t.bytes.data() + io.offset, size_t(std::min(io.bytes, t.bytes.size() - io.offset))) == 0);
    CHECK_EQ(status([&] { f->pread(data.p + 1, 4096, 0); }), int(SS_E_INVALID));
    CHECK_EQ(status([&] { f->pread(data.p, 4095, 0); }), int(SS_E_INVALID));
    CHECK_EQ(status([&] { f->pread(data.p, 4096, 1); }), int(SS_E_INVALID));
    CHECK_EQ(status([&] { f->pread(data.p, 8192, 39 * C); }), int(SS_E_INVALID));
    CHECK_EQ(status([&] { f->pread(data.p, 4096, UINT64_MAX - 4095); }), int(SS_E_INVALID));
    CHECK_EQ(status([&] { f->preadv(nullptr, 1); }), int(SS_E_INVALID));
    f->preadv(nullptr, 0);
  }
  ss_mfile_options o;
  ss_mfile_options_defaults(&o);
  o.verify_mode = SS_MFILE_VERIFY_ALWAYS;
  o.primary_sha256 = t.manifest.primary_sha256.c_str();
  ss_mfile* f = nullptr;
  CHECK_OK(ss_mfile_open(t.primary.u8string().c_str(), t.secondary.u8string().c_str(), &o, &f));
  CHECK(f != nullptr);
  CHECK_EQ(ss_mfile_size(f), uint64_t(t.bytes.size()));
  CHECK_EQ(ss_mfile_alignment(f), uint32_t(4096));
  CHECK_EQ(ss_mfile_multi(f), 1);
  CHECK(ss_mfile_location_path(f, 1) != nullptr);
  CHECK(ss_mfile_location_path(f, 2) == nullptr);
  CHECK_EQ(std::string(ss_mfile_open_report(f)), std::string());
  Buffer data(4096);
  uint64_t got = 0;
  CHECK_OK(ss_mfile_pread(f, data.p, 4096, C, &got));
  CHECK_EQ(got, uint64_t(4096));
  CHECK(std::memcmp(data.p, t.bytes.data() + C, 4096) == 0);
  ss_mfile_io io{data.p, 39 * C, 4096};
  CHECK_OK(ss_mfile_preadv(f, &io, 1));
  CHECK(std::memcmp(data.p, t.bytes.data() + 39 * C, 123) == 0);
  // a large read (>= 8 MiB) also uses the stripe; the small reads above stay on the primary
  Buffer span(size_t(8 * C));
  CHECK_OK(ss_mfile_pread(f, span.p, 8 * C, 0, &got));
  CHECK_EQ(got, uint64_t(8 * C));
  CHECK(std::memcmp(span.p, t.bytes.data(), size_t(8 * C)) == 0);
  ss_mfile_counters c{};
  c.struct_size = sizeof c;
  CHECK_OK(ss_mfile_get_counters(f, &c));
  CHECK(c.bytes[1] > 0);
  ss_mfile_close(f);
  CHECK(ss_mfile_pread(nullptr, data.p, 4096, 0, &got) == SS_E_INVALID);
  CHECK(std::strlen(ss_last_error()) > 0);
  alignas(ss_mfile_options) unsigned char short_options[16];
  std::memset(short_options, 0xa5, sizeof short_options);
  ss_mfile_options_defaults_sized(reinterpret_cast<ss_mfile_options*>(short_options), 8);
  CHECK_EQ(*reinterpret_cast<uint32_t*>(short_options), uint32_t(8));
  CHECK_EQ(short_options[8], 0xa5);
}

TEST(mfile_corruption_and_failure_fallback) {
  Fixture t;
  t.corrupt(0);
  {
    auto f = t.open();
    CHECK(t.equal(*f, C, C));
    CHECK(t.equal(*f, C, 4096));
    auto c = counters(*f);
    CHECK_EQ(c.verify_fail, uint64_t(1));
    CHECK_EQ(c.demoted_chunks, uint64_t(1));
    CHECK_EQ(c.bytes[1], uint64_t(0));
  }
  // Restore before opening; the serving handle excludes writers once open.
  // Rebuild the fixture's stripe data without the tool.
  {
    std::ofstream data(t.secondary / t.manifest.data_name, std::ios::binary);
    std::vector<char> slot(size_t(C), 0);
    for (const auto& run : t.manifest.runs) {
      std::fill(slot.begin(), slot.end(), char(0));
      size_t n = size_t(std::min(C, uint64_t(t.bytes.size()) - run.first_chunk * C));
      std::memcpy(slot.data(), t.bytes.data() + run.first_chunk * C, n);
      data.write(slot.data(), std::streamsize(C));
    }
  }
  for (int kind : {1, 2, 3}) {
    auto f = t.open();
    f->test_inject(1, kind, 0);
    CHECK(t.equal(*f, C, C));
    auto first = counters(*f);
    CHECK_EQ(first.demoted_chunks, uint64_t(1));
    CHECK(first.fallback_bytes >= C);
    if (kind == 3) CHECK_EQ(first.verify_fail, uint64_t(1));
    else CHECK_EQ(first.read_errors, uint64_t(1));
    // Distinct chunks still assigned to the failed location count toward its disable threshold.
    CHECK(t.equal(*f, 3 * C, C));
    CHECK(t.equal(*f, 5 * C, C));
    CHECK(counters(*f).disabled_mask & 2);
    CHECK(t.equal(*f, 7 * C, C));
  }
  for (int kind : {1, 2, 3}) {
    auto f = t.open();
    f->test_inject(1, kind, 0);
    Buffer dst(size_t(4 * C));
    ss_mfile_io ios[] = {{dst.p, C, C}, {dst.p + C, 3 * C, C}, {dst.p + 2 * C, 5 * C, C},
                          {dst.p + 3 * C, 0, C}};
    f->preadv(ios, 4);
    for (const auto& io : ios) CHECK(std::memcmp(io.dst, t.bytes.data() + io.offset, size_t(io.bytes)) == 0);
    auto c = counters(*f);
    CHECK(c.disabled_mask & 2);
    CHECK(c.fallback_bytes >= 3 * C);
  }
}

TEST(mfile_consecutive_chunks_to_unaligned_eof) {
  // the last two chunks placed back to back (the final one short): a read covering the end of chunk 38 and the whole
  // final chunk must still treat the final chunk as whole, so ALWAYS re-checks it inline
  Fixture t;
  t.manifest.runs = {{38, 2, 0}};
  t.manifest.hashes.clear();
  {
    std::ofstream data(t.secondary / t.manifest.data_name, std::ios::binary | std::ios::trunc);
    std::vector<char> slot(static_cast<size_t>(C));
    for (uint64_t chunk = 38; chunk < 40; ++chunk) {
      std::fill(slot.begin(), slot.end(), char(0));
      size_t n = size_t(std::min(C, uint64_t(t.bytes.size()) - chunk * C));
      std::memcpy(slot.data(), t.bytes.data() + chunk * C, n);
      t.manifest.hashes.push_back(xxh64(slot.data(), n));
      data.write(slot.data(), std::streamsize(C));
    }
  }
  t.manifest.data_size = 2 * C;
  t.publish();
  auto f = t.open();
  CHECK(t.equal(*f, 38 * C, C + 4096));
  CHECK_EQ(counters(*f).verify_fail, uint64_t(0));
  f->test_inject(1, 3, 0);
  CHECK(t.equal(*f, 39 * C - 4096, 8192));
  auto c = counters(*f);
  CHECK_EQ(c.verify_fail, uint64_t(1));
  CHECK_EQ(c.demoted_chunks, uint64_t(1));
  CHECK(t.equal(*f, 39 * C, 4096));
}

TEST(mfile_size_rule_routing) {
  // the primary holds every byte: requests under 8 MiB and short batches read the primary only; large requests and
  // batches of at least 64 scattered reads also use the stripe; bytes are the primary's either way
  Fixture t;
  auto f = t.open(SS_MFILE_VERIFY_ALWAYS, true, false);
  auto base = counters(*f);
  CHECK(t.equal(*f, C, C));
  CHECK(t.equal(*f, 0, 7 * C));
  auto small = counters(*f);
  CHECK_EQ(small.bytes[1], base.bytes[1]);
  CHECK(t.equal(*f, 0, 8 * C));
  auto large = counters(*f);
  CHECK_EQ(large.bytes[1] - small.bytes[1], uint64_t(4 * C));
  CHECK_EQ(large.bytes[0] - small.bytes[0], uint64_t(4 * C));
  std::vector<std::unique_ptr<Buffer>> pages;  // Buffer owns raw aligned memory: never copied
  std::vector<ss_mfile_io> ios;
  // 64 pages on stripe chunks (odd chunks, 4 KiB apart inside them)
  for (int i = 0; i < 64; ++i) pages.push_back(std::make_unique<Buffer>(4096));
  for (int i = 0; i < 64; ++i) ios.push_back({pages[size_t(i)]->p, uint64_t(2 * (i % 19) + 1) * C + uint64_t(i / 19) * 4096, 4096});
  f->preadv(ios.data(), 63);
  auto short_batch = counters(*f);
  CHECK_EQ(short_batch.bytes[1], large.bytes[1]);
  f->preadv(ios.data(), 64);
  auto batch = counters(*f);
  CHECK_EQ(batch.bytes[1] - short_batch.bytes[1], uint64_t(32 * 4096));
  CHECK_EQ(batch.bytes[0] - short_batch.bytes[0], uint64_t(32 * 4096));
  for (int i = 0; i < 64; ++i)
    CHECK(std::memcmp(pages[size_t(i)]->p, t.bytes.data() + uint64_t(2 * (i % 19) + 1) * C + uint64_t(i / 19) * 4096, 4096) == 0);
  CHECK_EQ(batch.fallback_bytes, uint64_t(0));
}

TEST(mfile_adaptive_shares_turn_a_slow_stripe_off) {
  // large requests: shares follow measured drive throughput, every 16th request reads the primary alone, and while
  // split requests are slower than that baseline the stripe gets nothing; bytes are the primary's throughout
  Fixture t;
  auto f = t.open(SS_MFILE_VERIFY_CACHED, true, false);
  for (int i = 0; i < 4; ++i) CHECK(t.equal(*f, 0, 8 * C));
  CHECK(counters(*f).bytes[1] > 0);  // equal drives: the stripe takes part
  f->test_delay(1, 30000);           // the stripe becomes far slower than the primary
  for (int i = 0; i < 40; ++i) CHECK(t.equal(*f, uint64_t(i % 4) * 8 * C, 8 * C));
  auto off = counters(*f);
  for (int i = 0; i < 8; ++i) CHECK(t.equal(*f, 0, 8 * C));
  auto later = counters(*f);
  CHECK_EQ(later.bytes[1], off.bytes[1]);  // switched off: the stripe served nothing more
  CHECK_EQ(later.fallback_bytes, uint64_t(0));
}

TEST(mfile_rejected_locations) {
  Fixture t;
  auto rejected = [&](const std::string& dirs, MFile::Options o) {
    auto f = MFile::open(t.primary.u8string(), dirs, o);
    CHECK(!f->multi());
    CHECK(!f->reasons().empty());
    CHECK(counters(*f).disabled_mask & 2);
    CHECK(t.equal(*f, C - 4096, 2 * C));
  };
  rejected((t.root / "missing").u8string(), t.options());
  auto wrong = t.options();
  wrong.primary_sha256 = std::string(64, '0');
  rejected(t.secondary.u8string(), wrong);
  fs::rename(t.secondary / t.manifest.data_name, t.secondary / "saved");
  rejected(t.secondary.u8string(), t.options());
  fs::rename(t.secondary / "saved", t.secondary / t.manifest.data_name);
  fs::resize_file(t.secondary / t.manifest.data_name, C);
  rejected(t.secondary.u8string(), t.options());
  fs::remove(t.secondary / t.manifest.data_name);
  fs::create_hard_link(t.primary, t.secondary / t.manifest.data_name);
  // Matching file size is required before identity checking; both rejection routes are safe.
  rejected(t.secondary.u8string(), t.options());
}

TEST(mfile_registry_cache_and_directories) {
  Fixture t;
  auto primary_dir = t.primary.parent_path().u8string();
  auto dirs = MFile::secondary_dirs(t.primary.u8string(), primary_dir + ";" + t.secondary.u8string() + ";" +
                                   (t.secondary / ".").u8string() + ";;" + primary_dir);
  CHECK_EQ(dirs.size(), size_t(1));
#if defined(_WIN32)
  std::string upper = t.secondary.u8string();
  for (char& c : upper) if (c >= 'a' && c <= 'z') c = char(c - 'a' + 'A');
  CHECK_EQ(MFile::secondary_dirs(t.primary.u8string(), upper + ";" + t.secondary.u8string()).size(), size_t(1));
#endif
  std::string many;
  for (int i = 0; i < 8; ++i) many += (i ? ";" : "") + (t.root / ("other" + std::to_string(i))).u8string();
  CHECK_EQ(status([&] { MFile::secondary_dirs(t.primary.u8string(), many); }), int(SS_E_INVALID));
  CHECK(MFile::secondary_dirs(t.primary.u8string(), "").empty());
  CHECK(MFile::secondary_dirs(t.primary.u8string(), primary_dir).empty());
  {
    auto a = t.open(SS_MFILE_VERIFY_CACHED);
    auto b = t.open(SS_MFILE_VERIFY_CACHED);
    CHECK(a.get() == b.get());
    CHECK(counters(*a).verified_bytes > 0);
    auto stricter = t.open(SS_MFILE_VERIFY_ALWAYS);
    CHECK(a.get() != stricter.get());
    CHECK_EQ(counters(*stricter).verify_mode, int(SS_MFILE_VERIFY_ALWAYS));
    auto off = t.open(SS_MFILE_VERIFY_OFF);
    CHECK(off.get() == a.get());
    auto changed = t.options();
    changed.primary_sha256 = std::string(64, '0');
    auto mismatch = MFile::open(t.primary.u8string(), t.secondary.u8string(), changed);
    CHECK(mismatch.get() != a.get());
    CHECK(!mismatch->multi());
  }
  {
    auto second = t.open(SS_MFILE_VERIFY_CACHED);
    CHECK_EQ(counters(*second).verified_bytes, uint64_t(0));
    CHECK(t.equal(*second, C, 4096));
  }
  auto data = t.secondary / t.manifest.data_name;
  fs::last_write_time(data, fs::last_write_time(data) + std::chrono::seconds(2));
  {
    auto third = t.open(SS_MFILE_VERIFY_CACHED);
    CHECK(counters(*third).verified_bytes > 0);
  }
  {
    auto off = t.open(SS_MFILE_VERIFY_OFF);
    CHECK_EQ(counters(*off).verified_bytes, uint64_t(0));
    auto cached = t.open(SS_MFILE_VERIFY_CACHED);
    CHECK(off.get() != cached.get());
  }
  // Concurrent opens finish verification under the registry lock and return one session.
  std::vector<std::shared_ptr<MFile>> opened(8);
  std::vector<std::thread> threads;
  for (size_t i = 0; i < opened.size(); ++i) threads.emplace_back([&, i] { opened[i] = t.open(); });
  for (auto& thread : threads) thread.join();
  for (const auto& f : opened) CHECK(f.get() == opened.front().get());
}

TEST(mfile_multiple_locations_and_identity_aliases) {
  Fixture t;
  fs::path extra = t.root / "extra";
  fs::create_directories(extra);
  const std::string list = t.secondary.u8string() + ";" + extra.u8string();
  const std::string name = stripe_manifest_name(t.manifest.primary_basename);
  // A conflicting directory is rejected whole, rather than stealing the first directory's chunks.
  fs::copy_file(t.secondary / t.manifest.data_name, extra / t.manifest.data_name);
  write_file_atomic((extra / name).u8string(), stripe_manifest_write(t.manifest));
  {
    auto f = MFile::open(t.primary.u8string(), list, t.options());
    CHECK(f->multi());
    CHECK(counters(*f).disabled_mask & 4);
    CHECK(f->reasons().find("already claimed") != std::string::npos);
    CHECK(t.equal(*f, 0, 8 * C));
  }
  StripeManifest evens = t.manifest;
  evens.runs.clear(); evens.hashes.clear();
  for (uint64_t c = 0; c < 40; c += 2) {
    evens.runs.push_back({c, 1, evens.hashes.size()});
    evens.hashes.push_back(xxh64(t.bytes.data() + c * C, size_t(C)));
  }
  fs::remove(extra / t.manifest.data_name);
  fs::create_hard_link(t.secondary / t.manifest.data_name, extra / t.manifest.data_name);
  write_file_atomic((extra / name).u8string(), stripe_manifest_write(evens));
  {
    auto f = MFile::open(t.primary.u8string(), list, t.options());
    CHECK(counters(*f).disabled_mask & 4);
    CHECK(f->reasons().find("aliases") != std::string::npos);
    CHECK(t.equal(*f, 0, 8 * C));
  }
  fs::remove(extra / t.manifest.data_name);
  {
    std::ofstream data(extra / t.manifest.data_name, std::ios::binary);
    for (const auto& r : evens.runs) data.write(t.bytes.data() + r.first_chunk * C, std::streamsize(C));
  }
  {
    auto f = MFile::open(t.primary.u8string(), list, t.options());
    CHECK_EQ(counters(*f).disabled_mask, uint32_t(0));
    CHECK_EQ(counters(*f).locations, uint32_t(3));
    CHECK(t.equal(*f, 0, 8 * C));
    auto c = counters(*f);
    CHECK(c.bytes[1] && c.bytes[2]);
  }
  // Use an exactly chunk-sized primary so the alias also passes the physical data-size check.
  fs::path alias_primary = t.root / "alias.bin";
  put(alias_primary, t.bytes.data(), size_t(20 * C));
  fs::remove(extra / t.manifest.data_name);
  fs::create_hard_link(alias_primary, extra / t.manifest.data_name);
  StripeManifest alias = t.manifest;
  alias.primary_basename = "alias.bin";
  alias.primary_size = 20 * C;
  alias.primary_sha256 = sha256_file(alias_primary.u8string());
  alias.runs = {{0, 20, 0}};
  alias.hashes.clear();
  for (uint64_t c = 0; c < 20; ++c) alias.hashes.push_back(xxh64(t.bytes.data() + c * C, size_t(C)));
  write_file_atomic((extra / stripe_manifest_name(alias.primary_basename)).u8string(), stripe_manifest_write(alias));
  auto o = t.options(); o.primary_sha256 = alias.primary_sha256;
  auto f = MFile::open(alias_primary.u8string(), extra.u8string(), o);
  CHECK(!f->multi());
  CHECK(f->reasons().find("aliases") != std::string::npos);
  Buffer dst{size_t(C)};
  CHECK_EQ(f->pread(dst.p, C, 0), C);
  CHECK(std::memcmp(dst.p, t.bytes.data(), size_t(C)) == 0);
}

TEST(mfile_primary_only_and_sized_counters) {
  Fixture t;
  for (const std::string& dirs : {std::string(), t.primary.parent_path().u8string()}) {
    auto f = MFile::open(t.primary.u8string(), dirs, t.options());
    CHECK(!f->multi());
    CHECK_EQ(f->locations(), 1);
    CHECK_EQ(counters(*f).verified_bytes, uint64_t(0));
    CHECK(t.equal(*f, 0, 8 * C));
    CHECK(t.equal(*f, 39 * C, 4096));
    alignas(ss_mfile_counters) unsigned char bytes[32];
    std::memset(bytes, 0xa5, sizeof bytes);
    auto* out = reinterpret_cast<ss_mfile_counters*>(bytes);
    out->struct_size = 16;
    f->counters(out);
    CHECK_EQ(out->struct_size, uint32_t(16));
    CHECK_EQ(bytes[16], 0xa5);
    Buffer data(8192);
    ss_mfile_io ios[] = {{data.p, C, 4096}, {data.p + 4096, 39 * C, 4096}};
    f->preadv(ios, 2);
    CHECK(std::memcmp(data.p, t.bytes.data() + C, 4096) == 0);
    CHECK(std::memcmp(data.p + 4096, t.bytes.data() + 39 * C, 123) == 0);
  }
  auto o = t.options();
  o.primary_sha256.clear();
  auto hashed = MFile::open(t.primary.u8string(), t.secondary.u8string(), o);
  CHECK(hashed->multi());
  CHECK(t.equal(*hashed, C, C));
}

TEST(mfile_concurrent_reads_and_injection) {
  Fixture t;
  auto f = t.open();
  std::atomic<bool> start{false};
  std::atomic<int> failures{0};
  std::vector<std::thread> threads;
  for (unsigned id = 0; id < 8; ++id) threads.emplace_back([&, id] {
    std::mt19937 rng(id + 1);
    while (!start.load()) std::this_thread::yield();
    try {
      for (int i = 0; i < 80; ++i) {
        uint64_t off = i % 7 == 0 ? uint64_t(rng() % 38) * C : uint64_t(rng() % (38 * C / 4096)) * 4096;
        uint64_t n = i % 7 == 0 ? C : uint64_t(1 + rng() % 16) * 4096;
        if (!t.equal(*f, off, n)) failures.fetch_add(1);
      }
    } catch (...) { failures.fetch_add(1); }
  });
  std::thread injector([&] {
    start.store(true);
    for (int i = 0; i < 100; ++i) {
      f->test_inject(1, 1 + i % 3, uint64_t(i % 4));
      std::this_thread::yield();
    }
  });
  for (auto& thread : threads) thread.join();
  injector.join();
  CHECK_EQ(failures.load(), 0);
  CHECK(t.equal(*f, 0, 8 * C));
}

TEST(mfile_open_validates_options_without_stripes) {
  Fixture t;
  const auto p = t.primary.u8string(), absent = (t.root / "absent").u8string(), secondary = t.secondary.u8string();
  for (const char* dirs : {static_cast<const char*>(nullptr), "", absent.c_str(), secondary.c_str()}) {
    ss_mfile_options o;
    ss_mfile_options_defaults(&o);
    for (uint32_t readers : {0u, 33u, UINT32_MAX}) {
      o.drive_readers = readers;
      ss_mfile* f = nullptr;
      CHECK(ss_mfile_open(p.c_str(), dirs, &o, &f) == SS_E_INVALID);
      CHECK(!f);
    }
    o.drive_readers = 4;
    for (int mode : {-1, int(SS_MFILE_VERIFY_OFF), 3}) {
      o.verify_mode = mode;
      ss_mfile* f = nullptr;
      CHECK(ss_mfile_open(p.c_str(), dirs, &o, &f) == SS_E_INVALID);
      CHECK(!f);
    }
  }
  auto o = t.options();
  for (uint32_t readers : {0u, 33u, UINT32_MAX}) {
    o.drive_readers = readers;
    CHECK_EQ(status([&] { MFile::open(p, "", o); }), int(SS_E_INVALID));
  }
  ss_mfile_options valid;
  ss_mfile_options_defaults(&valid);
  for (uint32_t readers : {1u, 32u}) {
    valid.drive_readers = readers;
    ss_mfile* f = nullptr;
    CHECK_OK(ss_mfile_open(p.c_str(), nullptr, &valid, &f));
    ss_mfile_close(f);
  }
  ss_mfile* f = nullptr;
  CHECK(ss_mfile_open(t.primary.parent_path().u8string().c_str(), nullptr, nullptr, &f) == SS_E_IO);
}

TEST(mfile_pools_shared_across_shards_and_absent_manifests_reused) {
  Fixture t;
  auto a = t.open();
  const auto pools = MFile::test_pools_created();
  std::vector<std::shared_ptr<MFile>> shards;
  for (int i = 0; i < 12; ++i) {
    const auto shard = t.primary.parent_path() / ("shard-" + std::to_string(i) + ".bin");
    fs::create_hard_link(t.primary, shard);
    shards.push_back(MFile::open(shard.u8string(), "", t.options()));
  }
  CHECK_EQ(MFile::test_pools_created(), pools);
  // Concurrent calls share a bounded volume pool and each waits for its own jobs, including fallback.
  a->test_inject(1, 2, 0);
  std::atomic<bool> equal{true};
  std::vector<std::thread> readers;
  for (const auto& f : shards) readers.emplace_back([&, f] { if (!t.equal(*f, 0, 8 * C)) equal.store(false); });
  CHECK(t.equal(*a, 0, 8 * C));
  for (auto& r : readers) r.join();
  CHECK(equal.load());
  CHECK(counters(*a).fallback_bytes > 0);
  auto changed = t.options();
  changed.drive_readers = 3;
  auto other = MFile::open(t.primary.u8string(), t.secondary.u8string(), changed);
  CHECK(other.get() != a.get());
  CHECK_EQ(MFile::test_pools_created(), pools + 1);
  const auto missing = t.root / "missing";
  fs::create_directory(missing);
  const auto dirs = t.secondary.u8string() + ";" + missing.u8string();
  auto first = MFile::open(t.primary.u8string(), dirs, t.options());
  const auto sessions = MFile::test_sessions_created();
  auto second = MFile::open(t.primary.u8string(), dirs, t.options());
  CHECK(first == second);
  CHECK_EQ(MFile::test_sessions_created(), sessions);
  put(missing / stripe_manifest_name(t.manifest.primary_basename), "invalid", 7);
  auto appeared = MFile::open(t.primary.u8string(), dirs, t.options());
  CHECK(appeared != first);
  CHECK_EQ(MFile::test_sessions_created(), sessions + 1);
}

TEST(mfile_tail_zero_and_fallback_learning) {
  Fixture t;
  // Physical padding is excluded from the logical hash, but must never change the caller's EOF tail.
  {
    std::fstream data(t.secondary / t.manifest.data_name, std::ios::binary | std::ios::in | std::ios::out);
    data.seekp(std::streamoff(19 * C + 123));
    data.put(char(0x55));
    CHECK(bool(data));
  }
  auto stripe = t.open();
  auto primary = MFile::open(t.primary.u8string(), "", t.options());
  Buffer dst(4096);
  for (auto& f : {stripe, primary}) {
    std::memset(dst.p, 0x66, 4096);
    CHECK_EQ(f->pread(dst.p, 4096, 39 * C), uint64_t(123));
    CHECK(std::memcmp(dst.p, t.bytes.data() + 39 * C, 123) == 0);
    CHECK(std::all_of(dst.p + 123, dst.p + 4096, [](char c) { return c == 0; }));
    std::memset(dst.p, 0x66, 4096);
    ss_mfile_io io{dst.p, 39 * C, 4096};
    f->preadv(&io, 1);
    CHECK(std::all_of(dst.p + 123, dst.p + 4096, [](char c) { return c == 0; }));
  }
  stripe->test_inject(1, 1, 0);
  CHECK(t.equal(*stripe, 0, 8 * C));
  CHECK(counters(*stripe).fallback_bytes > 0);
  // Every planned secondary piece fell back: its initial throughput estimate must stay unchanged.
  CHECK_EQ(stripe->test_rate(1), 6.0);
  CHECK(stripe->test_rate(0) > 0);
}

#if !defined(_WIN32)
TEST(mfile_rejects_special_files_without_blocking) {
  Fixture t;
  const auto fifo = t.root / "fifo";
  CHECK_EQ(mkfifo(fifo.c_str(), 0600), 0);
  CHECK_EQ(status([&] { RawFile f(fifo.u8string(), false); }), int(SS_E_IO));
  ss_mfile* file = nullptr;
  CHECK(ss_mfile_open(fifo.c_str(), nullptr, nullptr, &file) == SS_E_IO);
  const auto manifest_path = t.secondary / stripe_manifest_name(t.manifest.primary_basename);
  fs::remove(manifest_path);
  CHECK_EQ(mkfifo(manifest_path.c_str(), 0600), 0);
  {
    auto f = t.open();
    CHECK(!f->multi());
    CHECK(t.equal(*f, C, 4096));
  }
  fs::remove(manifest_path);
  t.publish();
  const auto data = t.secondary / t.manifest.data_name;
  fs::remove(data);
  CHECK_EQ(mkfifo(data.c_str(), 0600), 0);
  auto f = t.open();
  CHECK(!f->multi());
  CHECK(t.equal(*f, C, 4096));
}
#endif

int main() { return sstest::run_all(); }
