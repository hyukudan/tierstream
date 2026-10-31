// ts-read-bench: every read pattern and submission mode delivers the exact bytes (--check PASS) from a primary alone
// and from a primary plus a stripe location; the check itself detects a wrong byte; the built executable runs.
#define TS_VRAM_BENCH_TESTING
#include "../tools/ts_vram_bench.cpp"

#include "check.hpp"
#include "errors.hpp"
#include "raw_file.hpp"
#include "sha256.hpp"
#include "stripe_manifest.hpp"
#include "xxh64.hpp"

namespace {

constexpr uint64_t C = kMiB;  // the stripe chunk
std::string g_exe;

void env(const char* name, const std::string& value) {
#if defined(_WIN32)
  _putenv_s(name, value.c_str());
#else
  if (value.empty()) unsetenv(name);
  else setenv(name, value.c_str(), 1);
#endif
}

void put(const fs::path& p, const void* data, size_t n) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(static_cast<const char*>(data), std::streamsize(n));
  ts::require(bool(f), SS_E_IO, "test file write");
}

// A primary of 39 MiB + 123 bytes (an unaligned end) and a secondary directory holding its odd chunks.
struct Fixture {
  fs::path root, primary, secondary, cache;
  std::string old_cache, sha;
  std::vector<char> bytes;
#if defined(_WIN32)
  const char* cache_env = "LOCALAPPDATA";
#else
  const char* cache_env = "XDG_CACHE_HOME";
#endif
  Fixture() {
    root = fs::current_path() /
           ("read-bench-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    primary = root / "primary" / "weights.bin";
    secondary = root / "secondary";
    cache = root / "cache";
    fs::create_directories(primary.parent_path());
    fs::create_directories(secondary);
    fs::create_directories(cache);
    if (const char* p = std::getenv(cache_env)) old_cache = p;
    env(cache_env, cache.u8string());
    bytes.resize(size_t(39 * C + 123));
    uint32_t r = 29;
    for (size_t i = 0; i < bytes.size(); ++i) {
      r ^= r << 13; r ^= r >> 17; r ^= r << 5;
      bytes[i] = char(r);
    }
    put(primary, bytes.data(), bytes.size());
    ts::StripeManifest m;
    m.chunk_bytes = C;
    m.primary_basename = "weights.bin";
    m.primary_size = bytes.size();
    m.primary_sha256 = sha = ts::sha256_file(primary.u8string());
    m.data_name = "stripe.data";
    std::ofstream data(secondary / m.data_name, std::ios::binary);
    std::vector<char> slot(size_t(C), 0);
    for (uint64_t chunk = 1; chunk < 40; chunk += 2) {
      const uint64_t n = std::min(C, uint64_t(bytes.size()) - chunk * C);
      std::fill(slot.begin(), slot.end(), char(0));
      std::memcpy(slot.data(), bytes.data() + chunk * C, size_t(n));
      m.runs.push_back({chunk, 1, m.hashes.size()});
      m.hashes.push_back(ts::xxh64(slot.data(), size_t(n)));
      data.write(slot.data(), std::streamsize(C));
    }
    data.close();
    m.data_size = m.hashes.size() * C;
    ts::write_file_atomic((secondary / ts::stripe_manifest_name("weights.bin")).u8string(),
                          ts::stripe_manifest_write(m));
  }
  ~Fixture() {
    env(cache_env, old_cache);
    std::error_code ec;
    fs::remove_all(root, ec);
  }
};

Fixture& fx() {
  static Fixture f;
  return f;
}

std::string plain_text(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// Runs the tool in process; returns the exit code and the parsed --json result.
int bench(std::vector<std::string> args, ts::Json* out) {
  const fs::path json = fx().root / "result.json";
  std::error_code ec;
  fs::remove(json, ec);
  args.insert(args.begin(), {"ts-read-bench", "--file", fx().primary.u8string(), "--json", json.u8string()});
  std::vector<char*> argv;
  for (auto& a : args) argv.push_back(a.data());
  const int rc = bench_main(int(argv.size()), argv.data());
  if (out && fs::exists(json)) *out = ts::json_parse(plain_text(json));
  return rc;
}

std::string check_result(const ts::Json& j) {
  const ts::Json* c = j.find("check");
  const ts::Json* r = c ? c->find("result") : nullptr;
  return r ? r->text : "";
}

uint64_t loc_bytes(const ts::Json& j, size_t loc) {
  const ts::Json* c = j.find("counters");
  const ts::Json* l = c ? c->find("locations") : nullptr;
  if (!l || l->arr.size() <= loc) return 0;
  return ts::json_u64(*l->arr[loc].find("bytes"), "bytes");
}

TEST(read_bench_primary_patterns) {
  for (const std::vector<std::string>& extra : std::vector<std::vector<std::string>>{
           {"--pattern", "seq", "--unit-bytes", "4194304", "--units", "16", "--depth", "4"},
           {"--pattern", "seq", "--unit-bytes", "4194304", "--units", "16", "--depth", "4", "--batch"},
           {"--pattern", "random", "--unit-bytes", "1048576", "--units", "24", "--depth", "3", "--seed", "7"},
           {"--pattern", "seq", "--unit-bytes", "4194304", "--units", "10", "--depth", "2", "--buffered"}}) {
    std::vector<std::string> args = extra;
    args.insert(args.end(), {"--check", "5"});
    ts::Json j;
    CHECK_EQ(bench(args, &j), 0);
    CHECK_EQ(check_result(j), std::string("PASS"));
    CHECK(j.find("counters")->find("locations")->arr.size() == 1);
    CHECK(ts::json_u64(*j.find("bytes"), "bytes") > 0);
  }
}

TEST(read_bench_ranges_to_eof) {
  // unaligned ranges, the last one ending at the primary's unaligned end
  const uint64_t size = fx().bytes.size();
  const fs::path ranges = fx().root / "ranges.json";
  const std::string text = "[[0, 4096], [5000, 70000], [" + std::to_string(size - 3 * C) + ", " +
                           std::to_string(3 * C) + "], [" + std::to_string(size - 10) + ", 10]]";
  put(ranges, text.data(), text.size());
  ts::Json j;
  CHECK_EQ(bench({"--pattern", "ranges:" + ranges.u8string(), "--depth", "2", "--check", "4"}, &j), 0);
  CHECK_EQ(check_result(j), std::string("PASS"));
  CHECK_EQ(ts::json_u64(*j.find("units"), "units"), uint64_t(4));
  const std::string bad = "[[0, " + std::to_string(size + 1) + "]]";
  put(ranges, bad.data(), bad.size());
  CHECK_EQ(bench({"--pattern", "ranges:" + ranges.u8string()}, nullptr), 1);
}

TEST(read_bench_stripe) {
  uint64_t secondary = 0;
  for (const std::vector<std::string>& extra : std::vector<std::vector<std::string>>{
           {"--verify", "always", "--depth", "2"},
           {"--verify", "cached", "--depth", "3", "--batch"},
           {"--verify", "cached", "--depth", "2", "--pattern", "random", "--seed", "3"}}) {
    std::vector<std::string> args = extra;
    args.insert(args.end(), {"--weight-dirs", fx().secondary.u8string(), "--unit-bytes", "8388608", "--units", "12",
                             "--check", "6", "--primary-sha256", fx().sha});
    ts::Json j;
    CHECK_EQ(bench(args, &j), 0);
    CHECK_EQ(check_result(j), std::string("PASS"));
    CHECK(j.find("counters")->find("multi")->b);
    CHECK_EQ(j.find("counters")->find("locations")->arr.size(), size_t(2));
    CHECK_EQ(ts::json_u64(*j.find("counters")->find("verify_fail"), "verify_fail"), uint64_t(0));
    secondary += loc_bytes(j, 1);
  }
  CHECK(secondary > 0);  // the stripe served part of the large units
}

TEST(read_bench_check_detects_mismatch) {
  std::vector<char> got(fx().bytes.begin() + 8192, fx().bytes.begin() + 8192 + 65536);
  CHECK(same_bytes(fx().primary.u8string(), 8192, got.data(), got.size()));
  got[777] ^= 1;
  CHECK(!same_bytes(fx().primary.u8string(), 8192, got.data(), got.size()));
  // past the end of the file the reference is zero-filled, as ss_mfile reads are
  const uint64_t tail = fx().bytes.size() / kAlign * kAlign;
  std::vector<char> end(size_t(kAlign), 0);
  std::memcpy(end.data(), fx().bytes.data() + tail, size_t(fx().bytes.size() - tail));
  CHECK(same_bytes(fx().primary.u8string(), tail, end.data(), end.size()));
}

TEST(read_bench_usage) {
  CHECK_EQ(bench({"--mode", "bogus"}, nullptr), 2);
  CHECK_EQ(bench({"--unit-bytes", "1000"}, nullptr), 2);
  CHECK_EQ(bench({"--depth", "0"}, nullptr), 2);
#if !defined(TS_VRAM_CUDA)
  CHECK_EQ(bench({"--mode", "read+h2d"}, nullptr), 2);
#endif
}

TEST(read_bench_executable) {
  if (g_exe.empty()) return;
  const fs::path json = fx().root / "exe.json";
  std::string cmd = "\"" + g_exe + "\" --file \"" + fx().primary.u8string() + "\" --weight-dirs \"" +
                    fx().secondary.u8string() + "\" --unit-bytes 8388608 --units 8 --depth 4 --check 4 --json \"" +
                    json.u8string() + "\"";
#if defined(_WIN32)
  cmd = "\"" + cmd + "\"";  // cmd /c strips one outer pair of quotes
#endif
  CHECK_EQ(std::system(cmd.c_str()), 0);
  CHECK_EQ(check_result(ts::json_parse(plain_text(json))), std::string("PASS"));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) g_exe = argv[1];
  const int failures = sstest::run_all();
  return failures ? 1 : 0;
}
