// CPU source integration with real, verified stripe sessions.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include "check.hpp"
#include "dg_fixture.hpp"
#include "dg_load.hpp"
#include "fixture.hpp"
#include "mfile.hpp"
#include "source_file.hpp"
#include "strix_stream/ss.h"

namespace {
constexpr uint64_t kChunk = uint64_t(1) << 20;

struct SourceFixture : sstest::Fixture {
  explicit SourceFixture(const std::string& tag) : Fixture(tag) {
    // Put the record units in an odd chunk; the stacked tensors remain on the primary.
    auto doc = ts::json_parse(json);
    auto& layer = doc["layers"].arr[2];
    const uint64_t off = ts::json_u64(*layer["parts"].arr[0].find("offset"), "offset") + kChunk;
    layer["parts"].arr[0]["offset"] = ts::Json::number(off);
    write_file(dir / "experts.bin", 0, off + kE2 * kRec + 512);
    layer["hash"] = ts::Json::array();
    for (int u = 0; u < kE2; ++u) {
      auto& bytes = expected[2][size_t(u)];
      bytes = slice(0, off + uint64_t(u) * kRec, kRec);
      layer["hash"].push(ts::Json::string(hex(ts::xxh64(bytes.data(), bytes.size()))));
    }
    json = ts::json_write(doc);
  }
};

struct Run {
  ss_desc* d = nullptr;
  ss_plan* p = nullptr;
  ss_ctx* c = nullptr;
  ~Run() { ss_close(c); ss_plan_free(p); ss_desc_free(d); }
  void open(const std::string& json, const char* dirs, bool direct, uint32_t drive_readers = 3, bool old_options = false) {
    CHECK_OK(ss_desc_from_json(json.data(), json.size(), &d));
    ss_budget b;
    ss_budget_defaults(&b);
    b.device_bytes = uint64_t(128) << 20;
    b.host_pinned_bytes = uint64_t(8) << 20;
    b.margin_gib = 1.0 / 1024;
    b.margin_fraction = 0;
    b.ring_slots = 32;
    const std::string plan = R"({"tiers":["","RPPDDDDDDDDDDDDD","RPDDDDDD","D","RPPDDDDDDDDDDDDD"]})";
    CHECK_OK(ss_plan_from_json(d, &b, plan.data(), plan.size(), &p));
    ss_options o;
    ss_options_defaults(&o);
    o.staging_bytes = 256 * 1024;
    o.staging_buffers = 2;
    o.reader_threads = 4;
    o.verify_hash = 1;
    o.direct_io = direct;
    o.weight_dirs = dirs;
    o.drive_readers = drive_readers;
    o.weight_verify = SS_MFILE_VERIFY_ALWAYS;
    if (old_options) o.struct_size = uint32_t(offsetof(ss_options, weight_dirs));
    CHECK_OK(ss_open(d, &b, p, &o, &c));
  }
};

struct StripeDir {
  std::filesystem::path root, path;
  explicit StripeDir(const std::filesystem::path& dir) : root(dir), path(root / "secondary") {
    std::filesystem::create_directories(path);
  }
  void manifest(const char* name) {
    const auto primary = root / name;
    ts::StripeManifest m;
    m.chunk_bytes = kChunk;
    m.primary_basename = name;
    m.primary_size = std::filesystem::file_size(primary);
    m.primary_sha256 = ts::sha256_file(primary.u8string());
    m.data_name = std::string(name) + ".stripe.data";
    std::ifstream in(primary, std::ios::binary);
    std::ofstream out(path / m.data_name, std::ios::binary);
    std::vector<char> bytes(size_t(kChunk), 0);
    for (uint64_t off = kChunk; off < m.primary_size; off += 2 * kChunk) {
      const auto n = std::min(kChunk, m.primary_size - off);
      std::fill(bytes.begin(), bytes.end(), char(0));
      in.seekg(std::streamoff(off));
      in.read(bytes.data(), std::streamsize(n));
      CHECK(bool(in));
      m.runs.push_back({off / kChunk, 1, m.hashes.size()});
      m.hashes.push_back(ts::xxh64(bytes.data(), size_t(n)));
      out.write(bytes.data(), std::streamsize(kChunk));
    }
    CHECK(bool(out));
    out.close();
    m.data_size = m.hashes.size() * kChunk;
    ts::write_file_atomic((path / ts::stripe_manifest_name(name)).u8string(), ts::stripe_manifest_write(m));
  }
  ~StripeDir() { std::filesystem::remove_all(path); }
};

ss_mfile_counters counters(const ts::MFile& file) {
  ss_mfile_counters c{};
  c.struct_size = sizeof c;
  file.counters(&c);
  return c;
}

std::shared_ptr<ts::MFile> session(const std::filesystem::path& primary, const std::string& dirs, bool direct,
                                   uint32_t readers = 3) {
  ts::MFile::Options o;
  o.direct = direct;
  o.drive_readers = readers;
  o.verify_mode = SS_MFILE_VERIFY_ALWAYS;
  return ts::MFile::open(primary.u8string(), dirs, o);
}

ts::Json stats(ss_ctx* c) {
  char* text = nullptr;
  CHECK_OK(ss_stats_to_json(c, &text, nullptr));
  const auto j = ts::json_parse(text ? text : "{}");
  ss_free(text);
  return j;
}

using Bytes = std::vector<std::vector<std::vector<uint8_t>>>;
Bytes read_all(ss_ctx* c, const sstest::Fixture& fx) {
  Bytes out(size_t(fx.n_layers));
  CHECK_OK(ss_begin(c, nullptr, 0));
  for (int l : {1, 2, 3, 4}) {
    CHECK_OK(ss_layer_all(c, l));
    ss_layer_view v{};
    CHECK_OK(ss_acquire(c, l, nullptr, &v));
    for (size_t u = 0; u < fx.expected[size_t(l)].size(); ++u) {
      const auto& expected = fx.expected[size_t(l)][u];
      CHECK(v.table_host && v.table_host[u]);
      if (!v.table_host || !v.table_host[u]) continue;
      const auto p = reinterpret_cast<const uint8_t*>(uintptr_t(v.table_host[u]));
      out[size_t(l)].emplace_back(p, p + expected.size());
      CHECK(out[size_t(l)].back() == expected);
    }
    CHECK_OK(ss_release(c, l, nullptr));
  }
  CHECK_OK(ss_end(c));
  return out;
}
}

TEST(source_default_never_constructs_session) {
  SourceFixture fx("source-default");
  StripeDir stripe(fx.dir);
  // A manifest in the primary directory must not turn on multi mode.
  const auto primary_manifest = fx.dir / "experts.bin.stripe.json";
  stripe.manifest("experts.bin");
  std::filesystem::copy_file(stripe.path / "experts.bin.stripe.json", primary_manifest,
                             std::filesystem::copy_options::overwrite_existing);
  std::filesystem::remove(stripe.path / "experts.bin.stripe.json");
  const auto before = ts::MFile::test_sessions_created();
  const std::string primary = fx.dir.u8string();
  const std::string aliases = primary + ";" + (fx.dir / ".").u8string() + ";;";
  for (const char* dirs : {static_cast<const char*>(nullptr), "", primary.c_str(), aliases.c_str()}) {
    Run r;
    r.open(fx.json, dirs, false);
    CHECK(r.c);
    if (r.c) {
      CHECK(read_all(r.c, fx) == fx.expected);
      CHECK(stats(r.c).find("source_sessions") == nullptr);
    }
  }
  // A secondary without this source's manifest also keeps the original readers.
  {
    Run r;
    const auto dirs = stripe.path.u8string();
    r.open(fx.json, dirs.c_str(), false);
    CHECK(r.c);
  }
  // An older sized options struct ignores every appended multi-location field.
  stripe.manifest("experts.bin");
  {
    Run r;
    const auto dirs = stripe.path.u8string();
    r.open(fx.json, dirs.c_str(), false, 3, true);
    CHECK(r.c);
  }
  CHECK_EQ(ts::MFile::test_sessions_created(), before);
  std::filesystem::remove(primary_manifest);
}

TEST(source_multi_bytes_alignment_options_and_physical_mode) {
  SourceFixture fx("source-multi");
  StripeDir stripe(fx.dir);
  stripe.manifest("experts.bin");
  stripe.manifest("blob.bin");
  Run plain;
  plain.open(fx.json, nullptr, false);
  CHECK(plain.c);
  if (!plain.c) return;
  const auto expected = read_all(plain.c, fx);
  for (bool direct : {false, true}) {
    const auto before = ts::MFile::test_sessions_created();
    Run multi;
    // Descriptor alignment may be smaller or larger than the uniform session alignment.
    auto doc = ts::json_parse(fx.json);
    doc["sources"].arr[0]["align"] = ts::Json::number(uint64_t(1));
    doc["sources"].arr[1]["align"] = ts::Json::number(direct ? uint64_t(512) : uint64_t(65536));
    const std::string list = fx.dir.u8string() + ";" + stripe.path.u8string() + ";" + stripe.path.u8string();
    std::string dirs = list;
    multi.open(ts::json_write(doc), dirs.c_str(), direct, 7);
    dirs.assign(4096, 'x'); // caller no longer retains the option storage
    CHECK(multi.c);
    if (!multi.c) continue;
    CHECK_EQ(ts::MFile::test_sessions_created() - before, uint64_t(2));
    auto experts = session(fx.dir / "experts.bin", list, direct, 7);
    auto blob = session(fx.dir / "blob.bin", list, direct, 7);
    CHECK_EQ(ts::MFile::test_sessions_created() - before, uint64_t(2));
    CHECK(experts->multi() && blob->multi());
    // the context reads through these shared sessions; serve stripe chunks from the stripe (routing has its own tests)
    experts->test_prefer_stripe(true);
    blob->test_prefer_stripe(true);
    CHECK(read_all(multi.c, fx) == expected);
    for (const auto& file : {experts, blob}) {
      const auto c = counters(*file);
      CHECK_EQ(c.verify_mode, int(SS_MFILE_VERIFY_ALWAYS));
      CHECK(c.verified_bytes > 0);
      CHECK(c.bytes[0] > 0 && c.bytes[1] > 0);
      CHECK_EQ(c.verify_fail, uint64_t(0));
      CHECK_EQ(c.read_errors, uint64_t(0));
      CHECK_EQ(c.disabled_mask, 0u);
      CHECK_EQ(c.direct_mask, direct ? 3u : 0u);
    }
    const bool physical = counters(*experts).direct_mask == 3u;
    CHECK_EQ(ss_source_direct(multi.c, 0), physical ? 1 : 0);
    const auto st = stats(multi.c);
    CHECK_EQ(st.find("sources_direct")->arr[0].b, physical);

  }
}

TEST(source_manifest_selection_and_single_location_fallback) {
  SourceFixture fx("source-selection");
  StripeDir stripe(fx.dir);
  stripe.manifest("experts.bin");
  const auto before = ts::MFile::test_sessions_created();
  Run r;
  const auto dirs = stripe.path.u8string();
  r.open(fx.json, dirs.c_str(), false, 4);
  CHECK(r.c);
  if (!r.c) return;
  CHECK_EQ(ts::MFile::test_sessions_created() - before, uint64_t(1));
  ss_options o;
  ss_options_defaults(&o);
  o.weight_dirs = dirs.c_str();
  o.direct_io = 0;
  o.weight_verify = SS_MFILE_VERIFY_ALWAYS;
  auto experts = ts::open_source((fx.dir / "experts.bin").u8string(), o);
  auto blob = ts::open_source((fx.dir / "blob.bin").u8string(), o);
  CHECK(experts->session() && experts->session()->multi());
  CHECK(!blob->session());
  CHECK_EQ(ts::MFile::test_sessions_created() - before, uint64_t(1));
  CHECK(read_all(r.c, fx) == fx.expected);
  const auto st = stats(r.c);
  CHECK_EQ(st.find("sources_direct")->arr.size(), size_t(2));
}

TEST(source_rejected_stripe_uses_primary) {
  SourceFixture fx("source-rejected");
  StripeDir stripe(fx.dir);
  stripe.manifest("experts.bin");
  std::filesystem::remove(stripe.path / "experts.bin.stripe.data");
  const auto dirs = stripe.path.u8string();
  for (bool direct : {false, true}) {
    Run r;
    r.open(fx.json, dirs.c_str(), direct);
    CHECK(r.c);
    if (!r.c) continue;
    CHECK(read_all(r.c, fx) == fx.expected);
    // Inspect a second adapter: rejected locations may be retried by another opener.
    auto file = session(fx.dir / "experts.bin", dirs, direct);
    CHECK(!file->multi());
    CHECK(!file->reasons().empty());
    CHECK(counters(*file).disabled_mask & 2u);
  }
}

TEST(source_read_error_falls_back_within_shared_session) {
  SourceFixture fx("source-fallback");
  StripeDir stripe(fx.dir);
  stripe.manifest("experts.bin");
  const auto dirs = stripe.path.u8string();
  for (bool direct : {false, true}) {
    // Keep this session alive so the runtime borrows its fault and counters.
    auto file = session(fx.dir / "experts.bin", dirs, direct);
    file->test_prefer_stripe(true);  // small fixture reads would stay on the primary under the size rule
    file->test_inject(1, 1, 0);
    const auto before = ts::MFile::test_sessions_created();
    Run r;
    r.open(fx.json, dirs.c_str(), direct);
    CHECK(r.c);
    if (!r.c) continue;
    CHECK_EQ(ts::MFile::test_sessions_created(), before);
    CHECK(read_all(r.c, fx) == fx.expected);
    const auto c = counters(*file);
    CHECK(c.fallback_bytes > 0);
    CHECK_EQ(c.read_errors, uint64_t(1));
    CHECK_EQ(c.demoted_chunks, uint64_t(1));
    CHECK(c.bytes[0] > 0);
    CHECK(!file->multi());
  }
}

TEST(source_loader_workers_borrow_one_session) {
  sstest::DgModel model("source-load", 2, 128);
  auto d = ts::desc_from_json(ts::json_parse(model.json_hashed()));
  StripeDir stripe(model.dir);
  stripe.manifest("experts.bin");
  // Two descriptor sources naming the same file must share the same session too.
  d.sources.push_back(d.sources[0]);
  d.sources[0].align = 65536;
  d.sources[1].align = 1;
  for (auto& l : d.layers) l.parts[1].source = 1;
  const auto before = ts::MFile::test_sessions_created();
  const auto dirs = stripe.path.u8string();
  ss_options o;
  ss_options_defaults(&o);
  o.weight_dirs = dirs.c_str();
  o.direct_io = 0;
  o.weight_verify = SS_MFILE_VERIFY_ALWAYS;
  std::vector<std::unique_ptr<ts::SourceFile>> sources;
  for (const auto& s : d.sources) sources.push_back(ts::open_source(s.path, o));
  CHECK_EQ(ts::MFile::test_sessions_created() - before, uint64_t(1));
  CHECK(sources[0]->session() == sources[1]->session());
  if (sources[0]->session()) sources[0]->session()->test_prefer_stripe(true);  // small fixture: use the stripe too
  std::vector<ts::DgLoadUnit> units;
  std::vector<std::vector<char>> homes(size_t(model.G * model.U));
  std::vector<std::vector<char>> device(homes.size());
  for (int l = 0; l < model.G; ++l) {
    for (int u = 0; u < model.U; ++u) {
      const auto k = units.size();
      homes[k].resize(size_t(model.unit()) + 13, char(0x55));
      device[k].resize(size_t(model.unit()), char(0x33));
      units.push_back({l, u, u % 2 == 0 ? int32_t(k) : -1, u % 2 == 0 ? nullptr : homes[k].data() + 13});
    }
  }
  const uint64_t capacity = model.unit() * 4;
  std::vector<std::vector<char>> storage(2, std::vector<char>(size_t(capacity) + 8192, char(0x66)));
  std::vector<char*> buffers;
  for (auto& b : storage) buffers.push_back(b.data() + 123);
  const auto st = ts::dg_load_pipeline(d, units, buffers, capacity, 8, false, true, [](size_t) {},
      [&](size_t, const std::vector<ts::DgLoadCopy>& copies) {
        for (const auto& c : copies) {
          uint64_t off = 0;
          for (size_t p = 0; p < c.part; ++p) off += d.layers[0].parts[p].bytes;
          std::memcpy(device[size_t(c.slot)].data() + off, c.data, size_t(c.bytes));
        }
      }, &sources);
  CHECK_EQ(ts::MFile::test_sessions_created() - before, uint64_t(1));
  const auto c = counters(*sources[0]->session());
  CHECK(c.bytes[0] > 0 && c.bytes[1] > 0);
  CHECK_EQ(c.verify_mode, int(SS_MFILE_VERIFY_ALWAYS));
  CHECK_EQ(c.verify_fail, uint64_t(0));
  CHECK_EQ(c.read_errors, uint64_t(0));
  CHECK_EQ(st.payload_bytes, uint64_t(units.size()) * model.unit());
  for (size_t k = 0; k < units.size(); ++k) {
    const auto expected = model.bytes(units[k].layer, units[k].unit);
    const auto actual = units[k].slot >= 0 ? device[k].data() : units[k].host;
    CHECK(std::memcmp(actual, expected.data(), expected.size()) == 0);
    for (size_t i = 0; i < 13; ++i) CHECK_EQ(homes[k][i], char(0x55));
  }
  for (const auto& b : storage) {
    for (size_t i = 0; i < 123; ++i) CHECK_EQ(b[i], char(0x66));
    for (size_t i = size_t(capacity) + 123; i < b.size(); ++i) CHECK_EQ(b[i], char(0x66));
  }
}

TEST(source_options_sized_defaults) {
  ss_options o;
  ss_options_defaults(&o);
  CHECK(o.weight_dirs == nullptr);
  CHECK_EQ(o.drive_readers, 4u);
  CHECK_EQ(o.weight_verify, int(SS_MFILE_VERIFY_CACHED));
  // An old caller's guard bytes and a future caller's trailing bytes stay untouched.
  alignas(ss_options) unsigned char storage[sizeof(ss_options) + 16];
  for (size_t size : {offsetof(ss_options, weight_dirs), sizeof(ss_options) + 16}) {
    std::memset(storage, 0xa5, sizeof storage);
    ss_options_defaults_sized(reinterpret_cast<ss_options*>(storage), uint32_t(size));
    const size_t wrote = std::min(size, sizeof(ss_options));
    for (size_t i = wrote; i < sizeof storage; ++i) CHECK_EQ(storage[i], 0xa5);
    CHECK_EQ(reinterpret_cast<ss_options*>(storage)->struct_size, uint32_t(wrote));
  }
}

TEST(source_open_validates_weight_options_without_manifests) {
  SourceFixture fx("source-options");
  StripeDir stripe(fx.dir);
  ss_desc* d = nullptr;
  CHECK_OK(ss_desc_from_json(fx.json.data(), fx.json.size(), &d));
  ss_budget b;
  ss_budget_defaults(&b);
  b.device_bytes = uint64_t(128) << 20;
  b.margin_gib = 0;
  const auto dirs = stripe.path.u8string();
  for (const char* list : {static_cast<const char*>(nullptr), "", dirs.c_str()}) {
    ss_options o;
    ss_options_defaults(&o);
    o.weight_dirs = list;
    for (uint32_t readers : {0u, 33u, UINT32_MAX}) {
      o.drive_readers = readers;
      ss_ctx* c = nullptr;
      CHECK(ss_open(d, &b, nullptr, &o, &c) == SS_E_INVALID);
      CHECK(!c);
    }
    o.drive_readers = 4;
    for (int mode : {-1, int(SS_MFILE_VERIFY_OFF), 3}) {
      o.weight_verify = mode;
      ss_ctx* c = nullptr;
      CHECK(ss_open(d, &b, nullptr, &o, &c) == SS_E_INVALID);
      CHECK(!c);
    }
  }
  ss_desc_free(d);
}

int main(int argc, char** argv) { return sstest::run_all(argc > 1 ? argv[1] : nullptr); }
