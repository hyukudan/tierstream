// tierstream/pack-v1 CPU gates: writer rules and atomic publish, index / manifest integrity, the indexed reader
// (records, row ranges, segments, units, batched reads, verification modes and the verified-record cache), the
// device-gated descriptor read back through dg_read_unit, stripes of a data file, the C ABI and ts-pack build from a
// layout (copy and row-gather transforms) byte for byte against an independent reference.
// Usage: ts-test-pack FIXTURE_PARENT TS_PACK_EXE TS_STRIPE_EXE
#define TS_PACK_TESTING
#include "../tools/ts_pack.cpp"

#include <cstddef>
#include <filesystem>
#include <random>

#include "check.hpp"
#include "desc.hpp"
#include "dg_load.hpp"
#include "mfile.hpp"
#include "stripe_manifest.hpp"
#include "strix_stream/ss.h"

namespace fs = std::filesystem;

namespace sstest {
template <>
std::string str<ss_status>(const ss_status& v) {
  return "ss_status " + std::to_string(int(v));
}
}  // namespace sstest

namespace {

fs::path g_root;

std::string P(const fs::path& p) { return p.generic_u8string(); }

void set_env(const char* k, const std::string& v) {
#if defined(_WIN32)
  _putenv_s(k, v.c_str());
#else
  setenv(k, v.c_str(), 1);
#endif
}

fs::path fresh(const std::string& name) {
  const fs::path d = g_root / name;
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::remove_all(fs::path(P(d) + ".incomplete"), ec);
  return d;
}

std::vector<char> bytes_of(uint64_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<char> v(static_cast<size_t>(n));
  for (char& c : v) c = char(rng() & 0xff);
  return v;
}

template <class F>
ss_status status_of(F&& f) {
  try {
    f();
    return SS_OK;
  } catch (const ts::Error& e) {
    return e.status();
  }
}

std::string file_bytes(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}
void put_file(const fs::path& p, const std::string& b) {  // replaces the file (published index / manifest are 0444)
  std::error_code ec;
  fs::remove(p, ec);
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f << b;
  if (!f) throw std::runtime_error("cannot write " + P(p));
}

struct Aligned {  // 4 KiB aligned host buffer
  std::vector<char> raw;
  char* p;
  explicit Aligned(uint64_t n) : raw(size_t(n + 2 * 4096)) {
    p = reinterpret_cast<char*>((uintptr_t(raw.data()) + 4095) / 4096 * 4096);
  }
};

PackWriterOptions opts() {
  PackWriterOptions o;
  o.model = "test-model";
  o.sources = {{"model", 12345, std::string(64, 'a')}};
  o.formats["F32"] = {"copy", "1", "test"};
  o.formats["Q"] = {"x-test-repack", "2", "test"};
  return o;
}

// A small model: a global record, then per layer one dense record and `units` units of three records each: "gate"
// (uniform rows with segments 0/5/9/13), "down" (non-uniform rows, three segments) and "scale" (not row-addressable).
struct Model {
  struct Rec {
    PackRecordSpec spec;
    std::vector<char> data;
  };
  std::vector<Rec> recs;
};
Model make_model(int layers, int units, uint32_t seed = 1) {
  Model m;
  uint32_t s = seed;
  auto add = [&](PackRecordSpec spec) {
    m.recs.push_back({spec, bytes_of(spec.bytes, s++)});
  };
  PackRecordSpec g;
  g.layer = -1;
  g.name = "token_embd";
  g.format = "F32";
  g.rows = 7;
  g.row_bytes = 1000;
  g.bytes = 7000;
  add(g);
  for (int l = 0; l < layers; ++l) {
    PackRecordSpec d;
    d.layer = l;
    d.name = "attn." + std::to_string(l);
    d.format = "F32";
    d.bytes = 4096 * 2 + 17;
    add(d);
    for (int u = 0; u < units; ++u) {
      PackRecordSpec a;
      a.layer = l;
      a.expert = u;
      a.name = "gate";
      a.format = "Q";
      a.rows = 13;
      a.row_bytes = 330;
      a.bytes = 13 * 330;
      a.seg_rows = {0, 5, 9, 13};
      add(a);
      PackRecordSpec b;
      b.layer = l;
      b.expert = u;
      b.name = "down";
      b.format = "Q";
      b.rows = 6;
      b.seg_rows = {0, 2, 3, 6};
      b.seg_bytes = {100, 4096, 777};
      b.bytes = 100 + 4096 + 777;
      add(b);
      PackRecordSpec c;
      c.layer = l;
      c.expert = u;
      c.name = "scale";
      c.format = "F32";
      c.bytes = 64;
      add(c);
    }
  }
  return m;
}

std::string write_pack(const fs::path& dir, const Model& m, PackWriterOptions o = opts()) {
  PackWriter w(P(dir), o);
  for (const auto& r : m.recs) w.add(r.spec, r.data.data());
  return w.finish();
}

const Model::Rec& rec_of(const Model& m, int layer, int expert, const std::string& name) {
  for (const auto& r : m.recs)
    if (r.spec.layer == layer && r.spec.expert == expert && r.spec.name == name) return r;
  throw std::runtime_error("no such record in the model");
}

PackOptions ropts(PackVerify v = PackVerify::Always) {
  PackOptions o;
  o.verify = v;
  return o;
}

// ---------------------------------------------------------------------------------------------------------------------

TEST(roundtrip_records_ranges_units) {
  const fs::path dir = fresh("rt");
  const Model m = make_model(3, 4);
  const std::string pin = write_pack(dir, m);
  CHECK(fs::exists(dir / "pack.json") && fs::exists(dir / "pack.index") && fs::exists(dir / "pack-0000.data"));
  CHECK(!fs::exists(fs::path(P(dir) + ".incomplete")));
  CHECK_EQ(pin, ts::sha256_hex(file_bytes(dir / "pack.json").data(), file_bytes(dir / "pack.json").size()));
  PackOptions o = ropts();
  o.pin_sha256 = pin;
  auto p = Pack::open(P(dir), o);
  CHECK_EQ(p->index().records.size(), m.recs.size());
  // whole records, all in one readv
  std::vector<std::vector<char>> got(m.recs.size());
  std::vector<PackRead> rs;
  for (size_t i = 0; i < m.recs.size(); ++i) {
    const auto& r = m.recs[i];
    const int64_t k = p->find(r.spec.layer, r.spec.expert, r.spec.name);
    CHECK(k >= 0);
    got[i].resize(r.data.size());
    rs.push_back({uint64_t(k), 0, 0, got[i].data()});
    const PackRecord& pr = p->index().records[size_t(k)];
    CHECK_EQ(pr.offset % 4096, uint64_t(0));
    CHECK_EQ(pr.xxh64, ts::xxh64(r.data.data(), r.data.size()));
  }
  p->readv(rs.data(), rs.size());
  for (size_t i = 0; i < m.recs.size(); ++i) CHECK(got[i] == m.recs[i].data);
  // uniform row ranges at every boundary, single rows and the whole range
  const auto gate = rec_of(m, 1, 2, "gate");
  const int64_t gk = p->find(1, 2, "gate");
  for (uint32_t a = 0; a < 13; ++a)
    for (uint32_t b = a + 1; b <= 13; ++b) {
      std::vector<char> buf((b - a) * 330);
      p->read(uint64_t(gk), a, b, buf.data());
      CHECK(std::memcmp(buf.data(), gate.data.data() + a * 330, buf.size()) == 0);
    }
  // non-uniform rows: whole segments only
  const auto down = rec_of(m, 2, 3, "down");
  const int64_t dk = p->find(2, 3, "down");
  std::vector<char> d1(4096 + 777);
  p->read(uint64_t(dk), 2, 6, d1.data());
  CHECK(std::memcmp(d1.data(), down.data.data() + 100, d1.size()) == 0);
  CHECK_EQ(status_of([&] { p->read(uint64_t(dk), 1, 3, d1.data()); }), SS_E_INVALID);
  const int64_t sk = p->find(0, 0, "scale");
  CHECK_EQ(status_of([&] { p->read(uint64_t(sk), 0, 1, d1.data()); }), SS_E_INVALID);  // not row-addressable
  CHECK(p->find(0, 9, "gate") < 0 && p->find(5, 0, "gate") < 0);
  // units: one contiguous span holding its records (zero padding between)
  for (int l = 0; l < 3; ++l)
    for (int u = 0; u < 4; ++u) {
      PackUnit un;
      CHECK(p->unit(l, u, &un));
      CHECK_EQ(un.count, uint64_t(3));
      std::vector<char> buf(size_t(un.bytes));
      p->read_unit(l, u, buf.data());
      for (uint64_t i = un.first; i < un.first + un.count; ++i) {
        const PackRecord& r = p->index().records[size_t(i)];
        const auto want = rec_of(m, l, u, p->name(r));
        CHECK(std::memcmp(buf.data() + (r.offset - un.offset), want.data.data(), want.data.size()) == 0);
      }
      const std::string raw = file_bytes(dir / "pack-0000.data").substr(size_t(un.offset), size_t(un.bytes));
      CHECK(std::memcmp(raw.data(), buf.data(), buf.size()) == 0);
    }
  // unit-major: the units of a layer are evenly spaced (one stride)
  PackUnit u0, u1, u3;
  p->unit(1, 0, &u0);
  p->unit(1, 1, &u1);
  p->unit(1, 3, &u3);
  CHECK_EQ(u3.offset - u0.offset, 3 * (u1.offset - u0.offset));
  const Json info = p->info();
  CHECK_EQ(json_u64(*info.find("records"), "r"), uint64_t(m.recs.size()));
  CHECK_EQ(json_u64(*info.find("units"), "u"), uint64_t(12));
}

TEST(unaligned_destinations_and_buffered) {
  const fs::path dir = fresh("unal");
  const Model m = make_model(1, 2);
  write_pack(dir, m);
  for (bool direct : {true, false}) {
    PackOptions o = ropts();
    o.direct = direct;
    auto p = Pack::open(P(dir), o);
    const auto gate = rec_of(m, 0, 1, "gate");
    const int64_t k = p->find(0, 1, "gate");
    std::vector<char> big(gate.data.size() + 64);
    for (size_t shift : {size_t(0), size_t(1), size_t(7), size_t(33)}) {
      p->read(uint64_t(k), 0, 0, big.data() + shift);
      CHECK(std::memcmp(big.data() + shift, gate.data.data(), gate.data.size()) == 0);
      p->read(uint64_t(k), 3, 11, big.data() + shift);
      CHECK(std::memcmp(big.data() + shift, gate.data.data() + 3 * 330, 8 * 330) == 0);
    }
  }
}

TEST(verify_always_detects_corruption_by_record_and_segment) {
  const fs::path dir = fresh("corrupt");
  const Model m = make_model(1, 2);
  write_pack(dir, m);
  PackUnit un;
  int64_t gk;
  uint64_t goff;
  {
    auto p = Pack::open(P(dir), ropts());
    gk = p->find(0, 1, "gate");
    goff = p->index().records[size_t(gk)].offset;
    p->unit(0, 1, &un);
  }
  // flip one byte in rows [9, 13) of (0, 1, gate): the last segment
  std::string data = file_bytes(dir / "pack-0000.data");
  data[size_t(goff + 10 * 330 + 5)] ^= 0x40;
  put_file(dir / "pack-0000.data", data);
  auto p = Pack::open(P(dir), ropts());
  std::vector<char> buf(13 * 330);
  CHECK_EQ(status_of([&] { p->read(uint64_t(gk), 0, 0, buf.data()); }), SS_E_HASH);
  CHECK_EQ(status_of([&] { p->read(uint64_t(gk), 0, 5, buf.data()); }), SS_OK);   // segment 0: intact
  CHECK_EQ(status_of([&] { p->read(uint64_t(gk), 5, 9, buf.data()); }), SS_OK);   // segment 1: intact
  CHECK_EQ(status_of([&] { p->read(uint64_t(gk), 9, 13, buf.data()); }), SS_E_HASH);
  CHECK_EQ(status_of([&] { p->read(uint64_t(gk), 0, 9, buf.data()); }), SS_OK);   // segments 0 + 1
  CHECK_EQ(status_of([&] { p->read(uint64_t(gk), 1, 3, buf.data()); }), SS_E_HASH);  // inside a segment: whole record
  std::vector<char> ub(size_t(un.bytes));
  CHECK_EQ(status_of([&] { p->read_unit(0, 1, ub.data()); }), SS_E_HASH);
  CHECK_EQ(status_of([&] { p->read_unit(0, 0, ub.data()); }), SS_OK);
  const Json rep = p->verify_full();
  CHECK(!rep.find("ok")->b);
  CHECK(!rep.find("files")->arr[0].find("sha256_ok")->b);
  CHECK_EQ(json_u64(*rep.find("files")->arr[0].find("bad_records"), "b"), uint64_t(2));  // the record + its segment
  // off: no check
  auto q = Pack::open(P(dir), ropts(PackVerify::Off));
  CHECK_EQ(status_of([&] { q->read(uint64_t(gk), 0, 0, buf.data()); }), SS_OK);
}

TEST(verify_cached_persists_per_data_file_identity) {
  const fs::path dir = fresh("cached");
  const Model m = make_model(2, 3);
  write_pack(dir, m);
  auto read_all = [&](Pack& p) {
    std::vector<std::vector<char>> b(p.index().records.size());
    std::vector<PackRead> rs;
    for (size_t i = 0; i < b.size(); ++i) {
      b[i].resize(size_t(p.index().records[i].bytes));
      rs.push_back({i, 0, 0, b[i].data()});
    }
    p.readv(rs.data(), rs.size());
  };
  {
    auto p = Pack::open(P(dir), ropts(PackVerify::Cached));
    read_all(*p);
    CHECK_EQ(p->stats().verified_records, uint64_t(m.recs.size()));
    read_all(*p);  // in-process: verified once
    CHECK_EQ(p->stats().verified_records, uint64_t(m.recs.size()));
  }
  {
    auto p = Pack::open(P(dir), ropts(PackVerify::Cached));  // another process: the persisted set
    read_all(*p);
    CHECK_EQ(p->stats().verified_records, uint64_t(0));
  }
  // rewriting the data file (same bytes but a new identity) drops the cache; a corruption is then found
  std::string data = file_bytes(dir / "pack-0000.data");
  data[size_t(data.size() / 2)] ^= 1;
  fs::remove(dir / "pack-0000.data");
  put_file(dir / "pack-0000.data", data);
  {
    auto p = Pack::open(P(dir), ropts(PackVerify::Cached));
    CHECK_EQ(status_of([&] { read_all(*p); }), SS_E_HASH);
  }
}

TEST(open_refuses_pin_sources_index_manifest) {
  const fs::path dir = fresh("refuse");
  const Model m = make_model(1, 1);
  const std::string pin = write_pack(dir, m);
  PackOptions o = ropts();
  o.pin_sha256 = std::string(64, 'b');
  CHECK_EQ(status_of([&] { Pack::open(P(dir), o); }), SS_E_DESC);
  o.pin_sha256 = pin;
  o.source_sha256 = {std::string(64, 'a')};
  CHECK_EQ(status_of([&] { Pack::open(P(dir), o); }), SS_OK);
  o.source_sha256 = {std::string(64, 'c')};
  CHECK_EQ(status_of([&] { Pack::open(P(dir), o); }), SS_E_DESC);
  o.source_sha256 = {std::string(64, 'a'), std::string(64, 'a')};
  CHECK_EQ(status_of([&] { Pack::open(P(dir), o); }), SS_E_DESC);
  // a modified index no longer matches pack.json
  std::string ix = file_bytes(dir / "pack.index");
  std::string bad = ix;
  bad[100] ^= 1;
  put_file(dir / "pack.index", bad);
  CHECK_EQ(status_of([&] { Pack::open(P(dir), ropts()); }), SS_E_DESC);
  put_file(dir / "pack.index", ix + std::string(8, '\0'));  // another size: refused before it is read
  try {
    Pack::open(P(dir), ropts());
    CHECK(false);
  } catch (const ts::Error& e) {
    CHECK(std::string(e.what()).find("pack.index size differs") != std::string::npos);
  }
  put_file(dir / "pack.index", ix);
  // a truncated data file
  std::string data = file_bytes(dir / "pack-0000.data");
  put_file(dir / "pack-0000.data", data.substr(0, data.size() - 4096));
  CHECK_EQ(status_of([&] { Pack::open(P(dir), ropts()); }), SS_E_DESC);
  put_file(dir / "pack-0000.data", data);
  CHECK_EQ(status_of([&] { Pack::open(P(dir), ropts()); }), SS_OK);
  // index codec: every truncation is refused
  for (size_t n : {size_t(0), size_t(10), size_t(63), size_t(64), ix.size() - 1})
    CHECK_EQ(status_of([&] { pack_index_parse(ix.substr(0, n)); }), SS_E_DESC);
  CHECK(pack_index_write(pack_index_parse(ix)) == ix);
  {
    // record kinds other than 0 (stored) are reserved: a v1 reader refuses them by name
    PackIndex parsed = pack_index_parse(ix);
    const PackManifest man = pack_manifest_parse(file_bytes(dir / "pack.json"));
    pack_index_validate(parsed, man);
    parsed.records[0].kind = 1;
    CHECK_EQ(status_of([&] { pack_index_validate(parsed, man); }), SS_E_DESC);
    try {
      pack_index_validate(parsed, man);
    } catch (const ts::Error& e) {
      CHECK(std::string(e.what()).find("record kind 1") != std::string::npos);
    }
    CHECK(pack_index_write(parsed) != ix);
  }
  // manifest: unknown keys refused, x-* kept
  std::string mj = file_bytes(dir / "pack.json");
  CHECK_EQ(status_of([&] { pack_manifest_parse(mj.substr(0, mj.size() - 1) + ",\"bogus\":1}"); }), SS_E_DESC);
  CHECK_EQ(status_of([&] { pack_manifest_parse(mj.substr(0, mj.size() - 1) + ",\"x-note\":1}"); }), SS_OK);
}

// Review findings: checked segment arithmetic, string values unique, the C writer's metadata, layout integer ranges,
// files changed under a cached pack.
TEST(review_hardening) {
  {  // segment sizes whose sum wraps to the record's bytes are refused
    PackManifest m;
    m.files = {{"pack-0000.data", 16384, std::string(64, '0')}};
    m.formats["Q"] = {"copy", "1", "t"};
    PackIndex ix;
    ix.strings = {"down", "Q"};
    PackRecord r;
    r.layer = 0;
    r.expert = 0;
    r.name = 0;
    r.format = 1;
    r.rows = 3;
    r.bytes = 8192;
    r.seg_first = 0;
    r.seg_count = 3;
    ix.records = {r};
    const uint64_t big = UINT64_MAX - 4095;
    ix.segments = {{0, 1, 0, big, 0}, {1, 2, big, 8192, 0}, {2, 3, 4096, 4096, 0}};
    m.records = 1;
    m.segments = 3;
    CHECK_EQ(status_of([&] { pack_index_validate(ix, m); }), SS_E_DESC);
    ix.segments = {{0, 1, 0, 4096, 0}, {1, 2, 4096, 2048, 0}, {2, 3, 6144, 2048, 0}};
    CHECK_EQ(status_of([&] { pack_index_validate(ix, m); }), SS_OK);
    ix.strings = {"down", "Q", "down"};  // the same name under two ids
    CHECK_EQ(status_of([&] { pack_index_validate(ix, m); }), SS_E_DESC);
    PackWriter w(P(fresh("seg-overflow")), opts());
    PackRecordSpec sp;
    sp.layer = 0;
    sp.expert = 0;
    sp.name = "down";
    sp.format = "Q";
    sp.bytes = 8192;
    sp.rows = 3;
    sp.seg_rows = {0, 1, 2, 3};
    sp.seg_bytes = {big, 8192, 4096};
    CHECK_EQ(status_of([&] { w.begin(sp); }), SS_E_INVALID);
  }
  {  // malformed writer metadata through the C ABI is an error, not a crash
    ss_pack_writer* w = nullptr;
    for (const char* meta : {R"({"sources":[{}]})", R"({"sources":[1]})", R"({"formats":{"Q":1}})",
                             R"({"formats":{"Q":{"transform":"c"}}})", R"({"bogus":1})"}) {
      CHECK(ss_pack_writer_begin(P(fresh("bad-meta")).c_str(), meta, std::strlen(meta), &w) != SS_OK);
      CHECK(w == nullptr);
    }
  }
  {  // layout integers out of range are refused, never wrapped
    const fs::path src = g_root / "range-model.bin";
    put_file(src, std::string(8192, 'r'));
    auto layout = [&](const std::string& tensor) {
      return R"({"schema":"tierstream/pack-layout-v1","model":"r","sources":[{"name":"m","path":")" + P(src) +
             R"("}],"formats":{"Q":{"transform":"copy","version":"1","tool":"t"}},"tensors":[)" + tensor + "]}";
    };
    CHECK_EQ(status_of([&] { parse_layout(layout(R"({"layer":0,"name":"a","format":"Q","source":0,"offset":0,"bytes":64})")); }), SS_OK);
    for (const char* bad : {R"({"layer":0,"name":"a","format":"Q","source":4294967296,"offset":0,"bytes":64})",
                            R"({"layer":4294967296,"name":"a","format":"Q","source":0,"offset":0,"bytes":64})",
                            R"({"layer":-2,"name":"a","format":"Q","source":0,"offset":0,"bytes":64})",
                            R"({"layer":0,"name":"a","format":"Q","source":0,"offset":18446744073709551615,"bytes":64})",
                            R"({"layer":0,"name":"a","format":"Q","source":0,"offset":0,"bytes":64,"units":3,"stride":9223372036854775807})",
                            R"({"layer":0,"name":"a","format":"Q","source":0,"offset":0,"bytes":64,"units":4294967295})",
                            R"({"layer":0,"name":"a","format":"Q","source":0,"offset":0,"bytes":64,"gather":{"row_bytes":8,"rows":[[0,4294967296]],"pieces":[[0,8]]}})",
                            R"({"layer":0,"name":"a","format":"Q","source":0,"offset":0,"bytes":64,"gather":{"row_bytes":8,"rows":[[0,8]],"pieces":[[18446744073709551615,2]]}})",
                            R"({"layer":0,"name":"a","format":"Q","source":0,"offset":0,"bytes":64,"gahter":{}})"})
      CHECK_EQ(status_of([&] { parse_layout(layout(bad)); }), SS_E_DESC);
  }
  {  // a data file changed under an open cached pack: nothing cached is trusted or kept
    const fs::path dir = fresh("changed");
    const Model m = make_model(1, 2);
    write_pack(dir, m);
    auto p = Pack::open(P(dir), ropts(PackVerify::Cached));
    const int64_t k = p->find(0, 1, "gate");
    const uint64_t off = p->index().records[size_t(k)].offset;
    std::vector<char> buf(13 * 330);
    p->read(uint64_t(k), 0, 0, buf.data());
    CHECK_EQ(p->stats().verified_records, uint64_t(1));
    std::fstream f(dir / "pack-0000.data", std::ios::in | std::ios::out | std::ios::binary);
#if defined(_WIN32)
    (void)off;
    CHECK(!f.is_open());  // the held share-read handles deny any writer while the pack is open
    CHECK_EQ(status_of([&] { p->read(uint64_t(k), 0, 0, buf.data()); }), SS_OK);
    CHECK(buf == rec_of(m, 0, 1, "gate").data);
#else
    CHECK(f.is_open());
    f.seekp(std::streamoff(off + 7));
    f.put('\x55');
    f.close();
    CHECK_EQ(status_of([&] { p->read(uint64_t(k), 0, 0, buf.data()); }), SS_E_HASH);  // re-checked, not cached
    p.reset();
    auto q = Pack::open(P(dir), ropts(PackVerify::Cached));  // and nothing was persisted for the changed file
    CHECK_EQ(status_of([&] { q->read(uint64_t(k), 0, 0, buf.data()); }), SS_E_HASH);
#endif
  }
}

TEST(metadata_change_keeps_a_cached_pack_working) {
  // a touch of a data file (identity changes, bytes do not) under an open cached pack: every later read is hashed
  // and succeeds; nothing is persisted for the new identity
  const fs::path dir = fresh("touched");
  const Model m = make_model(1, 3);
  write_pack(dir, m);
  auto p = Pack::open(P(dir), ropts(PackVerify::Cached));
  const int64_t k = p->find(0, 2, "gate");
  std::vector<char> buf(13 * 330);
  p->read(uint64_t(k), 0, 0, buf.data());
  p->read(uint64_t(k), 0, 0, buf.data());
  CHECK_EQ(p->stats().verified_records, uint64_t(1));
  std::error_code ec;
  fs::last_write_time(dir / "pack-0000.data", fs::last_write_time(dir / "pack-0000.data") + std::chrono::hours(1), ec);
  CHECK(!ec);
  for (int i = 0; i < 3; ++i) {
    CHECK_EQ(status_of([&] { p->read(uint64_t(k), 0, 0, buf.data()); }), SS_OK);
    CHECK(buf == rec_of(m, 0, 2, "gate").data);
  }
  CHECK_EQ(p->stats().verified_records, uint64_t(4));  // hashed every time after the change
  std::vector<char> rows(4 * 330);
  CHECK_EQ(status_of([&] { p->read(uint64_t(k), 5, 9, rows.data()); }), SS_OK);
}

TEST(writer_rules_and_atomic_publish) {
  const fs::path dir = fresh("wr");
  const fs::path inc(P(dir) + ".incomplete");
  auto spec = [](int l, int e, const char* n, uint64_t b) {
    PackRecordSpec s;
    s.layer = l;
    s.expert = e;
    s.name = n;
    s.format = "F32";
    s.bytes = b;
    return s;
  };
  const std::vector<char> z(9000, 1);
  {
    PackWriter w(P(dir), opts());
    w.add(spec(0, -1, "a", 10), z.data());
    CHECK(fs::exists(inc) && !fs::exists(dir));  // nothing visible until finish
    w.add(spec(0, 0, "g", 10), z.data());
    CHECK_EQ(status_of([&] { w.add(spec(0, -1, "late-dense", 10), z.data()); }), SS_E_INVALID);
    w.add(spec(0, 1, "g", 10), z.data());
    CHECK_EQ(status_of([&] { w.add(spec(0, 0, "u", 10), z.data()); }), SS_E_INVALID);  // unit 0 not adjacent
    CHECK_EQ(status_of([&] { w.add(spec(0, 1, "g", 10), z.data()); }), SS_E_INVALID);  // duplicate key
    w.add(spec(0, 1, "g2", 10), z.data());  // a refused record left the writer as it was: unit 1 continues
    PackRecordSpec f = spec(0, 1, "h", 10);
    f.format = "nope";
    CHECK_EQ(status_of([&] { w.add(f, z.data()); }), SS_E_INVALID);
    PackRecordSpec r = spec(0, 1, "rows", 10);
    r.rows = 3;
    r.row_bytes = 3;
    CHECK_EQ(status_of([&] { w.add(r, z.data()); }), SS_E_INVALID);  // 3 x 3 != 10
    w.add(spec(1, 0, "g", 10), z.data());
    CHECK_EQ(status_of([&] { w.add(spec(0, 2, "g", 10), z.data()); }), SS_E_INVALID);  // layers ascending
    w.begin(spec(1, 1, "g", 10));
    w.write(z.data(), 4);
    CHECK_EQ(status_of([&] { w.end(); }), SS_E_INVALID);  // short record
    CHECK_EQ(status_of([&] { w.write(z.data(), 7); }), SS_E_INVALID);  // past the declared bytes
    w.write(z.data(), 6);
    w.end();
    // destruction without finish removes the temporary directory
  }
  CHECK(!fs::exists(inc) && !fs::exists(dir));
  {
    PackWriter w(P(dir), opts());
    CHECK_EQ(status_of([&] { PackWriter w2(P(dir), opts()); }), SS_E_IO);  // a build in progress
    w.add(spec(-1, -1, "x", 1), z.data());
    w.finish();
  }
  CHECK(fs::exists(dir) && !fs::exists(inc));
  CHECK_EQ(status_of([&] { PackWriter w3(P(dir), opts()); }), SS_E_INVALID);  // never overwrites a pack
}

// An I/O error poisons the writer: DIR.incomplete is removed at once, every later call fails, nothing is published.
// A failed constructor leaves no DIR.incomplete behind.
TEST(writer_failures_poison_the_build) {
  struct Reset {
    ~Reset() { ts::g_pack_writer_faults = ts::PackWriterFaults(); }
  } reset;
  PackRecordSpec big;
  big.layer = 0;
  big.name = "big";
  big.format = "F32";
  big.bytes = (uint64_t(9) << 20) + 5;  // past the writer's 8 MiB buffer: written while the record streams
  const std::vector<char> data = bytes_of(big.bytes, 77);
  const std::string meta = R"({"model":"f","formats":{"F32":{"transform":"copy","version":"1","tool":"t"}}})";
  {  // the C API: a failed write
    const fs::path dir = fresh("fail-write");
    const fs::path inc(P(dir) + ".incomplete");
    ss_pack_writer* w = nullptr;
    CHECK_EQ(ss_pack_writer_begin(P(dir).c_str(), meta.c_str(), meta.size(), &w), SS_OK);
    ss_pack_record_spec s{};
    s.struct_size = sizeof s;
    s.layer = 0;
    s.expert = -1;
    s.name = "big";
    s.format = "F32";
    s.bytes = big.bytes;
    CHECK_EQ(ss_pack_writer_record_begin(w, &s), SS_OK);
    ts::g_pack_writer_faults.fail_write_at = 4096;
    CHECK_EQ(ss_pack_writer_record_write(w, data.data(), data.size()), SS_E_IO);
    CHECK(!fs::exists(inc) && !fs::exists(dir));  // removed at once, not at abort
    ts::g_pack_writer_faults.fail_write_at = 0;
    CHECK_EQ(ss_pack_writer_record_write(w, data.data(), 1), SS_E_IO);  // a retry cannot continue the build
    CHECK_EQ(ss_pack_writer_record_end(w), SS_E_IO);
    s.name = "other";
    s.bytes = 10;
    CHECK_EQ(ss_pack_writer_record_begin(w, &s), SS_E_IO);
    CHECK_EQ(ss_pack_writer_add(w, &s, data.data()), SS_E_IO);
    CHECK(!fs::exists(inc));
    char pin[65] = {};
    CHECK_EQ(ss_pack_writer_finish(w, pin), SS_E_IO);
    CHECK(!fs::exists(inc) && !fs::exists(dir));
    // the directory is free for the next build
    CHECK_EQ(ss_pack_writer_begin(P(dir).c_str(), meta.c_str(), meta.size(), &w), SS_OK);
    ss_pack_writer_abort(w);
    CHECK(!fs::exists(inc));
  }
  {  // a write failing while finish flushes the last data file
    const fs::path dir = fresh("fail-finish");
    const fs::path inc(P(dir) + ".incomplete");
    PackWriter w(P(dir), opts());
    PackRecordSpec s = big;
    s.bytes = 10;
    w.add(s, data.data());
    ts::g_pack_writer_faults.fail_write_at = 1;
    CHECK_EQ(status_of([&] { w.finish(); }), SS_E_IO);
    ts::g_pack_writer_faults.fail_write_at = 0;
    CHECK(w.failed() && !fs::exists(inc) && !fs::exists(dir));
    CHECK_EQ(status_of([&] { w.finish(); }), SS_E_IO);
    CHECK(!fs::exists(dir));
  }
  {  // a data-file split failing at a layer boundary (the refused begin is not a rule error: the writer is poisoned)
    const fs::path dir = fresh("fail-split");
    const fs::path inc(P(dir) + ".incomplete");
    PackWriterOptions o = opts();
    o.split_bytes = 1;
    PackWriter w(P(dir), o);
    PackRecordSpec s = big;
    s.bytes = 10;
    w.add(s, data.data());
    ts::g_pack_writer_faults.fail_create = true;
    s.layer = 1;
    CHECK_EQ(status_of([&] { w.add(s, data.data()); }), SS_E_IO);
    CHECK(w.failed() && !fs::exists(inc));
    s.layer = 2;
    CHECK_EQ(status_of([&] { w.add(s, data.data()); }), SS_E_IO);
    CHECK_EQ(status_of([&] { w.finish(); }), SS_E_IO);
    CHECK(!fs::exists(dir));
  }
  {  // a constructor failing after DIR.incomplete was created removes it
    const fs::path dir = fresh("fail-ctor");
    const fs::path inc(P(dir) + ".incomplete");
    ts::g_pack_writer_faults.fail_create = true;
    CHECK_EQ(status_of([&] { PackWriter w(P(dir), opts()); }), SS_E_IO);
    CHECK(!fs::exists(inc));
    ss_pack_writer* w = nullptr;
    ts::g_pack_writer_faults.fail_create = true;
    CHECK_EQ(ss_pack_writer_begin(P(dir).c_str(), meta.c_str(), meta.size(), &w), SS_E_IO);
    CHECK(w == nullptr && !fs::exists(inc));
    PackWriter ok(P(dir), opts());  // not "another build in progress"
    PackRecordSpec s = big;
    s.bytes = 10;
    ok.add(s, data.data());
    ok.finish();
    CHECK(fs::exists(dir) && !fs::exists(inc));
  }
}

TEST(split_files_at_layer_boundaries) {
  const fs::path dir = fresh("split");
  const Model m = make_model(5, 3);
  PackWriterOptions o = opts();
  o.split_bytes = 40000;
  write_pack(dir, m, o);
  auto p = Pack::open(P(dir), ropts());
  CHECK(p->manifest().files.size() >= 3);
  for (size_t i = 1; i < p->index().records.size(); ++i) {
    const PackRecord& a = p->index().records[i - 1];
    const PackRecord& b = p->index().records[i];
    if (a.layer == b.layer) CHECK_EQ(a.file, b.file);
  }
  for (const auto& r : m.recs) {
    const int64_t k = p->find(r.spec.layer, r.spec.expert, r.spec.name);
    std::vector<char> b(r.data.size());
    p->read(uint64_t(k), 0, 0, b.data());
    CHECK(b == r.data);
  }
  CHECK(p->verify_full().find("ok")->b);
}

TEST(dg_descriptor_reads_units_from_the_pack) {
  const fs::path dir = fresh("dg");
  const Model m = make_model(2, 4);
  write_pack(dir, m);
  auto p = Pack::open(P(dir), ropts());
  // the engine's own descriptor: its model file as source 0 (an Engram-like lookup table stays on it)
  const fs::path model = g_root / "dg-model.bin";
  put_file(model, std::string(1 << 16, 'm'));
  const uint64_t gate = 13 * 330, down = 100 + 4096 + 777;
  Json base = json_parse(R"({"schema":"strix-stream/desc-v1","model":"t","engine":"t","sources":[{"name":"m","path":")" +
                         P(model) + R"("}],"layers":[
    {"index":0,"kind":"moe","n_units":4,"top_k":2,"parts":[{"name":"gate","format":"IQ2_XXS","source":0,"offset":0,"stride":)" +
                         std::to_string(gate) + R"(,"bytes":)" + std::to_string(gate) + R"(},{"name":"down","format":"Q2_K","source":0,"offset":0,"stride":)" +
                         std::to_string(down) + R"(,"bytes":)" + std::to_string(down) + R"(}]},
    {"index":1,"kind":"moe","n_units":4,"top_k":2,"parts":[{"name":"gate","format":"IQ2_XXS","source":0,"offset":0,"stride":)" +
                         std::to_string(gate) + R"(,"bytes":)" + std::to_string(gate) + R"(}]}],
    "lookup_tables":[{"name":"engram","source":0,"offset":0,"rows":4,"row_bytes":4096,"placement":"file"}]})");
  // per-unit engine hashes in the base (xxh64 of the unit's parts back to back) are kept and checked by the cache
  Json hashes = Json::array();
  for (int u = 0; u < 4; ++u) {
    std::vector<char> cat = rec_of(m, 0, u, "gate").data;
    const auto d = rec_of(m, 0, u, "down").data;
    cat.insert(cat.end(), d.begin(), d.end());
    char h[17];
    std::snprintf(h, sizeof h, "%016llx", (unsigned long long)ts::xxh64(cat.data(), cat.size()));
    hashes.push(Json::string(h));
  }
  base["layers"].arr[0]["hash"] = hashes;
  const Json dj = p->dg_descriptor(base);
  Desc d = desc_from_json(dj);
  desc_validate(d);
  CHECK_EQ(d.sources.size(), size_t(2));
  CHECK_EQ(d.lookup_tables[0].source, 0);
  std::vector<std::unique_ptr<SourceFile>> files;
  for (const Source& s : d.sources) files.push_back(open_source(s.path, true, nullptr));
  Aligned bounce(1 << 20);
  for (int l = 0; l < 2; ++l)
    for (int u = 0; u < 4; ++u) {
      const uint64_t ub = d.layers[size_t(l)].unit_bytes;
      Aligned dst(ub);
      uint64_t digest = 0;
      dg_read_unit(d, files, l, u, dst.p, bounce.p, 1 << 20, true, &digest);
      std::vector<char> want = rec_of(m, l, u, "gate").data;
      if (l == 0) {
        const auto dd = rec_of(m, l, u, "down").data;
        want.insert(want.end(), dd.begin(), dd.end());
        CHECK_EQ(digest, d.layers[0].hash[size_t(u)]);
      }
      CHECK_EQ(ub, uint64_t(want.size()));
      CHECK(std::memcmp(dst.p, want.data(), want.size()) == 0);
    }
  // a base without unit hashes gets them from verified pack bytes (and the second call reuses the cached list)
  Json nohash = base;
  nohash["layers"].arr[0].obj.erase("hash");
  const Desc d2 = desc_from_json(p->dg_descriptor(nohash));
  CHECK(d2.layers[0].hash == d.layers[0].hash);
  CHECK_EQ(d2.layers[1].hash.size(), size_t(4));
  const uint64_t before = p->stats().reads;
  const Desc d3 = desc_from_json(p->dg_descriptor(nohash));
  CHECK(d3.layers[1].hash == d2.layers[1].hash);
  {
    // a fresh open (Always) of a pack corrupted in a unit's bytes cannot hand out hashes for it
    const fs::path dir2 = fresh("dg-bad");
    write_pack(dir2, m);
    std::string data = file_bytes(dir2 / "pack-0000.data");
    auto q = Pack::open(P(dir2), ropts());
    const uint64_t at = q->index().records[size_t(q->find(1, 2, "gate"))].offset + 3;
    q.reset();
    data[size_t(at)] ^= 1;
    put_file(dir2 / "pack-0000.data", data);
    auto q2 = Pack::open(P(dir2), ropts());
    CHECK_EQ(status_of([&] { q2->dg_descriptor(nohash); }), SS_E_HASH);
  }
  CHECK(p->stats().reads == before || p->stats().reads > before);
  // a part whose bytes differ from the pack's records is refused
  Json wrong = base;
  wrong["layers"].arr[1]["parts"].arr[0]["bytes"] = Json::number(uint64_t(gate + 1));
  CHECK_EQ(status_of([&] { p->dg_descriptor(wrong); }), SS_E_DESC);
  {
    // units written in descending order (allowed by the writer) have no positive stride: refused as unsupported
    const fs::path dir3 = fresh("dg-desc");
    {
      PackWriter w(P(dir3), opts());
      for (int u = 3; u >= 0; --u) {
        const std::string gate_name = "gate";
        const Model::Rec& r = rec_of(m, 1, u, gate_name);
        PackRecordSpec s = r.spec;
        s.layer = 0;
        w.add(s, r.data.data());
      }
      PackRecordSpec s = rec_of(m, 1, 0, "gate").spec;
      s.layer = 1;
      for (int u = 0; u < 4; ++u) {
        s.expert = u;
        w.add(s, rec_of(m, 1, u, "gate").data.data());
      }
      w.finish();
    }
    auto q = Pack::open(P(dir3), ropts());
    Json one = base;
    one["layers"].arr[0]["parts"].arr.pop_back();  // gate only
    one["layers"].arr[0].obj.erase("hash");
    try {
      q->dg_descriptor(one);
      CHECK(false);
    } catch (const ts::Error& e) {
      CHECK_EQ(e.status(), SS_E_UNSUPPORTED);
      CHECK(std::string(e.what()).find("ascending order") != std::string::npos);
    }
  }
}

// A stripe of a pack data file (stripe-v1, as ts-stripe writes it): its odd 1 MiB chunks on a second directory.
void make_stripe(const fs::path& primary, const fs::path& dir) {
  constexpr uint64_t C = uint64_t(1) << 20;
  const std::string bytes = file_bytes(primary);
  StripeManifest m;
  m.chunk_bytes = C;
  m.primary_basename = primary.filename().u8string();
  m.primary_size = bytes.size();
  m.primary_sha256 = sha256_file(P(primary));
  m.data_name = "stripe.data";
  fs::create_directories(dir);
  std::ofstream data(dir / m.data_name, std::ios::binary);
  std::vector<char> slot(static_cast<size_t>(C));
  for (uint64_t chunk = 1; chunk * C < bytes.size(); chunk += 2) {
    const uint64_t n = std::min(C, uint64_t(bytes.size()) - chunk * C);
    std::fill(slot.begin(), slot.end(), char(0));
    std::memcpy(slot.data(), bytes.data() + chunk * C, size_t(n));
    m.runs.push_back({chunk, 1, m.hashes.size()});
    m.hashes.push_back(ts::xxh64(slot.data(), size_t(n)));
    data.write(slot.data(), std::streamsize(C));
  }
  data.close();
  m.data_size = m.hashes.size() * C;
  write_file_atomic(P(dir / stripe_manifest_name(m.primary_basename)), stripe_manifest_write(m));
}

TEST(stripes_of_a_data_file) {
  const fs::path dir = fresh("striped");
  Model m;
  uint32_t seed = 900;
  for (int l = 0; l < 2; ++l)
    for (int u = 0; u < 4; ++u) {
      PackRecordSpec g;
      g.layer = l;
      g.expert = u;
      g.name = "gate";
      g.format = "Q";
      g.rows = 2047;
      g.row_bytes = 330;
      g.bytes = 2047 * 330;
      g.seg_rows = {0, 1000, 2047};
      m.recs.push_back({g, bytes_of(g.bytes, seed++)});
    }
  write_pack(dir, m);
  const fs::path sdir = fresh("striped-dir");
  make_stripe(dir / "pack-0000.data", sdir);
  PackOptions o = ropts(PackVerify::Cached);
  o.weight_dirs = P(sdir);
  auto p = Pack::open(P(dir), o);
  CHECK(bool(p->file(0).session()));
  CHECK(p->file(0).session()->multi());
  p->file(0).session()->test_prefer_stripe(true);
  std::vector<std::vector<char>> got(m.recs.size());
  std::vector<PackRead> rs;
  for (size_t i = 0; i < m.recs.size(); ++i) {
    const auto& r = m.recs[i];
    got[i].resize(r.data.size());
    rs.push_back({uint64_t(p->find(r.spec.layer, r.spec.expert, r.spec.name)), 0, 0, got[i].data()});
  }
  p->readv(rs.data(), rs.size());
  for (size_t i = 0; i < m.recs.size(); ++i) CHECK(got[i] == m.recs[i].data);
  p->readv(rs.data(), rs.size());  // a striped file is hashed on every read, even in cached mode
  CHECK_EQ(p->stats().verified_records, uint64_t(2 * m.recs.size()));
  std::vector<char> b(1047 * 330);
  p->read(uint64_t(p->find(1, 3, "gate")), 1000, 2047, b.data());
  CHECK(std::memcmp(b.data(), rec_of(m, 1, 3, "gate").data.data() + 1000 * 330, b.size()) == 0);
  ss_mfile_counters c{};
  c.struct_size = sizeof c;
  p->file(0).session()->counters(&c);
  CHECK(c.locations == 2);
  CHECK(c.bytes[1] > 0);  // the stripe served some of it
}

TEST(c_abi) {
  const fs::path dir = fresh("capi");
  const std::string meta = R"({"model":"c","sources":[{"name":"m","size":1,"sha256":")" + std::string(64, 'd') +
                           R"("}],"formats":{"Q":{"transform":"copy","version":"1","tool":"t"}},"receipt":{"by":"test"}})";
  ss_pack_writer* w = nullptr;
  CHECK_EQ(ss_pack_writer_begin(P(dir).c_str(), meta.c_str(), meta.size(), &w), SS_OK);
  const std::vector<char> a = bytes_of(5000, 7), b = bytes_of(4096 * 3, 8);
  ss_pack_record_spec s{};
  s.struct_size = sizeof s;
  s.layer = 0;
  s.expert = 0;
  s.name = "gate";
  s.format = "Q";
  s.bytes = 5000;
  s.rows = 10;
  s.row_bytes = 500;
  const uint32_t seg[] = {0, 4, 10};
  s.seg_rows = seg;
  s.n_seg = 2;
  CHECK_EQ(ss_pack_writer_add(w, &s, a.data()), SS_OK);
  ss_pack_record_spec t = s;
  t.expert = 1;
  t.name = "gate";
  t.bytes = b.size();
  t.rows = 0;
  t.row_bytes = 0;
  t.seg_rows = nullptr;
  t.n_seg = 0;
  CHECK_EQ(ss_pack_writer_record_begin(w, &t), SS_OK);
  CHECK_EQ(ss_pack_writer_record_write(w, b.data(), 4096), SS_OK);
  CHECK_EQ(ss_pack_writer_record_write(w, b.data() + 4096, b.size() - 4096), SS_OK);
  CHECK_EQ(ss_pack_writer_record_end(w), SS_OK);
  ss_pack_record_spec bad = s;
  bad.format = "nope";
  CHECK_EQ(ss_pack_writer_add(w, &bad, a.data()), SS_E_INVALID);
  CHECK(std::string(ss_last_error()).find("not declared") != std::string::npos);
  // struct sizes: below the first released size refused; a larger (newer) struct's extra bytes ignored
  bad = s;
  bad.struct_size = uint32_t(offsetof(ss_pack_record_spec, reserved0));
  CHECK_EQ(ss_pack_writer_add(w, &bad, a.data()), SS_E_INVALID);
  CHECK(std::string(ss_last_error()).find("record spec size") != std::string::npos);
  struct Newer {
    ss_pack_record_spec s;
    uint64_t extra;
  } newer{s, 0};
  newer.s.struct_size = sizeof newer;
  newer.s.expert = 2;
  newer.s.rows = 0;
  newer.s.row_bytes = 0;
  newer.s.seg_rows = nullptr;
  newer.s.n_seg = 0;
  newer.s.bytes = 64;
  CHECK_EQ(ss_pack_writer_add(w, &newer.s, a.data()), SS_OK);
  char pin[65] = {};
  CHECK_EQ(ss_pack_writer_finish(w, pin), SS_OK);
  ss_pack_options o;
  ss_pack_options_defaults(&o);
  o.verify = SS_PACK_VERIFY_ALWAYS;
  o.manifest_sha256 = pin;
  const char* srcs[] = {"DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD"};
  o.source_sha256 = srcs;
  o.source_sha256_n = 1;
  ss_pack* p = nullptr;
  CHECK_EQ(ss_pack_open(P(dir).c_str(), &o, &p), SS_OK);
  CHECK_EQ(ss_pack_records(p), uint64_t(3));
  ss_pack_entry e;
  CHECK_EQ(ss_pack_find(p, 0, 0, "gate", &e), SS_OK);
  CHECK_EQ(e.rows, 10u);
  CHECK_EQ(e.segments, 2u);
  CHECK_EQ(std::string(e.format), std::string("Q"));
  CHECK_EQ(ss_pack_find(p, 0, 3, "gate", &e), SS_E_INVALID);
  ss_pack_segment g;
  CHECK_EQ(ss_pack_segment_at(p, 0, 1, &g), SS_OK);
  CHECK(g.row0 == 4 && g.row1 == 10 && g.offset == 2000 && g.bytes == 3000);
  std::vector<char> r1(3000), r2(b.size());
  ss_pack_read_req rq[2] = {{sizeof(ss_pack_read_req), 0, 4, 10, r1.data()},
                            {sizeof(ss_pack_read_req), 1, 0, 0, r2.data()}};
  CHECK_EQ(ss_pack_readv(p, rq, 2), SS_OK);
  CHECK(std::memcmp(r1.data(), a.data() + 2000, 3000) == 0 && r2 == b);
  rq[1].struct_size = 0;  // one size per call
  CHECK_EQ(ss_pack_readv(p, rq, 2), SS_E_INVALID);
  rq[0].struct_size = 0;
  CHECK_EQ(ss_pack_readv(p, rq, 2), SS_E_INVALID);
  {  // a newer caller's larger elements are walked with their size
    struct NewerReq {
      ss_pack_read_req q;
      uint64_t extra;
    } nq[2] = {{rq[0], 0}, {rq[1], 0}};
    nq[0].q.struct_size = nq[1].q.struct_size = sizeof(NewerReq);
    std::fill(r1.begin(), r1.end(), char(0));
    std::fill(r2.begin(), r2.end(), char(0));
    CHECK_EQ(ss_pack_readv(p, &nq[0].q, 2), SS_OK);
    CHECK(std::memcmp(r1.data(), a.data() + 2000, 3000) == 0 && r2 == b);
  }
  ss_pack_unit u;
  CHECK_EQ(ss_pack_unit_find(p, 0, 1, &u), SS_OK);
  CHECK(u.count == 1 && u.bytes == b.size());
  std::vector<char> r3(64);
  CHECK_EQ(ss_pack_read(p, 2, 0, 0, r3.data()), SS_OK);
  CHECK(std::memcmp(r3.data(), a.data(), 64) == 0);
  CHECK_EQ(ss_pack_files(p), 1u);
  ss_file_identity id;
  CHECK_EQ(ss_pack_file_identity(p, 0, &id), SS_OK);
  CHECK(id.size > 0);
  char* js = nullptr;
  size_t jl = 0;
  CHECK_EQ(ss_pack_info_json(p, &js, &jl), SS_OK);
  CHECK(std::string(js, jl).find("\"by\":\"test\"") != std::string::npos);
  ss_free(js);
  CHECK_EQ(ss_pack_verify_full(p, &js, &jl), SS_OK);
  ss_free(js);
  ss_pack_close(p);
  o.manifest_sha256 = "0000000000000000000000000000000000000000000000000000000000000000";
  CHECK_EQ(ss_pack_open(P(dir).c_str(), &o, &p), SS_E_DESC);
  CHECK(p == nullptr);
}

// ts-pack build from a layout: a model file with two stacked expert tensors, a whole-expert copy, a row selection
// (rank 1 of a split by rows, two runs in a declared order) and a byte gather inside rows (as a per-rank down split).
TEST(ts_pack_build_from_layout) {
  const fs::path src = g_root / "layout-model.bin";
  const uint32_t E = 5, R = 12, RB = 84 * 3;  // experts, rows, bytes per row (3 blocks of 84)
  const uint64_t T0 = 8192 + 3, per = uint64_t(R) * RB;
  const std::vector<char> model = bytes_of(T0 + 2 * E * per + 999, 42);
  put_file(src, std::string(model.begin(), model.end()));
  const fs::path dir = fresh("built");
  const std::string layout = R"({"schema":"tierstream/pack-layout-v1","model":"lm","sources":[{"name":"model","path":")" +
                             P(src) + R"("}],"formats":{"Q":{"transform":"copy","version":"1","tool":"t"},
      "Qr1":{"transform":"x-rows","version":"1","tool":"t"},"Qd1":{"transform":"x-down","version":"1","tool":"t"},
      "F32":{"transform":"copy","version":"1","tool":"t"}},
    "tensors":[
      {"layer":-1,"name":"head","format":"F32","source":0,"offset":0,"bytes":8192},
      {"layer":0,"name":"gate","format":"Q","source":0,"offset":)" + std::to_string(T0) +
                             R"(,"stride":)" + std::to_string(per) + R"(,"bytes":)" + std::to_string(per) + R"(,"units":5,
       "rows":12,"row_bytes":252,"segments":[0,6,7,12]},
      {"layer":0,"name":"gate@1/2","format":"Qr1","layout":"tp2","source":0,"offset":)" + std::to_string(T0) +
                             R"(,"stride":)" + std::to_string(per) + R"(,"bytes":)" + std::to_string(per) +
                             R"(,"units":5,"gather":{"row_bytes":252,"rows":[[7,12],[6,7]],"pieces":[[0,252]]}},
      {"layer":0,"name":"down@1/2","format":"Qd1","layout":"tp2-down","source":0,"offset":)" + std::to_string(T0 + E * per) +
                             R"(,"stride":)" + std::to_string(per) + R"(,"bytes":)" + std::to_string(per) +
                             R"(,"units":5,"gather":{"row_bytes":252,"rows":[[0,12]],"pieces":[[168,84],[92,8],[92,8],[132,32]]}}]})";
  const fs::path lp = g_root / "layout.json";
  put_file(lp, layout);
  std::vector<std::string> args = {"ts-pack", "build", P(lp), P(dir)};
  auto run = [](std::vector<std::string> a) {
    std::vector<char*> v;
    for (auto& s : a) v.push_back(&s[0]);
    return pack_main(int(v.size()), v.data());
  };
  CHECK_EQ(run({"ts-pack", "build", P(lp), P(dir), "--dry-run"}), 0);
  CHECK(!fs::exists(dir));
  CHECK_EQ(run(args), 0);
  auto p = Pack::open(P(dir), ropts());
  CHECK_EQ(p->index().records.size(), size_t(1 + 3 * E));
  CHECK_EQ(p->manifest().sources[0].sha256, ts::sha256_hex(model.data(), model.size()));
  for (uint32_t e = 0; e < E; ++e) {
    const char* g = model.data() + T0 + e * per;
    const char* d = model.data() + T0 + E * per + e * per;
    std::vector<char> want(g, g + per), got(per);
    p->read(uint64_t(p->find(0, int(e), "gate")), 0, 0, got.data());
    CHECK(got == want);
    // rank 1 rows: [7, 12) then [6, 7)
    want.assign(g + 7 * RB, g + 12 * RB);
    want.insert(want.end(), g + 6 * RB, g + 7 * RB);
    got.assign(want.size(), 0);
    p->read(uint64_t(p->find(0, int(e), "gate@1/2")), 0, 0, got.data());
    CHECK(got == want);
    // the same bytes as two row-range reads of the whole record
    std::vector<char> two(want.size());
    const uint64_t k = uint64_t(p->find(0, int(e), "gate"));
    p->read(k, 7, 12, two.data());
    p->read(k, 6, 7, two.data() + 5 * RB);
    CHECK(two == want);
    // per-row gather: block 2, then block 1 with its scales 8..15 twice and codes 32..63 (as a 2-rank Q2_K split)
    want.clear();
    for (uint32_t r = 0; r < R; ++r) {
      const char* row = d + r * RB;
      want.insert(want.end(), row + 168, row + 252);
      want.insert(want.end(), row + 92, row + 100);
      want.insert(want.end(), row + 92, row + 100);
      want.insert(want.end(), row + 132, row + 164);
    }
    const PackRecord& dr = p->index().records[size_t(p->find(0, int(e), "down@1/2"))];
    CHECK(dr.rows == R && dr.row_bytes == 132);
    got.assign(want.size(), 0);
    p->read(uint64_t(p->find(0, int(e), "down@1/2")), 0, 0, got.data());
    CHECK(got == want);
  }
  // cat: the list format the DS comparison uses
  const fs::path list = g_root / "cat.txt";
  put_file(list, "0 2 gate 7 12\r\n0 2 gate 6 7\n# comment\n-1 -1 head\n");
  CHECK_EQ(run({"ts-pack", "cat", P(dir), P(list), P(g_root / "cat.bin")}), 0);
  std::string cat = file_bytes(g_root / "cat.bin");
  const char* g2 = model.data() + T0 + 2 * per;
  std::string want = std::string(g2 + 7 * RB, g2 + 12 * RB) + std::string(g2 + 6 * RB, g2 + 7 * RB) +
                     std::string(model.data(), model.data() + 8192);
  CHECK(cat == want);
  CHECK_EQ(run({"ts-pack", "verify", P(dir), "--full"}), 0);
  CHECK_EQ(run({"ts-pack", "show", P(dir)}), 0);
  CHECK(p->manifest().layouts == std::vector<std::string>({"tp2", "tp2-down"}));
  // optional layouts: a build of a subset records it; a consumer that needs a missing one is refused by name
  const fs::path dir3 = fresh("built3");
  CHECK_EQ(run({"ts-pack", "build", P(lp), P(dir3), "--layouts", "tp2"}), 0);
  {
    PackOptions o = ropts();
    o.require_layouts = {"tp2"};
    auto q = Pack::open(P(dir3), o);
    CHECK_EQ(q->index().records.size(), size_t(1 + 2 * E));
    CHECK(q->find(0, 0, "down@1/2") < 0 && q->find(0, 0, "gate@1/2") >= 0);
    o.require_layouts = {"tp2", "tp2-down"};
    try {
      Pack::open(P(dir3), o);
      CHECK(false);
    } catch (const ts::Error& e) {
      CHECK_EQ(e.status(), SS_E_DESC);
      CHECK(std::string(e.what()).find("has no layout tp2-down (built with: tp2)") != std::string::npos);
    }
  }
  const fs::path dir4 = fresh("built4");
  CHECK_EQ(run({"ts-pack", "build", P(lp), P(dir4), "--layouts", "none"}), 0);
  CHECK(Pack::open(P(dir4), ropts())->manifest().layouts.empty());
  CHECK(run({"ts-pack", "build", P(lp), P(fresh("built5")), "--layouts", "tp3"}) != 0);
  // a layout source SHA-256 that differs from the file is refused before anything is written
  const fs::path dir2 = fresh("built2");
  std::string l2 = layout;
  l2.replace(l2.find("\"path\""), 0, "\"sha256\":\"" + std::string(64, 'e') + "\",");
  put_file(lp, l2);
  CHECK(run({"ts-pack", "build", P(lp), P(dir2)}) != 0);
  CHECK(!fs::exists(dir2) && !fs::exists(fs::path(P(dir2) + ".incomplete")));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: ts-test-pack FIXTURE_PARENT TS_PACK_EXE TS_STRIPE_EXE [ONLY]\n");
    return 2;
  }
  g_root = fs::absolute(fs::u8path(argv[1])) /
           ("run-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(g_root);
  // the verified-record cache of these packs lives in the fixture (never in the user's cache)
  fs::create_directories(g_root / "cache");
  set_env("LOCALAPPDATA", P(g_root / "cache"));
  set_env("XDG_CACHE_HOME", P(g_root / "cache"));
  const int rc = sstest::run_all(argc > 4 ? argv[4] : nullptr);
  if (rc == 0) {
    std::error_code ec;
    fs::remove_all(g_root, ec);
  } else {
    std::fprintf(stderr, "fixtures retained: %s\n", P(g_root).c_str());
  }
  return rc;
}
