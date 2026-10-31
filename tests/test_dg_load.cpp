#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <mutex>
#include <new>
#include <algorithm>
#include <vector>
#include "check.hpp"
#include "dg_fixture.hpp"
#include "dg_load.hpp"
#include "dg_layout.hpp"
#include "dg_proto.hpp"
#include "errors.hpp"
#include "fixture.hpp"
#include "file_reader.hpp"
#include "xxh64.hpp"
namespace {
struct Fixture {
  sstest::DgModel model{"load", 3, 32};
  ss::Desc d = ss::desc_from_json(ss::json_parse(model.json));
  std::vector<ss::DgLoadUnit> units;
  std::vector<std::vector<char>> homes, device;
  Fixture(bool alternating = false, bool all_resident = false, bool all_host = false) {
    homes.resize(size_t(model.G*model.U)); device.resize(homes.size());
    int32_t slot = 0;
    for (int l=0; l<model.G; ++l) {
      for (int u=0; u<model.U; ++u) {
        const size_t k = units.size();
        homes[k].resize(size_t(model.unit()), char(0x55));
        device[k].resize(size_t(model.unit()), char(0x33));
        // Alternating groups exercise interleaved placement and part-major batches.
        const bool resident = !all_host && (all_resident || (alternating ? u%2 == 0 : (u/8)%2 == 0));
        units.push_back({l,u,resident ? slot++ : -1, resident ? nullptr : homes[k].data()});
        auto bytes=model.bytes(l,u);
        d.layers[size_t(l)].hash.push_back(ss::xxh64(bytes.data(),bytes.size()));
      }
    }
  }
};
void exercise(bool direct, uint32_t readers, size_t count, bool alternating = false, bool all_resident = false) {
  Fixture f(alternating, all_resident);
  // Split handles exercise source ordering independently of part/hash ordering.
  f.d.sources.push_back(f.d.sources[0]);
  for (auto& layer : f.d.layers) layer.parts[1].source=1;
  const uint64_t capacity = all_resident ? 256*1024 : f.model.unit()*4;
  std::vector<std::vector<char>> storage(count, std::vector<char>(size_t(capacity+8192), char(0x6d)));
  std::vector<char*> buffers;
  for (auto& b : storage) buffers.push_back(b.data()+123); // intentionally unaligned caller address
  std::vector<std::vector<ss::DgLoadCopy>> pending(count);
  std::mutex mu;
  auto finish = [&](size_t i) {
    std::lock_guard<std::mutex> lk(mu);
    // Deliberately defer every copy until buffer reuse, checking pointer lifetime.
    for (const auto& c : pending[i]) {
      for (size_t k=0; k<f.units.size(); ++k) if (f.units[k].slot==c.slot) {
        uint64_t off=0;
        for (size_t p=0; p<c.part; ++p) off+=f.d.layers[0].parts[p].bytes;
        std::memcpy(f.device[k].data()+off,c.data,size_t(c.bytes));
      }
    }
    pending[i].clear();
  };
  size_t batches = 0, min_batch = size_t(-1);
  auto stats=ss::dg_load_pipeline(f.d,f.units,buffers,capacity,readers,direct,true,finish,
      [&](size_t i,const std::vector<ss::DgLoadCopy>& copies) {
        std::lock_guard<std::mutex> lk(mu); pending[i]=copies;
        ++batches; min_batch=std::min(min_batch,copies.size()/3);
      });
  for(size_t i=0;i<count;++i) finish(i);
  // Independent serial reads (descriptor order) are the byte oracle for both tiers.
  ss::File serial(f.d.sources[0].path,false);
  for(size_t k=0;k<f.units.size();++k) {
    const auto& u=f.units[k];
    std::vector<char> expected(size_t(f.model.unit()));
    uint64_t off=0;
    for(const auto& p:f.d.layers[size_t(u.layer)].parts) {
      CHECK_EQ(serial.pread(expected.data()+off,p.bytes,p.offset+uint64_t(u.unit)*p.stride),p.bytes);
      off+=p.bytes;
    }
    CHECK((u.slot>=0 ? f.device[k] : f.homes[k])==expected);
    CHECK_EQ(u.slot>=0,u.host==nullptr);
  }
  for (const auto& b : storage) {
    for (size_t i=0;i<123;++i) CHECK_EQ(b[i],char(0x6d));
    for (size_t i=size_t(capacity)+123;i<b.size();++i) CHECK_EQ(b[i],char(0x6d));
  }
  if (alternating && !direct) { CHECK_EQ(batches,12u); CHECK_EQ(min_batch,4u); }
  if (all_resident) CHECK_EQ(stats.bounce_bytes,0ull);
  CHECK_EQ(stats.payload_bytes,uint64_t(f.units.size())*f.model.unit());
  CHECK(stats.read_bytes>=stats.payload_bytes);
  CHECK(stats.read_seconds>0);
  if(!direct && !alternating) CHECK(stats.read_calls<f.units.size()*3); // adjacent residents coalesced
}
}
TEST(dg_load_bytes_and_delayed_reuse) {
  exercise(false,1,1); exercise(false,4,2); exercise(true,4,4); exercise(true,7,1);
  exercise(false,4,1,true); exercise(true,8,2,true);
  exercise(true,4,2,false,true);
}
TEST(dg_load_host_never_claims_ring) {
  Fixture f(false,false,true);
  std::vector<char> storage(size_t(f.model.unit()),char(0x42));
  const auto stats=ss::dg_load_pipeline(f.d,f.units,{storage.data()},storage.size(),8,true,true,
      [](size_t){CHECK(false);},[](size_t,const std::vector<ss::DgLoadCopy>&){CHECK(false);});
  CHECK_EQ(stats.payload_bytes,uint64_t(f.units.size())*f.model.unit());
  for(char c:storage) CHECK_EQ(c,char(0x42));
}
TEST(dg_load_covering_reads_in_host_home) {
  Fixture f(false,false,true);
  const size_t bytes=size_t(f.model.unit());
  struct Aligned {
    char* p;
    explicit Aligned(size_t n):p(static_cast<char*>(::operator new(n,std::align_val_t(4096)))){}
    ~Aligned(){::operator delete(p,std::align_val_t(4096));}
  } homes(bytes*f.units.size()+4096);
  std::memset(homes.p,0x6d,bytes*f.units.size()+4096);
  ss::File serial(f.d.sources[0].path,false);
  std::vector<std::vector<char>> expected(f.units.size(),std::vector<char>(bytes));
  for(auto& l:f.d.layers) {
    // Sector-aligned tensor bases with half-sector gate/up expert strides,
    // as in DS4.1. Reverse disk part order to test safe destination ordering.
    for(auto& p:l.parts) p.offset-=f.model.base;
    std::swap(l.parts[0].offset,l.parts[1].offset);
    l.hash.clear();
  }
  for(size_t k=0;k<f.units.size();++k) {
    auto& u=f.units[k]; u.host=homes.p+k*bytes;
    auto& l=f.d.layers[size_t(u.layer)]; uint64_t off=0;
    for(const auto& p:l.parts) {
      CHECK_EQ(serial.pread(expected[k].data()+off,p.bytes,p.offset+uint64_t(u.unit)*p.stride),p.bytes);
      off+=p.bytes;
    }
    l.hash.push_back(ss::xxh64(expected[k].data(),bytes));
  }
  std::vector<char> ring(bytes);
  const auto stats=ss::dg_load_pipeline(f.d,f.units,{ring.data()},ring.size(),8,true,true,
      [](size_t){CHECK(false);},[](size_t,const std::vector<ss::DgLoadCopy>&){CHECK(false);});
  CHECK_EQ(stats.bounce_bytes,0ull);
  for(size_t k=0;k<f.units.size();++k) CHECK(std::memcmp(f.units[k].host,expected[k].data(),bytes)==0);
  for(size_t k=bytes*f.units.size();k<bytes*f.units.size()+4096;++k) CHECK_EQ(homes.p[k],char(0x6d));
}
TEST(dg_load_hash_and_io_errors_join_readers) {
  Fixture f;
  std::vector<char> storage(size_t(f.model.unit()*2));
  auto invoke=[&] { ss::dg_load_pipeline(f.d,f.units,{storage.data()},storage.size(),4,true,true,
       [](size_t){},[](size_t,const std::vector<ss::DgLoadCopy>&){}); };
  f.d.layers[0].hash[0]^=1;
  try {invoke();CHECK(false);} catch(const ss::Error& e) {CHECK_EQ(int(e.status()),int(SS_E_HASH));}
  f.d.layers[0].hash[0]^=1;
  f.d.layers[0].parts[0].offset=1ull<<40;
  try {invoke();CHECK(false);} catch(const ss::Error& e) {CHECK_EQ(int(e.status()),int(SS_E_IO));}
}
TEST(dg_load_submit_error_joins_readers) {
  Fixture f; std::vector<char> storage(size_t(f.model.unit()));
  try {
    ss::dg_load_pipeline(f.d,f.units,{storage.data()},storage.size(),4,false,true,[](size_t){},
      [](size_t,const std::vector<ss::DgLoadCopy>&){ss::fail(SS_E_BACKEND,"injected transfer failure");});
    CHECK(false);
  } catch(const ss::Error& e) {CHECK_EQ(int(e.status()),int(SS_E_BACKEND));}
}
TEST(dg_load_admission_and_options) {
  struct Env {
    std::string name, old; bool had;
    Env(const char* n, const char* value) : name(n) {
      const char* v=std::getenv(n); had=v!=nullptr; if(v) old=v; set(value);
    }
    void set(const char* v) {
#if defined(_WIN32)
      _putenv_s(name.c_str(),v?v:"");
#else
      if(v) setenv(name.c_str(),v,1); else unsetenv(name.c_str());
#endif
    }
    ~Env(){set(had?old.c_str():nullptr);}
  } serial("DG_LOAD_SERIAL","1"), readers("DG_LOAD_READERS",nullptr),
    buffers("DG_LOAD_BUFFERS",nullptr), bytes("DG_LOAD_BUFFER_MIB",nullptr),
    overlap("DG_LOAD_OVERLAP_ALLOC",nullptr);
  Fixture f; ss_budget b; ss_options o;
  sstest::dg_budget(b,f.model,4,8); sstest::dg_options(o,8,SS_BACKEND_CPU);
  auto p=ss::plan_from_tiers(f.d,b,f.model.tiers(16));
  const auto old=ss::dg_layout(f.d,p,o);
  CHECK_EQ(old.load_buffers,0u);
  for (const char* value : {"true","on"}) {
    serial.set(value); CHECK(ss::dg_layout(f.d,p,o).load_serial);
  }
  for (const char* value : {"false","off"}) {
    serial.set(value); CHECK(!ss::dg_layout(f.d,p,o).load_serial);
  }
  for (const char* value : {"yes"," 1","+1","On"}) {
    serial.set(value);
    try{(void)ss::dg_layout(f.d,p,o); CHECK(false);}
    catch(const ss::Error& e){CHECK_EQ(int(e.status()),int(SS_E_INVALID));}
  }
  serial.set("1");
  p.budget.host_pinned_bytes=p.pinned+old.mapped_bytes+old.bounce_bytes+old.staging_bytes+4*f.model.unit();
  serial.set("0");
  const auto small=ss::dg_layout(f.d,p,o);
  CHECK_EQ(small.load_readers,8u);
  CHECK(small.load_overlap_alloc);  // default on (pipeline only)
  CHECK(!old.load_overlap_alloc);
  overlap.set("0"); CHECK(!ss::dg_layout(f.d,p,o).load_overlap_alloc);
  for (const char* value : {"1","true","on"}) { overlap.set(value); CHECK(ss::dg_layout(f.d,p,o).load_overlap_alloc); }
  for (const char* value : {"false","off"}) { overlap.set(value); CHECK(!ss::dg_layout(f.d,p,o).load_overlap_alloc); }
  overlap.set("yes");
  try{(void)ss::dg_layout(f.d,p,o); CHECK(false);}
  catch(const ss::Error& e){CHECK_EQ(int(e.status()),int(SS_E_INVALID));}
  overlap.set(nullptr);
  CHECK_EQ(small.staging_bytes,old.staging_bytes);  // kept: the reset spare (persist)
  CHECK_EQ(small.load_buffer_bytes,f.model.unit());
  CHECK(small.residents==old.residents);
  CHECK_EQ(small.blocks,old.blocks);
  bytes.set("1");
  try{(void)ss::dg_layout(f.d,p,o); CHECK(false);}
  catch(const ss::Error& e){CHECK_EQ(int(e.status()),int(SS_E_CAPACITY));}
  bytes.set(nullptr);
  for (const char* bad : {"0"," 4","+4","-4","4x"}) {  // digits only, no sign or leading whitespace
    readers.set(bad);
    try{(void)ss::dg_layout(f.d,p,o); CHECK(false);}
    catch(const ss::Error& e){CHECK_EQ(int(e.status()),int(SS_E_INVALID));}
  }
  readers.set("4"); CHECK_EQ(ss::dg_layout(f.d,p,o).load_readers,4u);
  readers.set(nullptr);
  {  // DG_UNIT_SUBREAD_KIB: one request per part by default (0), 1024 = 1 MiB pieces, a multiple of 4 KiB up to 64 MiB
    Env sub("DG_UNIT_SUBREAD_KIB", nullptr);
    CHECK_EQ(ss::dg_layout(f.d,p,o).unit_subread,0ull);
    sub.set("1024"); CHECK_EQ(ss::dg_layout(f.d,p,o).unit_subread,uint64_t(1)<<20);
    sub.set("0"); CHECK_EQ(ss::dg_layout(f.d,p,o).unit_subread,0ull);
    sub.set("256"); CHECK_EQ(ss::dg_layout(f.d,p,o).unit_subread,uint64_t(256)<<10);
    for (const char* bad : {"6","65540","x"," 8","+8","-0"}) {
      sub.set(bad);
      try{(void)ss::dg_layout(f.d,p,o); CHECK(false);}
      catch(const ss::Error& e){CHECK_EQ(int(e.status()),int(SS_E_INVALID));}
    }
  }
  CHECK_EQ(ss::dg_layout(f.d,p,o).disk_queue,0u);  // no disk tier: no reads in flight
  for (uint32_t q : {1u, 16u, 65u}) {  // dg_disk_queue needs the disk tier
    o.dg_disk_queue=q;
    try{(void)ss::dg_layout(f.d,p,o); CHECK(false);}
    catch(const ss::Error& e){CHECK_EQ(int(e.status()),int(SS_E_INVALID));}
  }
  o.dg_disk_queue=0;
}

// The three-tier read pool (dg_disk_queue): reads run concurrently and finish in any order; each wait returns once its
// own read finished, with its time, and rethrows its error; destruction with reads not waited for is safe.
TEST(dg_read_pool_overlaps_and_rethrows) {
  ts::DgReadPool pool(4);
  std::atomic<int> running{0}, most{0};
  std::vector<int> done(8, 0);
  std::vector<void*> h;
  for (int i = 0; i < 8; ++i)
    h.push_back(pool.submit([&, i] {
      const int now = ++running;
      int m = most.load();
      while (now > m && !most.compare_exchange_weak(m, now)) {
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2 * (8 - i)));  // the later ones finish first
      --running;
      done[size_t(i)] = 1;
      if (i == 5) ts::fail(SS_E_IO, "injected read failure");
    }));
  for (int i = 0; i < 8; ++i) {
    bool threw = false;
    uint64_t ns = 0;
    try {
      ns = pool.wait(h[size_t(i)]);
    } catch (const ts::Error& e) {
      threw = e.status() == SS_E_IO;
    }
    CHECK_EQ(done[size_t(i)], 1);
    CHECK_EQ(threw, i == 5);
    if (i != 5) CHECK(ns >= 1000000ull);
  }
  CHECK(most.load() > 1);
  ts::DgReadPool left(2);  // never waited for: the destructor drops the queued ones, waits for the running ones
  for (int i = 0; i < 6; ++i) left.submit([] { std::this_thread::sleep_for(std::chrono::milliseconds(3)); });
}

// Concurrent three-tier reads (dg_disk_queue > 1): every reader reads with its own bounce buffer (no lock is shared;
// only the open source is), and units whose gate / up parts are not sector multiples (DeepSeek's are 3,041,280 bytes)
// land the same bytes and digest from four readers at once. A part whose destination has its file offset's
// misalignment reads its aligned middle in place, only its head and tail sectors through the bounce buffer; any other
// unaligned part goes through it whole. Buffered: every part lands in place.
TEST(dg_read_unit_concurrent_readers_own_bounces) {
  const auto dir = sstest::Fixture::temp_root() / "ss-test-dg-read-concurrent";
  std::filesystem::create_directories(dir);
  const auto file = dir / "experts.bin";
  const int U = 16;
  const uint64_t pb[3] = {14336, 14336, 20480}, al = 4096;  // 3.5, 3.5 and 5 sectors
  uint64_t off[3];
  off[0] = 4096;
  off[1] = off[0] + uint64_t(U) * pb[0];
  off[2] = off[1] + uint64_t(U) * pb[1];
  const uint64_t size = off[2] + uint64_t(U) * pb[2] + 4096;
  {
    std::vector<char> buf(static_cast<size_t>(size));
    for (uint64_t i = 0; i < size; ++i) buf[size_t(i)] = char(sstest::byte_at(9, i));
    std::ofstream f(file, std::ios::binary);
    f.write(buf.data(), std::streamsize(size));
  }
  std::string json = R"({"schema":"strix-stream/desc-v1","engine":"dg-test","unit_align":256,"sources":[{"path":")" +
                     file.generic_string() + R"(","align":4096}],"layers":[{"index":0,"kind":"moe","n_units":)" +
                     std::to_string(U) + R"(,"top_k":4,"parts":[)";
  const char* names[3] = {"gate", "up", "down"};
  for (int p = 0; p < 3; ++p)
    json += std::string(p ? "," : "") + R"({"name":")" + names[p] + R"(","format":"Q4_K","offset":)" +
            std::to_string(off[p]) + R"(,"stride":)" + std::to_string(pb[p]) + R"(,"bytes":)" +
            std::to_string(pb[p]) + "}";
  json += "]}]}";
  const auto d = ss::desc_from_json(ss::json_parse(json));
  const uint64_t unit = pb[0] + pb[1] + pb[2];
  std::vector<std::vector<uint8_t>> expected(U);
  std::vector<uint64_t> digests(U);
  for (int u = 0; u < U; ++u) {
    for (int p = 0; p < 3; ++p)
      for (uint64_t i = 0; i < pb[p]; ++i) expected[size_t(u)].push_back(sstest::byte_at(9, off[p] + uint64_t(u) * pb[p] + i));
    digests[size_t(u)] = ss::xxh64(expected[size_t(u)].data(), expected[size_t(u)].size());
  }
  for (bool direct : {false, true}) {
    ss_options o;
    ss_options_defaults(&o);
    o.direct_io = direct;
    std::vector<std::unique_ptr<ss::SourceFile>> files;
    files.push_back(ss::open_source(d.sources[0].path, o));
    const bool dio = files[0]->direct();
    // the bytes one unit's read bounces: a part in place, its head and tail sectors, or the whole part
    auto bounced_of = [&](int u) {
      uint64_t want = 0, at = 0;
      for (int p = 0; p < 3; ++p, at += pb[p - 1]) {
        const uint64_t F = off[p] + uint64_t(u) * pb[p];
        if (!dio || (F % al == 0 && pb[p] % al == 0 && at % al == 0)) continue;
        if (at % al != F % al) {
          want += pb[p];
          continue;
        }
        const uint64_t lo = (al - F % al) % al, middle = pb[p] > lo ? (pb[p] - lo) / al * al : 0;
        want += pb[p] - middle;
      }
      return want;
    };
    std::atomic<int> bad{0};
    std::atomic<uint64_t> bounced{0}, saved{0};
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r)
      readers.emplace_back([&, r] {
        ss::DgBounce bounce(size_t(1) << 20, size_t(al));
        ss::DgBounce dst{size_t(unit), size_t(al)};  // aligned, as a pinned block
        for (int rep = 0; rep < 6; ++rep)
          for (int n = 0; n < U; ++n) {
            const int u = (r * 5 + rep + n) % U;
            std::memset(dst.p, 0x5a, size_t(unit));
            ss::DgLoadStats st;
            uint64_t digest = 0;
            try {
              ss::dg_read_unit(d, files, 0, u, dst.p, bounce.p, bounce.bytes, true, &digest, &st);
            } catch (...) {
              ++bad;
              continue;
            }
            if (std::memcmp(dst.p, expected[size_t(u)].data(), size_t(unit)) != 0 || digest != digests[size_t(u)] ||
                st.bounce_bytes != bounced_of(u))
              ++bad;
            bounced += st.bounce_bytes;
            uint64_t whole = 0;  // what bouncing every unaligned part whole would have copied
            for (int p = 0, at = 0; p < 3; at += int(pb[p]), ++p) {
              const uint64_t F = off[p] + uint64_t(u) * pb[p];
              if (dio && !(F % al == 0 && pb[p] % al == 0 && uint64_t(at) % al == 0)) whole += pb[p];
            }
            saved += whole - st.bounce_bytes;
          }
      });
    for (std::thread& t : readers) t.join();
    CHECK_EQ(bad.load(), 0);
    if (dio) {
      CHECK(bounced.load() > 0);
      CHECK(saved.load() > 0);  // some middles landed in place
    } else {
      CHECK_EQ(bounced.load(), 0ull);
    }
  }
}

// A short part with the destination's misalignment that ends before its next sector (the last part of a unit, at an
// unaligned end of file): bounced whole, nothing written past the unit, no read past the end of the file.
TEST(dg_read_unit_short_part_without_aligned_middle) {
  const auto dir = sstest::Fixture::temp_root() / "ss-test-dg-read-short-part";
  std::filesystem::create_directories(dir);
  const auto file = dir / "experts.bin";
  const uint64_t al = 4096, b0 = 2816, b1 = 768, off0 = 4096, off1 = 8192 + 2816;  // part 1 at the offset's misalignment
  const uint64_t size = off1 + b1;                                                     // the file ends inside a sector
  {
    std::vector<char> buf(static_cast<size_t>(size));
    for (uint64_t i = 0; i < size; ++i) buf[size_t(i)] = char(sstest::byte_at(11, i));
    std::ofstream f(file, std::ios::binary);
    f.write(buf.data(), std::streamsize(size));
  }
  const std::string json = R"({"schema":"strix-stream/desc-v1","engine":"dg-test","unit_align":256,"sources":[{"path":")" +
                           file.generic_string() +
                           R"(","align":4096}],"layers":[{"index":0,"kind":"moe","n_units":1,"top_k":1,"parts":[)" +
                           R"({"name":"up","format":"Q4_K","offset":)" + std::to_string(off0) + R"(,"stride":)" +
                           std::to_string(b0) + R"(,"bytes":)" + std::to_string(b0) + R"(},{"name":"down","format":)" +
                           R"("Q4_K","offset":)" + std::to_string(off1) + R"(,"stride":)" + std::to_string(b1) +
                           R"(,"bytes":)" + std::to_string(b1) + "}]}]}";
  const auto d = ss::desc_from_json(ss::json_parse(json));
  std::vector<uint8_t> expected;
  for (uint64_t i = 0; i < b0; ++i) expected.push_back(sstest::byte_at(11, off0 + i));
  for (uint64_t i = 0; i < b1; ++i) expected.push_back(sstest::byte_at(11, off1 + i));
  for (bool direct : {false, true}) {
    ss_options o;
    ss_options_defaults(&o);
    o.direct_io = direct;
    std::vector<std::unique_ptr<ss::SourceFile>> files;
    files.push_back(ss::open_source(d.sources[0].path, o));
    ss::DgBounce bounce(size_t(1) << 20, size_t(al));
    ss::DgBounce dst{size_t(2 * al), size_t(al)};  // aligned; past the unit = a guard
    std::memset(dst.p, 0x5a, size_t(2 * al));
    ss::DgLoadStats st;
    uint64_t digest = 0;
    ss::dg_read_unit(d, files, 0, 0, dst.p, bounce.p, bounce.bytes, true, &digest, &st);
    CHECK(std::memcmp(dst.p, expected.data(), expected.size()) == 0);
    CHECK_EQ(digest, ss::xxh64(expected.data(), expected.size()));
    bool guard = true;
    for (size_t i = expected.size(); i < size_t(2 * al); ++i) guard = guard && uint8_t(dst.p[i]) == 0x5a;
    CHECK(guard);
    if (files[0]->direct()) CHECK_EQ(st.bounce_bytes, b0 + b1);  // part 0 unaligned at its offset, part 1 bounced whole
  }
}

// A failed token's report (the CUDA context's ss_end; the text is built here): the device's own error word first, then
// what the host saw, then the poisoning. The CUDA module loading mode by name, and the open's notice when it is lazy
// with gated layers. ss_dg_stats.cuda_module_loading is appended (an older caller's prefix is unchanged).
TEST(dg_failure_report_and_module_loading) {
  CHECK(ss::dg_device_error(ss::kDgOk, 7).empty());
  CHECK(ss::dg_device_error(ss::kDgTimeout, 3) == "a wait timed out at gated layer 3");
  CHECK(ss::dg_device_error(ss::kDgBadId, 1) == "an id was out of range at gated layer 1");
  CHECK(ss::dg_device_error(ss::kDgDesync, 12) == "a step ran outside ss_begin / ss_end (sequence desync) (that "
                                                  "step's seq 12)");
  CHECK(ss::dg_device_error(ss::kDgAbsent, 4) == "a slot lookup met an absent entry");
  CHECK(ss::dg_device_error(ss::kDgHostAbort, 2) == "the host aborted the token at gated layer 2");
  const std::string tail = " (the context is poisoned, the token's outputs must be discarded)";
  const std::string both = ss::dg_token_failure(9, ss::dg_device_error(ss::kDgTimeout, 3),
                                                "the service: device-gated: export flag wait timed out");
  CHECK(both == "device-gated: token 9: the device: a wait timed out at gated layer 3; the service: device-gated: "
                "export flag wait timed out" + tail);
  CHECK(both.find("the device:") < both.find("the service:"));
  CHECK(ss::dg_token_failure(9, "", "the service: x") == "device-gated: token 9: the service: x" + tail);
  CHECK(ss::dg_token_failure(9, "an id was out of range at gated layer 1", "") ==
        "device-gated: token 9: the device: an id was out of range at gated layer 1" + tail);
  CHECK(std::string(ss::dg_module_loading_name(0)) == "unknown");
  CHECK(std::string(ss::dg_module_loading_name(1)) == "eager");
  CHECK(std::string(ss::dg_module_loading_name(2)) == "lazy");
  CHECK(std::string(ss::dg_module_loading_name(7)) == "unknown");
  const std::string notice = ss::dg_module_loading_notice(2, 3);
  CHECK(notice.find("lazy") != std::string::npos && notice.find("CUDA_MODULE_LOADING=EAGER") != std::string::npos);
  CHECK(ss::dg_module_loading_notice(2, 0).empty());
  CHECK(ss::dg_module_loading_notice(1, 3).empty());
  CHECK(ss::dg_module_loading_notice(0, 3).empty());
  CHECK_EQ(offsetof(ss_dg_stats, cuda_module_loading), offsetof(ss_dg_stats, round_cancelled) + 8);
  CHECK_EQ(sizeof(ss_dg_stats), offsetof(ss_dg_stats, reserved1) + 4);
  ss_dg_stats z{};
  CHECK_EQ(z.cuda_module_loading, 0u);  // 0 = unknown (a non-CUDA context, or a driver that does not say)
}

// The overlap (DG_LOAD_OVERLAP_ALLOC): the pinned blocks come from the helper thread while the loader reads. The test's
// allocator stands in for cudaHostAlloc: aligned host memory filled with a pattern, a delay per chunk, an optional
// failure, a count of the live chunks.
namespace {
using SteadyClock = std::chrono::steady_clock;
struct HostChunks {
  std::atomic<int> live{0};
  int delay_ms = 0, fail_at = -1;
  std::atomic<int64_t> last_ns{0};  // when the last chunk was returned
  ss::DgBlockAlloc::Alloc alloc() {
    return [this](size_t c, uint64_t bytes) -> void* {
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
      if (int(c) == fail_at) ss::fail(SS_E_NOMEM, "injected pinned allocation failure");
      void* p = ::operator new(size_t(bytes), std::align_val_t(4096));
      std::memset(p, 0x5a, size_t(bytes));
      ++live;
      last_ns = SteadyClock::now().time_since_epoch().count();
      return p;
    };
  }
  ss::DgBlockAlloc::Release release() {
    return [this](void* p) { ::operator delete(p, std::align_val_t(4096)); --live; };
  }
};
constexpr uint64_t kSpares = 4, kPerChunk = 5;
// Host units in blocks (index * 7) mod total: not in chunk order, so the chunks a batch needs vary.
std::vector<uint64_t> overlap_chunks(Fixture& f, uint64_t* total) {
  uint64_t host = 0;
  for (const auto& u : f.units) host += u.slot < 0;
  *total = host + kSpares;
  int64_t i = 0;
  for (auto& u : f.units)
    if (u.slot < 0) { u.host = nullptr; u.block = (i++ * 7) % int64_t(*total); }
  std::vector<uint64_t> chunks;
  for (uint64_t left = *total; left;) { chunks.push_back(std::min(kPerChunk, left)); left -= chunks.back(); }
  return chunks;
}
struct OverlapRun {
  std::vector<uint64_t> blocks, slots;
  ss::DgLoadStats stats;
  int64_t last_resident_ns = 0, last_chunk_ns = 0;
};
// overlap false: every chunk allocated before the loader starts (the serial order), the units given their pointers.
OverlapRun overlap_run(bool overlap, bool direct, uint32_t readers) {
  Fixture f;
  const uint64_t unit = f.model.unit(), stride = (unit + 4095) / 4096 * 4096;
  uint64_t total = 0;
  const auto chunks = overlap_chunks(f, &total);
  HostChunks host;
  host.delay_ms = overlap ? 20 : 0;
  OverlapRun r;
  std::vector<void*> owned;
  std::vector<char*> ptrs;
  {
    ss::DgBlockAlloc blocks(chunks, stride, host.alloc(), host.release());
    blocks.start();
    if (!overlap) {
      blocks.finish(&owned, &ptrs);
      for (auto& u : f.units) if (u.slot < 0) u.host = ptrs[size_t(u.block)];
    }
    size_t slots = 0;
    for (const auto& u : f.units) slots += u.slot >= 0;
    std::vector<char> device(size_t(slots * unit), char(0x33));
    const uint64_t capacity = unit * 4;
    std::vector<std::vector<char>> storage(2, std::vector<char>(size_t(capacity + 8192)));
    std::vector<char*> buffers{storage[0].data() + 123, storage[1].data() + 123};
    r.stats = ss::dg_load_pipeline(f.d, f.units, buffers, capacity, readers, direct, true, [](size_t) {},
        [&](size_t, const std::vector<ss::DgLoadCopy>& copies) {
          for (const auto& c : copies) {
            uint64_t off = 0;
            for (size_t p = 0; p < c.part; ++p) off += f.d.layers[0].parts[p].bytes;
            std::memcpy(device.data() + uint64_t(c.slot) * unit + off, c.data, size_t(c.bytes));
          }
          r.last_resident_ns = SteadyClock::now().time_since_epoch().count();
        }, nullptr, overlap ? &blocks : nullptr);
    if (overlap) blocks.finish(&owned, &ptrs);
    CHECK_EQ(ptrs.size(), size_t(total));
    for (char* b : ptrs) r.blocks.push_back(ss::xxh64(b, size_t(stride)));
    for (size_t s = 0; s < slots; ++s) r.slots.push_back(ss::xxh64(device.data() + s * unit, size_t(unit)));
  }
  r.last_chunk_ns = host.last_ns;
  CHECK_EQ(owned.size(), chunks.size());
  for (void* p : owned) host.release()(p);
  CHECK_EQ(host.live.load(), 0);
  return r;
}
}  // namespace

TEST(dg_load_overlap_alloc_matches_serial) {
  for (bool direct : {false, true}) {
    for (uint32_t readers : {1u, 4u, 8u}) {
      const OverlapRun serial = overlap_run(false, direct, readers), overlap = overlap_run(true, direct, readers);
      // the same bytes in every block (the spares untouched) and every slot
      CHECK(serial.blocks == overlap.blocks);
      CHECK(serial.slots == overlap.slots);
      CHECK_EQ(serial.stats.payload_bytes, overlap.stats.payload_bytes);
      // the residents were read (and handed to the device copies) while the chunks were still being allocated
      CHECK(overlap.last_resident_ns > 0 && overlap.last_resident_ns < overlap.last_chunk_ns);
      CHECK(overlap.stats.resident_seconds > 0 && overlap.stats.resident_seconds < overlap.stats.read_seconds);
      CHECK(overlap.stats.alloc_wait_seconds > 0);  // then the readers waited for the blocks
      CHECK_EQ(serial.stats.alloc_wait_seconds, 0.0);
    }
  }
}

TEST(dg_load_overlap_alloc_failure_frees_everything) {
  for (int fail_at : {0, 4}) {
    for (bool finish : {false, true}) {
      Fixture f;
      uint64_t total = 0;
      const auto chunks = overlap_chunks(f, &total);
      HostChunks host;
      host.delay_ms = 5;
      host.fail_at = fail_at;
      {
        const uint64_t stride = (f.model.unit() + 4095) / 4096 * 4096;
        ss::DgBlockAlloc blocks(chunks, stride, host.alloc(), host.release());
        blocks.start();
        std::vector<char> ring(size_t(f.model.unit() * 2));
        int status = 0;
        try {
          ss::dg_load_pipeline(f.d, f.units, {ring.data()}, ring.size(), 4, false, true, [](size_t) {},
                               [](size_t, const std::vector<ss::DgLoadCopy>&) {}, nullptr, &blocks);
        } catch (const ss::Error& e) {
          status = int(e.status());
        }
        CHECK_EQ(status, int(SS_E_NOMEM));  // the allocator's error, as when the blocks came first
        if (finish) {
          std::vector<void*> owned;
          std::vector<char*> ptrs;
          status = 0;
          try {
            blocks.finish(&owned, &ptrs);
          } catch (const ss::Error& e) {
            status = int(e.status());
          }
          CHECK_EQ(status, int(SS_E_NOMEM));
          CHECK_EQ(owned.size(), size_t(fail_at));  // what was allocated is the caller's to free
          CHECK_EQ(ptrs.size(), size_t(fail_at) * size_t(kPerChunk));
          for (void* p : owned) host.release()(p);
        }
      }
      CHECK_EQ(host.live.load(), 0);  // nothing leaked
    }
  }
  // destroyed while the helper still allocates: it stops after the chunk in progress, every chunk freed
  Fixture f;
  uint64_t total = 0;
  const auto chunks = overlap_chunks(f, &total);
  HostChunks host;
  host.delay_ms = 10;
  {
    ss::DgBlockAlloc blocks(chunks, 8192, host.alloc(), host.release());
    blocks.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  CHECK_EQ(host.live.load(), 0);
}

int main(int argc,char** argv) {return sstest::run_all(argc>1?argv[1]:nullptr);}
