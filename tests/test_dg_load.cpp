#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <mutex>
#include <new>
#include <algorithm>
#include <vector>
#include "check.hpp"
#include "dg_fixture.hpp"
#include "dg_load.hpp"
#include "dg_layout.hpp"
#include "errors.hpp"
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
  serial.set("yes");
  try{(void)ss::dg_layout(f.d,p,o); CHECK(false);}
  catch(const ss::Error& e){CHECK_EQ(int(e.status()),int(SS_E_INVALID));}
  serial.set("1");
  p.budget.host_pinned_bytes=p.pinned+old.mapped_bytes+old.bounce_bytes+old.staging_bytes+4*f.model.unit();
  serial.set("0");
  const auto small=ss::dg_layout(f.d,p,o);
  CHECK_EQ(small.load_readers,8u);
  CHECK(small.load_overlap_alloc);  // default on (pipeline only)
  CHECK(!old.load_overlap_alloc);
  overlap.set("0"); CHECK(!ss::dg_layout(f.d,p,o).load_overlap_alloc);
  for (const char* value : {"1","true","on"}) { overlap.set(value); CHECK(ss::dg_layout(f.d,p,o).load_overlap_alloc); }
  overlap.set("off");
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
  bytes.set(nullptr); readers.set("0");
  try{(void)ss::dg_layout(f.d,p,o); CHECK(false);}
  catch(const ss::Error& e){CHECK_EQ(int(e.status()),int(SS_E_INVALID));}
  readers.set(nullptr);
  {  // DG_UNIT_SUBREAD_KIB: one request per part by default (0), 1024 = 1 MiB pieces, a multiple of 4 KiB up to 64 MiB
    Env sub("DG_UNIT_SUBREAD_KIB", nullptr);
    CHECK_EQ(ss::dg_layout(f.d,p,o).unit_subread,0ull);
    sub.set("1024"); CHECK_EQ(ss::dg_layout(f.d,p,o).unit_subread,uint64_t(1)<<20);
    sub.set("0"); CHECK_EQ(ss::dg_layout(f.d,p,o).unit_subread,0ull);
    sub.set("256"); CHECK_EQ(ss::dg_layout(f.d,p,o).unit_subread,uint64_t(256)<<10);
    for (const char* bad : {"6","65540","x"}) {
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
