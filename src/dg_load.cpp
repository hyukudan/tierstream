#include "dg_load.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <thread>
#include "errors.hpp"
#include "source_file.hpp"
#include "xxh64.hpp"
namespace ts {
namespace {
struct Span { size_t source; uint64_t offset, bytes; char* dst; uint64_t room; };
struct Batch { size_t first, count; bool resident; size_t need; };  // need: chunks published before a host read
struct Ready { size_t buffer; std::vector<DgLoadCopy> copies; };
// Pageable aligned bounce storage: never an H2D source. Each reader owns one.
struct Bounce {
  char* p; size_t alignment;
  Bounce(size_t n, size_t a) : p(static_cast<char*>(::operator new(n, std::align_val_t(a)))), alignment(a) {}
  ~Bounce() { ::operator delete(p, std::align_val_t(alignment)); }
};
}
DgLoadStats dg_load_pipeline(const Desc& d, const std::vector<DgLoadUnit>& input,
    const std::vector<char*>& buffers, uint64_t buffer_bytes, uint32_t readers,
    bool direct, bool verify, const std::function<void(size_t)>& recycle,
    const std::function<void(size_t, const std::vector<DgLoadCopy>&)>& submit,
    const std::vector<std::unique_ptr<SourceFile>>* sources, DgBlockAlloc* blocks) {
  const auto start = std::chrono::steady_clock::now();
  require(!buffers.empty() && readers > 0, SS_E_INVALID, "startup loader: empty pipeline");
  require(!sources || sources->size() == d.sources.size(), SS_E_INVALID, "startup loader: source count differs");
  const bool covering = direct || (sources && std::any_of(sources->begin(), sources->end(),
      [](const std::unique_ptr<SourceFile>& f) { return f->direct(); }));
  for (const auto& u : input)
    require(u.slot >= 0 || u.host || (blocks && u.block >= 0 && uint64_t(u.block) < blocks->total()),
            SS_E_INVALID, "startup loader: a host unit without a home");
  // Reorder I/O only: each unit retains its original slot and host home.
  auto units = input;
  std::stable_sort(units.begin(), units.end(), [](const DgLoadUnit& a, const DgLoadUnit& b) {
    if (a.layer != b.layer) return a.layer < b.layer;
    if ((a.slot >= 0) != (b.slot >= 0)) return a.slot >= 0;
    return a.unit < b.unit;
  });
  std::vector<Batch> batches;
  for (size_t i = 0; i < units.size();) {
    const auto& u = units[i];
    const uint64_t bytes = d.layers[size_t(u.layer)].unit_bytes;
    require(bytes && buffer_bytes >= bytes, SS_E_CAPACITY, "startup loader: buffer smaller than one expert");
    uint64_t slack = 0;
    for (const auto& p : d.layers[size_t(u.layer)].parts) slack += 3*std::max<uint64_t>(4096, d.sources[size_t(p.source)].align);
    // Reserve worst-case covering slack per range; a one-unit tight ring still
    // works via the fallback. Host batching does not consume ring storage.
    const size_t cap = size_t(std::max<uint64_t>(1, buffer_bytes / (bytes + (covering && u.slot >= 0 ? slack : 0))));
    size_t n = 1;
    while (n < cap && i + n < units.size() && units[i+n].layer == u.layer &&
           (units[i+n].slot >= 0) == (u.slot >= 0)) ++n;
    size_t need = 0;
    for (size_t j = i; j < i + n; ++j)
      if (units[j].slot < 0 && !units[j].host) need = std::max(need, blocks->chunk_of(uint64_t(units[j].block)) + 1);
    batches.push_back({i, n, u.slot >= 0, need});
    i += n;
  }
  std::mutex mu;
  std::condition_variable cv;
  std::deque<size_t> free;
  for (size_t i = 0; i < buffers.size(); ++i) free.push_back(i);
  std::deque<Ready> ready;
  std::deque<Batch> host_batches;
  for (const auto& b : batches) if (!b.resident) host_batches.push_back(b);
  batches.erase(std::remove_if(batches.begin(), batches.end(), [](const Batch& b){ return !b.resident; }), batches.end());
  size_t next = 0, live = 0, alloc_waiters = 0;
  bool stop = false;
  std::exception_ptr error;
  DgLoadStats stats;
  auto alloc_wait_start = start;
  // The helper's publishes wake the readers (under the allocator's lock, then ours: readers never take its lock).
  struct Subscription {
    DgBlockAlloc* b;
    ~Subscription() { if (b) b->subscribe(nullptr); }
  } subscription{blocks};
  if (blocks) blocks->subscribe([&] { { std::lock_guard<std::mutex> lk(mu); } cv.notify_all(); });
  // the first host batch whose chunks are published (all are without an allocator)
  auto host_ready = [&]() -> size_t {
    const size_t r = blocks ? blocks->ready() : 0;
    for (size_t i = 0; i < host_batches.size(); ++i)
      if (host_batches[i].need <= r) return i;
    return host_batches.size();
  };
  auto worker = [&] {
    try {
      std::vector<std::unique_ptr<SourceFile>> files;
      uint64_t alignment = 4096;
      for (size_t i = 0; i < d.sources.size(); ++i) {
        const auto& src = d.sources[i];
        // a worker's own handle on the same file as the context's open source (bound, identity checked)
        files.push_back(open_source(src.path, direct, sources ? (*sources)[i]->session() : nullptr,
                                    sources ? (*sources)[i]->identity() : FileIdentity()));
        alignment = std::max(alignment, src.align);
      }
      const uint64_t bounce_bytes = std::max<uint64_t>(1ull << 20, 4 * alignment);
      std::unique_ptr<Bounce> bounce; // allocated only if a destination cannot hold covering-read slack
      for (;;) {
        Batch b{}; size_t bi = 0;
        bool done = false;
        {
          std::unique_lock<std::mutex> lk(mu);
          for (;;) {
            if (stop) { done = true; break; }
            if (blocks && blocks->failed()) std::rethrow_exception(blocks->error());
            if (next < batches.size() && !free.empty()) {  // resident work first
              b = batches[next++]; bi = free.front(); free.pop_front(); break;
            }
            const size_t h = host_ready();
            if (h < host_batches.size()) {
              b = host_batches[h]; host_batches.erase(host_batches.begin() + ptrdiff_t(h)); break;
            }
            if (next == batches.size() && host_batches.empty()) { done = true; break; }
            // no resident batch left to take: this reader waits for a chunk
            const bool for_blocks = next == batches.size();
            if (for_blocks && alloc_waiters++ == 0) alloc_wait_start = std::chrono::steady_clock::now();
            cv.wait(lk);
            if (for_blocks && --alloc_waiters == 0)
              stats.alloc_wait_seconds +=
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - alloc_wait_start).count();
          }
        }
        if (done) break;
        if (b.resident) recycle(bi); // event must complete before any byte in the buffer is changed
        const Layer& layer = d.layers[size_t(units[b.first].layer)];
        std::vector<Span> spans;
        std::vector<DgLoadCopy> copies;
        uint64_t part_base = 0;
        for (size_t p = 0; p < layer.parts.size(); ++p) {
          const Part& part = layer.parts[p];
          for (size_t j = 0; j < b.count; ++j) {
            const auto& u = units[b.first+j];
            char* home = b.resident ? nullptr : u.host ? u.host : blocks->block(uint64_t(u.block));
            char* dst = b.resident ? buffers[bi] + part_base * b.count + j * part.bytes : home + part_base;
            spans.push_back({size_t(part.source), part.offset + uint64_t(u.unit) * part.stride, part.bytes, dst, b.resident ? 0 : layer.unit_bytes-part_base});
            if (b.resident) copies.push_back({u.slot, p, dst, part.bytes});
          }
          part_base += part.bytes;
        }
        std::sort(spans.begin(), spans.end(), [&](const Span& a, const Span& rhs) {
          // Host covering reads may overwrite later, unread parts in this home.
          // Visit destination order and compact immediately; never touch a prior part.
          if (!b.resident) return uintptr_t(a.dst) < uintptr_t(rhs.dst);
          return a.source < rhs.source || (a.source == rhs.source && a.offset < rhs.offset);
        });
        // Repack coalesced resident ranges at aligned addresses with prefix/suffix
        // slack. Copy/hash pointers address only the payload, never the padding.
        // A tightly admitted ring may lack slack; retain the compact fallback.
        for (const auto& a : spans) {
          const auto size = files[a.source]->size();
          require(a.offset <= size && a.bytes <= size-a.offset, SS_E_IO, "startup loader: source range past EOF");
        }
        std::vector<Span> ranges;
        for (const auto& span : spans) {
          if (!ranges.empty() && ranges.back().source == span.source &&
              ranges.back().offset + ranges.back().bytes == span.offset &&
              ranges.back().dst + ranges.back().bytes == span.dst &&
              ranges.back().room == 0 && span.room == 0)
            ranges.back().bytes += span.bytes;
          else ranges.push_back(span);
        }
        if (b.resident && covering) {
          uint64_t cursor = 0;
          std::vector<char*> destinations;
          for (const auto& a : ranges) {
            const uint64_t al = files[a.source]->alignment(std::max<uint64_t>(4096, d.sources[a.source].align));
            const uint64_t pad = (al - (uintptr_t(buffers[bi]) + cursor) % al) % al;
            const uint64_t delta = a.offset % al;
            const uint64_t extent = (delta + a.bytes + al - 1) / al * al;
            if (cursor > buffer_bytes || pad + extent > buffer_bytes - cursor) break;
            destinations.push_back(buffers[bi] + cursor + pad + delta);
            cursor += pad + extent;
          }
          if (destinations.size() == ranges.size()) {
            for (auto& c : copies) {
              for (size_t i = 0; i < ranges.size(); ++i) {
                const auto& a = ranges[i];
                if (c.data >= a.dst && c.data < a.dst + a.bytes) {
                  c.data = destinations[i] + (c.data - a.dst); break;
                }
              }
            }
            for (size_t i = 0; i < ranges.size(); ++i) ranges[i].dst = destinations[i];
          } else destinations.clear();
          // Nonempty means every covering range fits inside this owned buffer.
          spans = std::move(ranges);
          if (!destinations.empty()) {
            for (auto& a : spans) {
              const uint64_t al = files[a.source]->alignment(std::max<uint64_t>(4096, d.sources[a.source].align));
              const uint64_t delta = a.offset % al;
              a.dst -= delta;
              a.bytes = (delta + a.bytes + al - 1) / al * al;
              a.offset -= delta;
            }
          }
        } else spans = std::move(ranges);
        uint64_t actual = 0, calls = 0, bounced = 0;
        for (size_t s = 0; s < spans.size();) {
          Span a = spans[s++];
          // Merge only when both disk range and destination are adjacent. Host homes
          // stay in place; resident batches are part-major to coalesce stacked tensors.
          while (s < spans.size() && spans[s].source == a.source && spans[s].offset == a.offset + a.bytes &&
                 spans[s].dst == a.dst + a.bytes && a.room == 0 && spans[s].room == 0) a.bytes += spans[s++].bytes;
          const auto& file = *files[a.source];
          require(a.offset < file.size(), SS_E_IO, "startup loader: source range past EOF");
          const uint64_t al = files[a.source]->alignment(std::max<uint64_t>(4096, d.sources[a.source].align));
          auto read = [&](void* dst, uint64_t n, uint64_t off, uint64_t need) {
            const uint64_t got = file.pread(dst,n,off);
            actual += got; ++calls;
            require(got >= need, SS_E_IO, "startup loader: short read");
          };
          if (!file.direct() || (a.offset % al == 0 && a.bytes % al == 0 && uintptr_t(a.dst) % al == 0))
            read(a.dst, a.bytes, a.offset, std::min(a.bytes, file.size()-a.offset));
          else if (a.room &&
                   (al - uintptr_t(a.dst)%al)%al + (a.offset%al+a.bytes+al-1)/al*al <= a.room) {
            const uint64_t delta = a.offset%al;
            char* aligned = a.dst + (al - uintptr_t(a.dst)%al)%al;
            const uint64_t n = (delta+a.bytes+al-1)/al*al;
            read(aligned, n, a.offset-delta, delta+a.bytes);
            std::memmove(a.dst, aligned+delta, size_t(a.bytes));
          } else {
            require(a.bytes <= file.size()-a.offset, SS_E_IO, "startup loader: source range past EOF");
            if (!bounce) bounce = std::make_unique<Bounce>(size_t(bounce_bytes), size_t(alignment));
            bounced += a.bytes;
            const uint64_t cap = (bounce_bytes - 2*al) / al * al;
            for (uint64_t c = 0; c < a.bytes;) {
              const uint64_t n = std::min(cap, a.bytes-c);
              const uint64_t off = (a.offset+c)/al*al, delta = a.offset+c-off;
              const uint64_t end = (a.offset+c+n+al-1)/al*al;
              read(bounce->p, end-off, off, delta+n);
              std::memcpy(a.dst+c, bounce->p+delta, size_t(n)); c += n;
            }
          }
        }
        const auto hash_start = std::chrono::steady_clock::now();
        if (verify && !layer.hash.empty()) {
          for (size_t j = 0; j < b.count; ++j) {
            Xxh64 hash; uint64_t base = 0;
            for (size_t p = 0; p < layer.parts.size(); ++p) {
              const auto& part = layer.parts[p];
              const auto& u = units[b.first+j];
              const char* src = b.resident ? copies[p*b.count+j].data
                                           : (u.host ? u.host : blocks->block(uint64_t(u.block)))+base;
              hash.update(src, size_t(part.bytes)); base += part.bytes;
            }
            require(hash.digest() == layer.hash[size_t(units[b.first+j].unit)], SS_E_HASH,
                    "startup loader: expert bytes differ from the descriptor hash");
          }
        }
        {
          std::lock_guard<std::mutex> lk(mu);
          stats.payload_bytes += uint64_t(b.count)*layer.unit_bytes;
          stats.read_bytes += actual; stats.read_calls += calls;
          stats.bounce_bytes += bounced;
          stats.hash_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-hash_start).count();
          stats.read_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
          if (b.resident) {
            stats.resident_seconds = stats.read_seconds;
            ready.push_back({bi, std::move(copies)});
          }
        }
        cv.notify_all();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lk(mu);
      if (!error) error = std::current_exception();
      stop = true;
    }
    { std::lock_guard<std::mutex> lk(mu); --live; }
    cv.notify_all();
  };
  std::vector<std::thread> threads;
  auto join = [&] { for (auto& t : threads) if (t.joinable()) t.join(); };
  try {
    // Keep live stable during construction: a fast worker can finish immediately.
    for (uint32_t i = 0; i < readers; ++i) {
      std::lock_guard<std::mutex> lk(mu);
      threads.emplace_back(worker); ++live;
    }
    for (;;) {
      Ready r;
      {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&]{return error || !ready.empty() || live == 0;});
        if (error) std::rethrow_exception(error);
        if (ready.empty() && live == 0) break;
        r = std::move(ready.front()); ready.pop_front();
      }
      submit(r.buffer, r.copies);
      { std::lock_guard<std::mutex> lk(mu); free.push_back(r.buffer); }
      cv.notify_all();
    }
  } catch (...) {
    { std::lock_guard<std::mutex> lk(mu); stop = true; }
    cv.notify_all(); join(); throw;
  }
  join();
  return stats;
}

void dg_read_unit(const Desc& d, const std::vector<std::unique_ptr<SourceFile>>& files, int layer, int unit,
                  void* dst, void* bounce, uint64_t bounce_bytes, bool hash, uint64_t* digest, DgLoadStats* stats,
                  std::mutex* bounce_mu, bool parallel, uint64_t subread) {
  const Layer& l = d.layers[size_t(layer)];
  auto read = [&](const SourceFile& file, void* out, uint64_t n, uint64_t off) {
    const uint64_t got = file.pread(out, n, off);
    if (stats) { stats->read_bytes += got; ++stats->read_calls; }
    return got;
  };
  auto out_of = [&](size_t i) {
    uint64_t pos = 0;
    for (size_t j = 0; j < i; ++j) pos += l.parts[j].bytes;
    return static_cast<char*>(dst) + pos;
  };
  // parallel: the in-place parts of each multi-location session, issued together after the others (a session with a
  // single such part reads it as before)
  struct Batch { MFile* file; std::vector<ss_mfile_io> ios; std::vector<size_t> parts; };
  std::vector<Batch> batches;
  std::vector<uint8_t> batched(l.parts.size(), 0);
  for (size_t i = 0; parallel && i < l.parts.size(); ++i) {
    const Part& part = l.parts[i];
    const SourceFile& f = *files[size_t(part.source)];
    MFile* m = f.session().get();
    if (!m) continue;
    const uint64_t al = f.alignment(d.sources[size_t(part.source)].align);
    const uint64_t F = part.offset + uint64_t(unit) * part.stride;
    char* out = out_of(i);
    if (F + part.bytes > f.size() || F % al != 0 || part.bytes % al != 0 || uintptr_t(out) % al != 0) continue;
    auto it = std::find_if(batches.begin(), batches.end(), [&](const Batch& b) { return b.file == m; });
    if (it == batches.end()) it = batches.insert(batches.end(), Batch{m, {}, {}});
    it->ios.push_back({out, F, part.bytes});
    it->parts.push_back(i);
  }
  for (const Batch& b : batches)
    if (b.parts.size() >= 2)
      for (size_t i : b.parts) batched[i] = 1;
  // subread: every in-place part as pieces, per source, all in flight together (replaces the batches above)
  struct Cut { const SourceFile* file; std::vector<RawFile::Piece> pieces; };
  std::vector<Cut> cuts;
  for (size_t i = 0; subread && i < l.parts.size(); ++i) {
    const Part& part = l.parts[i];
    const SourceFile& f = *files[size_t(part.source)];
    const uint64_t al = f.alignment(d.sources[size_t(part.source)].align);
    const uint64_t F = part.offset + uint64_t(unit) * part.stride;
    char* out = out_of(i);
    const bool in_place = !f.direct() || (F % al == 0 && part.bytes % al == 0 && uintptr_t(out) % al == 0);
    if (!in_place || F + part.bytes > f.size()) {
      batched[i] = 0;
      continue;
    }
    auto it = std::find_if(cuts.begin(), cuts.end(), [&](const Cut& c) { return c.file == &f; });
    if (it == cuts.end()) it = cuts.insert(cuts.end(), Cut{&f, {}});
    // pieces of `subread` bytes from the part's start (aligned: subread is a multiple of 4 KiB), cut at the grid
    const uint64_t grid = f.piece_grid();
    for (uint64_t at = 0; at < part.bytes;) {
      uint64_t len = std::min(subread, part.bytes - at);
      if (grid) len = std::min(len, grid - (F + at) % grid);
      it->pieces.push_back({out + at, F + at, len});
      at += len;
    }
    batched[i] = 2;
  }
  if (subread) batches.clear();
  for (size_t i = 0; i < l.parts.size(); ++i) {
    const Part& part = l.parts[i];
    const SourceFile& f = *files[size_t(part.source)];
    const uint64_t al = f.alignment(d.sources[size_t(part.source)].align);
    const uint64_t F = part.offset + uint64_t(unit) * part.stride;
    char* out = out_of(i);
    require(F + part.bytes <= f.size(), SS_E_IO, "unit (" + std::to_string(layer) + ", " + std::to_string(unit) +
                                                     ") part " + part.name + " is past the end of its source");
    if (batched[i]) continue;
    const bool in_place = !f.direct() || (F % al == 0 && part.bytes % al == 0 && uintptr_t(out) % al == 0);
    if (in_place) {
      require(read(f, out, part.bytes, F) == part.bytes, SS_E_IO, "short read");
    } else {
      if (stats) stats->bounce_bytes += part.bytes;
      std::unique_lock<std::mutex> lk;
      if (bounce_mu) lk = std::unique_lock<std::mutex>(*bounce_mu);
      const uint64_t cap = (bounce_bytes - 2 * al) / al * al;
      for (uint64_t c = 0; c < part.bytes;) {
        const uint64_t len = std::min(cap, part.bytes - c);
        const uint64_t a0 = (F + c) / al * al, delta = F + c - a0, a1 = (F + c + len + al - 1) / al * al;
        require(read(f, bounce, a1 - a0, a0) >= delta + len, SS_E_IO, "short read");
        std::memcpy(out + c, static_cast<char*>(bounce) + delta, size_t(len));
        c += len;
      }
    }
  }
  for (const Cut& c : cuts) {  // every piece lands or it throws (after none is in flight)
    c.file->read_pieces(c.pieces);
    if (stats) {
      for (const RawFile::Piece& p : c.pieces) stats->read_bytes += p.bytes;
      stats->pieces += c.pieces.size();
      ++stats->read_calls;
    }
  }
  for (const Batch& b : batches) {  // aligned and inside the file (checked above): every byte lands or it throws
    if (b.parts.size() < 2) continue;
    b.file->preadv(b.ios.data(), uint32_t(b.ios.size()));
    if (stats) {
      for (const ss_mfile_io& io : b.ios) stats->read_bytes += io.bytes;
      ++stats->read_calls;
    }
  }
  if (hash) {  // the parts in order, as each landed: the same digest
    const auto hash_start = std::chrono::steady_clock::now();
    Xxh64 h;
    for (size_t i = 0; i < l.parts.size(); ++i) h.update(out_of(i), size_t(l.parts[i].bytes));
    if (stats)
      stats->hash_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - hash_start).count();
    if (digest) *digest = h.digest();
  } else if (digest) {
    *digest = Xxh64().digest();
  }
}

DgBlockAlloc::DgBlockAlloc(std::vector<uint64_t> chunk_blocks, uint64_t stride, Alloc alloc, Release release)
    : counts_(std::move(chunk_blocks)), stride_(stride), alloc_(std::move(alloc)), release_(std::move(release)) {
  uint64_t first = 0;
  for (uint64_t n : counts_) {
    require(n > 0, SS_E_INVALID, "startup loader: an empty block chunk");
    first_.push_back(first);
    first += n;
  }
  ptrs_.assign(counts_.size(), nullptr);
}

DgBlockAlloc::~DgBlockAlloc() {
  cancel();
  if (thread_.joinable()) thread_.join();
  if (!handed_)
    for (size_t c = 0; c < ready(); ++c) release_(ptrs_[c]);
}

void DgBlockAlloc::start() {
  require(!thread_.joinable() && !handed_, SS_E_INVALID, "startup loader: block allocator started twice");
  thread_ = std::thread([this] { run(); });
}

void DgBlockAlloc::cancel() { cancel_.store(true, std::memory_order_release); }

uint64_t DgBlockAlloc::total() const { return first_.empty() ? 0 : first_.back() + counts_.back(); }

size_t DgBlockAlloc::chunk_of(uint64_t b) const {
  return size_t(std::upper_bound(first_.begin(), first_.end(), b) - first_.begin()) - 1;
}

char* DgBlockAlloc::block(uint64_t b) const {
  const size_t c = chunk_of(b);
  return static_cast<char*>(ptrs_[c]) + (b - first_[c]) * stride_;
}

void DgBlockAlloc::subscribe(std::function<void()> notify) {
  std::lock_guard<std::mutex> lk(mu_);
  notify_ = std::move(notify);
}

void DgBlockAlloc::run() {
  using Clock = std::chrono::steady_clock;
  for (size_t c = 0; c < counts_.size() && !cancel_.load(std::memory_order_acquire); ++c) {
    const auto t = Clock::now();
    void* p = nullptr;
    try {
      p = alloc_(c, counts_[c] * stride_);
    } catch (...) {
      seconds_ += std::chrono::duration<double>(Clock::now() - t).count();
      std::lock_guard<std::mutex> lk(mu_);
      error_ = std::current_exception();
      failed_.store(true, std::memory_order_release);
      if (notify_) notify_();
      break;
    }
    seconds_ += std::chrono::duration<double>(Clock::now() - t).count();
    std::lock_guard<std::mutex> lk(mu_);
    ptrs_[c] = p;
    ready_.store(c + 1, std::memory_order_release);
    if (notify_) notify_();
  }
  done_at_ = Clock::now();
  std::lock_guard<std::mutex> lk(mu_);
  if (notify_) notify_();
}

void DgBlockAlloc::finish(std::vector<void*>* out_chunks, std::vector<char*>* out_blocks) {
  if (thread_.joinable()) thread_.join();
  if (!handed_) {  // reserved first: nothing below can fail once a chunk is handed over
    const size_t n = ready();
    out_chunks->reserve(out_chunks->size() + n);
    out_blocks->reserve(out_blocks->size() + size_t(n ? first_[n - 1] + counts_[n - 1] : 0));
    for (size_t c = 0; c < n; ++c) {
      out_chunks->push_back(ptrs_[c]);
      for (uint64_t i = 0; i < counts_[c]; ++i) out_blocks->push_back(static_cast<char*>(ptrs_[c]) + i * stride_);
    }
    handed_ = true;
  }
  if (failed()) std::rethrow_exception(error_);
}

DgReadPool::DgReadPool(uint32_t threads) {
  try {
    for (uint32_t i = 0; i < std::max<uint32_t>(threads, 1); ++i) threads_.emplace_back([this] { work(); });
  } catch (...) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
    throw;
  }
}

DgReadPool::~DgReadPool() {
  {
    std::unique_lock<std::mutex> lk(mu_);
    queue_.clear();  // never started: nothing writes for them
    stop_ = true;
    done_cv_.wait(lk, [&] { return running_ == 0; });
  }
  cv_.notify_all();
  for (auto& t : threads_) t.join();
}

void DgReadPool::work() {
  for (;;) {
    Job* j = nullptr;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [&] { return stop_ || !queue_.empty(); });
      if (stop_) return;
      j = queue_.front();
      queue_.pop_front();
      ++running_;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::exception_ptr error;
    try {
      j->read();
    } catch (...) {
      error = std::current_exception();
    }
    const uint64_t ns = uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    {
      std::lock_guard<std::mutex> lk(mu_);
      j->error = error;
      j->ns = ns;
      j->done = true;
      --running_;
    }
    done_cv_.notify_all();
  }
}

void* DgReadPool::submit(std::function<void()> read) {
  auto j = std::make_unique<Job>();
  j->read = std::move(read);
  Job* p = j.get();
  {
    std::lock_guard<std::mutex> lk(mu_);
    jobs_.reserve(jobs_.size() + 1);  // nothing can fail once the job is queued
    queue_.push_back(p);
    jobs_.push_back(std::move(j));
  }
  cv_.notify_one();
  return p;
}

bool DgReadPool::done(void* handle) {
  std::lock_guard<std::mutex> lk(mu_);
  return static_cast<Job*>(handle)->done;
}

uint64_t DgReadPool::wait(void* handle) {
  Job* p = static_cast<Job*>(handle);
  std::unique_ptr<Job> j;
  {
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [&] { return p->done; });
    auto it = std::find_if(jobs_.begin(), jobs_.end(), [&](const std::unique_ptr<Job>& x) { return x.get() == p; });
    j = std::move(*it);
    jobs_.erase(it);
  }
  if (j->error) std::rethrow_exception(j->error);
  return j->ns;
}
}  // namespace ts
