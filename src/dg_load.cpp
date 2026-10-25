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
#include "file_reader.hpp"
#include "xxh64.hpp"
namespace ts {
namespace {
struct Span { size_t source; uint64_t offset, bytes; char* dst; uint64_t room; };
struct Batch { size_t first, count; bool resident; };
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
    const std::function<void(size_t, const std::vector<DgLoadCopy>&)>& submit) {
  const auto start = std::chrono::steady_clock::now();
  require(!buffers.empty() && readers > 0, SS_E_INVALID, "startup loader: empty pipeline");
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
    const size_t cap = size_t(std::max<uint64_t>(1, buffer_bytes / (bytes + (direct && u.slot >= 0 ? slack : 0))));
    size_t n = 1;
    while (n < cap && i + n < units.size() && units[i+n].layer == u.layer &&
           (units[i+n].slot >= 0) == (u.slot >= 0)) ++n;
    batches.push_back({i, n, u.slot >= 0});
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
  size_t next = 0, live = 0;
  bool stop = false;
  std::exception_ptr error;
  DgLoadStats stats;
  auto worker = [&] {
    try {
      std::vector<std::unique_ptr<File>> files;
      uint64_t alignment = 4096;
      for (const auto& src : d.sources) {
        files.push_back(std::make_unique<File>(src.path, direct));
        alignment = std::max(alignment, src.align);
      }
      const uint64_t bounce_bytes = std::max<uint64_t>(1ull << 20, 4 * alignment);
      std::unique_ptr<Bounce> bounce; // allocated only if a destination cannot hold covering-read slack
      for (;;) {
        Batch b{}; size_t bi = 0;
        {
          std::unique_lock<std::mutex> lk(mu);
          cv.wait(lk, [&]{ return stop || !host_batches.empty() || next == batches.size() || !free.empty(); });
          if (stop) break;
          if (next < batches.size() && !free.empty()) {
            b = batches[next++]; bi = free.front(); free.pop_front();
          } else if (!host_batches.empty()) {
            b = host_batches.front(); host_batches.pop_front();
          } else if (next == batches.size()) break;
          else continue;
        }
        if (b.resident) recycle(bi); // event must complete before any byte in the buffer is changed
        const Layer& layer = d.layers[size_t(units[b.first].layer)];
        std::vector<Span> spans;
        std::vector<DgLoadCopy> copies;
        uint64_t part_base = 0;
        for (size_t p = 0; p < layer.parts.size(); ++p) {
          const Part& part = layer.parts[p];
          for (size_t j = 0; j < b.count; ++j) {
            const auto& u = units[b.first+j];
            char* dst = b.resident ? buffers[bi] + part_base * b.count + j * part.bytes : u.host + part_base;
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
        if (b.resident && direct) {
          uint64_t cursor = 0;
          std::vector<char*> destinations;
          for (const auto& a : ranges) {
            const uint64_t al = std::max<uint64_t>(4096, d.sources[a.source].align);
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
              const uint64_t al = std::max<uint64_t>(4096, d.sources[a.source].align);
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
          const uint64_t al = std::max<uint64_t>(4096, d.sources[a.source].align);
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
              const char* src = b.resident ? copies[p*b.count+j].data : units[b.first+j].host+base;
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
          if (b.resident) ready.push_back({bi, std::move(copies)});
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
}  // namespace ts
