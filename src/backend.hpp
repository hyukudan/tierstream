// Device backend interface. The core never sees CUDA/HIP types: streams and events are opaque pointers.
#pragma once

#include <cstddef>
#include <memory>

#include "strix_stream/ss.h"

namespace ss {

class Backend {
 public:
  virtual ~Backend() = default;
  virtual const char* name() const = 0;
  // Unified: device memory is host-addressable (CPU, Strix Halo). File reads and host copies land in place; there is
  // no separate pinned tier to copy from (a pinned unit is just host memory the slot copy reads).
  virtual bool unified() const = 0;
  // Called once on every library thread before it touches the device (CUDA: cudaSetDevice).
  virtual void bind_thread() {}

  virtual void* alloc_device(size_t bytes) = 0;
  virtual void free_device(void* p) = 0;
  virtual void* alloc_pinned(size_t bytes) = 0;
  virtual void free_pinned(void* p) = 0;

  // Streams: nullptr means the library's copy stream. Events are created by the backend.
  virtual void* event_create() = 0;
  virtual void event_destroy(void* ev) = 0;
  virtual void event_record(void* ev, void* stream) = 0;
  virtual void event_sync(void* ev) = 0;            // host waits; returns at once if never recorded
  virtual void stream_wait(void* stream, void* ev) = 0;  // device-side wait
  virtual void copy_h2d(void* dst, const void* src, size_t n, void* stream) = 0;
  virtual void copy_d2h(void* dst, const void* src, size_t n, void* stream) = 0;
  virtual void copy_d2d(void* dst, const void* src, size_t n, void* stream) = 0;
  virtual void stream_sync(void* stream) = 0;
  // A second copy stream for device-to-host write-backs (exclusive tier mode), so they run in the opposite PCIe
  // direction from the H2D copies. nullptr = the copy stream (backends with synchronous copies).
  virtual void* writeback_stream() { return nullptr; }
};

std::unique_ptr<Backend> make_cpu_backend();
std::unique_ptr<Backend> make_cuda_backend(int device);  // throws SS_E_UNSUPPORTED when not compiled in
std::unique_ptr<Backend> make_hip_backend(int device);   // idem

}  // namespace ss
