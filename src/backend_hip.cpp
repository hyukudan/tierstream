// HIP backend STUB for Strix Halo (gfx1151, unified LPDDR5X). Not built or tested on this PC (no ROCm here); it is
// the interface the node replica adapters will exercise first.
//
// Model: unified memory. "Device" slots are host-pinned, GPU-mapped memory (hipHostMalloc, coherent), so:
//   - resident = mapped: the GPU reads the slot at full memory bandwidth; there is no H2D copy and no pinned tier
//     (plan with host_pinned_bytes = 0 so every non-resident unit is a disk unit);
//   - a miss is a local NVMe read: the reader thread reads the unit's aligned range unbuffered straight into the slot
//     (in place) or through the staging ring when the record is not sector-aligned;
//   - before writing a slot the reader host-waits the slot's release event (hipEventSynchronize): the C-1 rule of the
//     USB4 review applies to any host write into memory the GPU may still read;
//   - acquire host-waits the job (landed). A later revision gates the engine stream on a mapped flag instead
//     (hipStreamWaitValue64 on a host-written word) so hits run while misses land.
#include "backend.hpp"
#include "errors.hpp"

#if defined(SS_WITH_HIP)
#include <hip/hip_runtime.h>

#include <cstring>
#include <string>

namespace ss {
namespace {

void ck(hipError_t e, const char* what) {
  if (e != hipSuccess)
    fail(e == hipErrorOutOfMemory ? SS_E_NOMEM : SS_E_BACKEND,
         std::string("hip: ") + what + ": " + hipGetErrorString(e));
}

class HipBackend final : public Backend {
 public:
  explicit HipBackend(int device) : device_(device) { ck(hipSetDevice(device_), "hipSetDevice"); }
  const char* name() const override { return "hip"; }
  bool unified() const override { return true; }
  void bind_thread() override { ck(hipSetDevice(device_), "hipSetDevice"); }

  void* alloc_device(size_t bytes) override {
    void* p = nullptr;
    ck(hipHostMalloc(&p, bytes ? bytes : 1, hipHostMallocMapped | hipHostMallocCoherent), "hipHostMalloc");
    return p;
  }
  void free_device(void* p) override {
    if (p) (void)hipHostFree(p);
  }
  void* alloc_pinned(size_t bytes) override { return alloc_device(bytes); }
  void free_pinned(void* p) override { free_device(p); }

  void* event_create() override {
    hipEvent_t e = nullptr;
    ck(hipEventCreateWithFlags(&e, hipEventDisableTiming), "hipEventCreate");
    return e;
  }
  void event_destroy(void* ev) override {
    if (ev) (void)hipEventDestroy(static_cast<hipEvent_t>(ev));
  }
  void event_record(void* ev, void* stream) override {
    ck(hipEventRecord(static_cast<hipEvent_t>(ev), static_cast<hipStream_t>(stream)), "hipEventRecord");
  }
  void event_sync(void* ev) override {
    if (ev) ck(hipEventSynchronize(static_cast<hipEvent_t>(ev)), "hipEventSynchronize");
  }
  void stream_wait(void* stream, void* ev) override {
    if (ev && stream) ck(hipStreamWaitEvent(static_cast<hipStream_t>(stream), static_cast<hipEvent_t>(ev), 0),
                         "hipStreamWaitEvent");
  }
  // Unified: every copy is a host memcpy into mapped memory (the library only copies after the slot is free).
  void copy_h2d(void* dst, const void* src, size_t n, void*) override { std::memcpy(dst, src, n); }
  void copy_d2h(void* dst, const void* src, size_t n, void* stream) override {
    if (stream) ck(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "hipStreamSynchronize");
    std::memcpy(dst, src, n);
  }
  void copy_d2d(void* dst, const void* src, size_t n, void*) override { std::memmove(dst, src, n); }
  void stream_sync(void* stream) override {
    if (stream) ck(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "hipStreamSynchronize");
  }

 private:
  int device_;
};

}  // namespace

std::unique_ptr<Backend> make_hip_backend(int device) { return std::make_unique<HipBackend>(device); }

}  // namespace ss
#endif
