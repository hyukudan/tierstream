// CUDA backend (discrete GPU: RTX PRO 6000, 4090, any CUDA card). Resident pool and stream ring in VRAM, pinned host
// tier and staging ring in cudaHostAlloc memory, one non-blocking copy stream for every H2D copy (one DMA engine is
// enough: a second copy stream added nothing on the PRO, spec-rtx-prefill §2.1).
#include <cuda_runtime.h>

#include <string>

#include "backend.hpp"
#include "errors.hpp"

namespace ss {
namespace {

void ck(cudaError_t e, const char* what) {
  if (e != cudaSuccess)
    fail(e == cudaErrorMemoryAllocation ? SS_E_NOMEM : SS_E_BACKEND,
         std::string("cuda: ") + what + ": " + cudaGetErrorString(e));
}

class CudaBackend final : public Backend {
 public:
  explicit CudaBackend(int device) : device_(device) {
    ck(cudaSetDevice(device_), "cudaSetDevice");
    ck(cudaStreamCreateWithFlags(&copy_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
  }
  ~CudaBackend() override {
    cudaSetDevice(device_);
    if (copy_) {
      cudaStreamSynchronize(copy_);
      cudaStreamDestroy(copy_);
    }
  }
  const char* name() const override { return "cuda"; }
  bool unified() const override { return false; }

  void* alloc_device(size_t bytes) override {
    ck(cudaSetDevice(device_), "cudaSetDevice");
    void* p = nullptr;
    ck(cudaMalloc(&p, bytes ? bytes : 1), "cudaMalloc");
    return p;
  }
  void free_device(void* p) override {
    if (p) cudaFree(p);
  }
  void* alloc_pinned(size_t bytes) override {
    ck(cudaSetDevice(device_), "cudaSetDevice");
    void* p = nullptr;
    ck(cudaHostAlloc(&p, bytes ? bytes : 1, cudaHostAllocPortable), "cudaHostAlloc");
    return p;
  }
  void free_pinned(void* p) override {
    if (p) cudaFreeHost(p);
  }

  void* event_create() override {
    cudaEvent_t e = nullptr;
    ck(cudaEventCreateWithFlags(&e, cudaEventDisableTiming), "cudaEventCreate");
    return e;
  }
  void event_destroy(void* ev) override {
    if (ev) cudaEventDestroy(static_cast<cudaEvent_t>(ev));
  }
  void event_record(void* ev, void* stream) override {
    ck(cudaEventRecord(static_cast<cudaEvent_t>(ev), s(stream)), "cudaEventRecord");
  }
  void event_sync(void* ev) override {
    if (ev) ck(cudaEventSynchronize(static_cast<cudaEvent_t>(ev)), "cudaEventSynchronize");
  }
  void stream_wait(void* stream, void* ev) override {
    if (ev) ck(cudaStreamWaitEvent(s(stream), static_cast<cudaEvent_t>(ev), 0), "cudaStreamWaitEvent");
  }
  void copy_h2d(void* dst, const void* src, size_t n, void* stream) override {
    ck(cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, s(stream)), "cudaMemcpyAsync H2D");
  }
  void copy_d2h(void* dst, const void* src, size_t n, void* stream) override {
    ck(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToHost, s(stream)), "cudaMemcpyAsync D2H");
  }
  void copy_d2d(void* dst, const void* src, size_t n, void* stream) override {
    ck(cudaMemcpyAsync(dst, src, n, cudaMemcpyDeviceToDevice, s(stream)), "cudaMemcpyAsync D2D");
  }
  void stream_sync(void* stream) override { ck(cudaStreamSynchronize(s(stream)), "cudaStreamSynchronize"); }

 private:
  // nullptr = the library's copy stream. The engine's legacy default stream must be passed as cudaStreamLegacy
  // (0 would be read as "copy stream").
  cudaStream_t s(void* stream) const { return stream ? static_cast<cudaStream_t>(stream) : copy_; }

  int device_;
  cudaStream_t copy_ = nullptr;
};

}  // namespace

std::unique_ptr<Backend> make_cuda_backend(int device) { return std::make_unique<CudaBackend>(device); }

}  // namespace ss
