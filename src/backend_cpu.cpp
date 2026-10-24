// Host-memory backend: "device" memory is aligned host memory and every copy is a synchronous memcpy. Used by the
// CPU tests and by CPU engines; it is unified, like the Strix Halo backend, so it exercises the in-place read path.
#include <cstdlib>
#include <cstring>
#include <new>

#include "backend.hpp"
#include "errors.hpp"

#if !defined(SS_WITH_CUDA)
#define SS_NO_CUDA_BACKEND 1
#endif
#if !defined(SS_WITH_HIP)
#define SS_NO_HIP_BACKEND 1
#endif

namespace ss {
namespace {

void* aligned_alloc_4k(size_t bytes) {
  if (bytes == 0) bytes = 1;
  size_t n = (bytes + 4095) / 4096 * 4096;
#if defined(_WIN32)
  return _aligned_malloc(n, 4096);
#else
  return std::aligned_alloc(4096, n);
#endif
}

void aligned_free(void* p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

class CpuBackend final : public Backend {
 public:
  const char* name() const override { return "cpu"; }
  bool unified() const override { return true; }
  void* alloc_device(size_t bytes) override {
    void* p = aligned_alloc_4k(bytes);
    require(p != nullptr, SS_E_NOMEM, "cpu backend: out of memory");
    return p;
  }
  void free_device(void* p) override { aligned_free(p); }
  void* alloc_pinned(size_t bytes) override { return alloc_device(bytes); }
  void free_pinned(void* p) override { aligned_free(p); }
  void* event_create() override { return new int(0); }
  void event_destroy(void* ev) override { delete static_cast<int*>(ev); }
  void event_record(void*, void*) override {}
  void event_sync(void*) override {}
  void stream_wait(void*, void*) override {}
  void copy_h2d(void* dst, const void* src, size_t n, void*) override { std::memcpy(dst, src, n); }
  void copy_d2h(void* dst, const void* src, size_t n, void*) override { std::memcpy(dst, src, n); }
  void copy_d2d(void* dst, const void* src, size_t n, void*) override { std::memmove(dst, src, n); }
  void stream_sync(void*) override {}
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend() { return std::make_unique<CpuBackend>(); }

#if defined(SS_NO_CUDA_BACKEND)
std::unique_ptr<Backend> make_cuda_backend(int) {
  fail(SS_E_UNSUPPORTED, "strix-stream was built without the CUDA backend (configure with -DSS_CUDA=ON)");
}
#endif
#if defined(SS_NO_HIP_BACKEND)
std::unique_ptr<Backend> make_hip_backend(int) {
  fail(SS_E_UNSUPPORTED, "strix-stream was built without the HIP backend (configure with -DSS_HIP=ON)");
}
#endif

}  // namespace ss
