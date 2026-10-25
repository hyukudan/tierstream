// Device-gated decode mode: the kernels the launchers put on the engine's stream (dg_kernels.cu, CUDA only).
#pragma once

#include <cstdint>

#include "dg_proto.hpp"

namespace ts {
namespace dgk {

// Every pointer the kernels read: fixed at open, so captured graphs replay for the context's life.
struct Dev {
  // device memory
  uint32_t* seq = nullptr;
  uint32_t* abort = nullptr;
  uint32_t* err_count = nullptr;
  uint32_t* err_code = nullptr;
  uint32_t* err_aux = nullptr;
  uint32_t* any_miss = nullptr;  // [G]
  uint32_t* absent_err = nullptr;  // set by a slot lookup that met an absent entry without abort (ss_dg_slot)
  const int32_t* slot_of = nullptr;
  const uint32_t* unit_base = nullptr;  // [G + 1]
  // mapped host memory, device addresses
  DgRecord* rec = nullptr;
  DgPredRecord* prec = nullptr;
  uint32_t* flag = nullptr;
  uint32_t* pflag = nullptr;
  uint32_t* ready = nullptr;
  uint32_t* done = nullptr;
  uint32_t* m_err_code = nullptr;
  uint32_t* m_err_aux = nullptr;
  uint32_t* host_abort = nullptr;
  uint32_t* host_seq = nullptr;     // written by ss_begin: the one sequence the host authorized
  uint32_t* began = nullptr;        // the begin kernel's seq, after a system fence
  uint32_t* m_err_count = nullptr;  // the device's wait timeouts, copied by the end kernel
  // dg_diag only (null = off): globaltimer stamps per gated layer, [4g] export, [4g + 1] wait entry, [4g + 2] ready
  // seen (both 0 when the layer had no miss), [4g + 3] unused
  unsigned long long* gts = nullptr;
  unsigned long long timeout_ns = 0;
  uint32_t test_flags = 0;  // test-only mutations (kSkipWait)
};
constexpr uint32_t kSkipWait = 1;  // the wait kernel returns at once (must make the GPU test fail)

// Each returns the launch's cudaError_t as an int (0 = success); none synchronizes or allocates.
int launch_begin(const Dev& d, void* stream);
int launch_export(const Dev& d, uint32_t g, const int32_t* ids, uint32_t n, uint32_t n_pred, void* stream);
int launch_predict(const Dev& d, uint32_t g, const int32_t* ids, uint32_t n, void* stream);
int launch_wait(const Dev& d, uint32_t g, void* stream);
int launch_end(const Dev& d, void* stream);
// Loads the kernels (no lazy loading inside a capture). Returns a cudaError_t as an int.
int preload();
// Test-only: a kernel that spins `ns` nanoseconds on the stream (delayed readers in the GPU test).
int launch_spin(unsigned long long ns, void* stream);

}  // namespace dgk
}  // namespace ts
