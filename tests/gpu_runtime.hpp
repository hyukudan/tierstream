// The GPU test (test_cuda.cu) on either runtime: CUDA as written, or HIP (ts-test-hip, Strix Halo) through these
// names. Only what the test uses is mapped.
#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>

#define SS_TEST_BACKEND SS_BACKEND_HIP
#define SS_TEST_RUNTIME "hip"
#define SS_TEST_UNIFIED 1

#define cudaDeviceProp hipDeviceProp_t
#define cudaErrorNotReady hipErrorNotReady
#define cudaError_t hipError_t
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#else
#include <cuda_runtime.h>

#define SS_TEST_BACKEND SS_BACKEND_CUDA
#define SS_TEST_RUNTIME "cuda"
#define SS_TEST_UNIFIED 0
#endif
