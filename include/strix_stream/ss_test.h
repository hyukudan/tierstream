/* Device-gated decode: test-only hooks (design rev 3.4 §11.5). Not part of the stable ABI and not in ss.h. Each one
 * refuses (SS_E_INVALID) unless the context was opened with ss_options.dg_test_hooks = 1, and each acts only between
 * tokens unless noted. */
#ifndef STRIX_STREAM_SS_TEST_H
#define STRIX_STREAM_SS_TEST_H

#include "strix_stream/ss.h"

#ifdef __cplusplus
extern "C" {
#endif

/* A cache-core fault injection that the tests must catch (DgCore::Mutation; 0 = none). */
ss_status ss_test_dg_mutate(ss_ctx* c, int mutation);
/* The gate-4 pressure knob: demand misses take at most free_per_token free slots per token, and every boundary also
 * evicts the evict_recent most recently used units. 0, 0 = off. */
ss_status ss_test_dg_pressure(ss_ctx* c, uint32_t free_per_token, uint32_t evict_recent);
/* Checks every ownership invariant of the cache (a failure poisons nothing, it reports SS_E_BACKEND). */
ss_status ss_test_dg_check(ss_ctx* c);
/* The library's demand (0) or prefetch (1) stream waits for `event` (a cudaEvent_t): delays its copies. */
ss_status ss_test_dg_gate(ss_ctx* c, int stream, void* event);
/* Kernel mutations: 1 = the wait kernel returns at once, 2 = ss_end skips the completion edge (both must fail the GPU
 * test). Set before a graph is captured: the flags are kernel arguments. */
ss_status ss_test_dg_flags(ss_ctx* c, uint32_t flags);
/* Stops the service thread (its next token fails closed). */
ss_status ss_test_dg_stop_service(ss_ctx* c);

#ifdef __cplusplus
}
#endif

#endif
