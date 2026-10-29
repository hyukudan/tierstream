/* Measurement-only capture, version 1. Host memory; never a device ABI.
 * Bind/unbind and read only between completed tokens. The engine and service
 * are alternating single owners, synchronized by begin/terminal publication.
 * Storage is caller-owned and preallocated. No wrap: first overflow latches
 * incomplete and stops collection. lost counts attempted records after full.
 */
#ifndef SS_CAPTURE_H
#define SS_CAPTURE_H
#include <stdint.h>
#define SS_DG_CAPTURE 1
/* kinds: 1 export, 2 unique decision, 3 WB enqueue, 4 host timeline,
 * 5 device timeline (at end, only existing diagnostics).
 * decision flags: 1 export miss, 2 fetched, 4 resurrected, 8 bank,
 * 16 covered, 32 prefetch. homes follow DgCore::Home (Host=0).
 * Host timeline stamps[0..2] are existing seen/demand/end steady-clock ns.
 * Decision valid=1 means its first masked occurrence was processed.
 * Timeline validity: bit 0 host timeline, 1 export stamp, 2 wait entry,
 * 3 ready observed, 4 existing device durations. GPU stamps are globaltimer ns;
 * host durations are us, GPU durations [dma, first, wb_left] are ms.
 */
typedef struct ss_dg_capture_event {
  uint32_t kind, seq, layer, n_ids;
  uint64_t mask;
  uint32_t ids[64];
  uint32_t flags, home_before, home_after, valid;
  int32_t slot, block;
  uint64_t stamps[3], counts[7];
  double times[5];
} ss_dg_capture_event;
typedef struct ss_dg_capture_sink {
  uint32_t struct_size, event_size;
  ss_dg_capture_event* records;
  uint64_t capacity, count, lost;
  uint32_t incomplete, reserved;
} ss_dg_capture_sink;
/* Internal/public helper: no allocation, callbacks, atomics, locks or syscalls.
 * The caller supplies a valid sink, or NULL. Not safe for simultaneous owners. */
static inline ss_dg_capture_event* ss_dg_capture_reserve(ss_dg_capture_sink* s) {
  if (!s) return 0;
  if (s->incomplete || s->count == s->capacity) {
    s->incomplete = 1;
    ++s->lost;
    return 0;
  }
  return &s->records[s->count++];
}
#endif
