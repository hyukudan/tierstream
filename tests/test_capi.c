/* The C ABI compiles and links from C. */
#include <stdio.h>
#include <string.h>

#include "strix_stream/ss.h"

/* ss_snapshot_info layout as a C caller sees it (append-only). */
typedef char snapshot_flags_at_4[offsetof(ss_snapshot_info, flags) == 4 ? 1 : -1];
typedef char snapshot_sequence_at_8[offsetof(ss_snapshot_info, sequence) == 8 ? 1 : -1];
typedef char snapshot_readahead_at_24[offsetof(ss_snapshot_info, readahead_buffers) == 24 ? 1 : -1];
typedef char snapshot_exclusive_at_96[offsetof(ss_snapshot_info, exclusive_writebacks) == 96 ? 1 : -1];
typedef char snapshot_v1_prefix_160[offsetof(ss_snapshot_info, exclusive_blocks) == 152 ? 1 : -1];
typedef char snapshot_discards_at_160[offsetof(ss_snapshot_info, exclusive_discards) == 160 ? 1 : -1];

int main(void) {
  const char* js =
      "{\"schema\":\"strix-stream/desc-v1\",\"engine\":\"c\",\"sources\":[{\"path\":\"none.bin\"}],"
      "\"layers\":[{\"index\":0,\"kind\":\"moe\",\"n_units\":4,\"top_k\":1,"
      "\"parts\":[{\"format\":\"BF16\",\"offset\":0,\"bytes\":4096}]}]}";
  ss_desc* d = NULL;
  ss_plan* p = NULL;
  ss_budget b;
  char* out = NULL;
  int rc = 0;
  ss_snapshot_info info;
  ss_layer_stats total;
  if (ss_abi_version() != SS_ABI_VERSION) return 1;
  if (ss_stats_snapshot(NULL, &total, &info) != SS_E_INVALID) return 1; /* the inline wrapper compiles from C */
  if (ss_desc_from_json(js, strlen(js), &d) != SS_OK) {
    printf("desc: %s\n", ss_last_error());
    return 1;
  }
  ss_budget_defaults(&b);
  b.device_bytes = (uint64_t)3 << 30;
  if (ss_plan_make(d, &b, NULL, 0, &p) != SS_OK) {
    printf("plan: %s\n", ss_last_error());
    rc = 1;
  } else if (ss_plan_to_json(p, &out, NULL) != SS_OK || !strstr(out, "\"tiers\":[\"RRRR\"]")) {
    printf("plan json: %s\n", out ? out : ss_last_error());
    rc = 1;
  }
  ss_free(out);
  ss_plan_free(p);
  ss_desc_free(d);
  printf(rc ? "FAIL capi\n" : "PASS capi\n");
  return rc;
}
