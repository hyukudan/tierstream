/* The tierstream names, from C (wave4/wstream/rename/design.md §2.1): ss.h first, then ts.h (the other order is
 * tests/test_ts_names.cpp). The old names stay real: a consumer may name its own members ss_budget / ts_budget, paste
 * tokens onto ss_, and call either name; the ts_ functions give what the ss_ ones give. */
#include <stdio.h>
#include <string.h>

#include "strix_stream/ss.h"
#include "tierstream/ts.h"

static int failures = 0;
#define EXPECT(c)                                                     \
  do {                                                                \
    if (!(c)) {                                                       \
      printf("  %s:%d: EXPECT(%s) failed\n", __FILE__, __LINE__, #c); \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

/* a consumer's own identifiers that match old and new names */
struct consumer {
  int ss_budget;
  int ts_budget;
};

/* token pasting and stringizing onto the old prefix */
#define OLD(name) ss_##name
#define OLD_NAME(name) "ss_" #name

int main(void) {
  struct consumer k;
  k.ss_budget = 1;
  k.ts_budget = 2;
  EXPECT(k.ss_budget + k.ts_budget == 3);

  EXPECT(OLD(abi_version)() == ss_abi_version());
  EXPECT(strcmp(OLD_NAME(abi_version), "ss_abi_version") == 0);
  EXPECT(ts_abi_version() == ss_abi_version());
  {
    /* through volatile values: the compile-time comparisons are the point, not a constant-condition warning */
    volatile int v[8] = {TS_ABI_VERSION, SS_ABI_VERSION, TS_OK, SS_OK, TS_E_INVALID, SS_E_INVALID,
                         TS_TIER_MODE_EXCLUSIVE, SS_TIER_MODE_EXCLUSIVE};
    EXPECT(v[0] == v[1] && v[2] == v[3] && v[4] == v[5] && v[6] == v[7]);
  }

  /* the same types: a ts_budget is an ss_budget, filled the same way by either name */
  ss_budget a;
  ts_budget b;
  ss_budget_defaults(&a);
  ts_budget_defaults(&b);
  EXPECT(memcmp(&a, &b, sizeof a) == 0);
  ss_options oa;
  ts_options ob;
  ss_options_defaults(&oa);
  ts_options_defaults(&ob);
  EXPECT(memcmp(&oa, &ob, sizeof oa) == 0);

  /* the same descriptor through both names */
  const char* js = "{\"schema\":\"strix-stream/desc-v1\",\"engine\":\"t\",\"sources\":[],\"layers\":[{\"index\":0,\"kind\":\"dense\"}]}";
  ss_desc* d1 = NULL;
  ts_desc* d2 = NULL;
  EXPECT(ss_desc_from_json(js, strlen(js), &d1) == ts_desc_from_json(js, strlen(js), &d2));
  if (d1 && d2) {
    EXPECT(ss_desc_n_layers(d1) == ts_desc_n_layers(d2));
    char *j1 = NULL, *j2 = NULL;
    EXPECT(ss_desc_to_json(d1, &j1, NULL) == SS_OK && ts_desc_to_json(d2, &j2, NULL) == TS_OK);
    EXPECT(j1 && j2 && strcmp(j1, j2) == 0);
    ss_free(j1);
    ts_free(j2);
  }
  ss_desc_free(d1);
  ts_desc_free(d2);

  /* the new format id is accepted on input, the output keeps the old one */
  const char* js_new = "{\"schema\":\"tierstream/desc-v1\",\"engine\":\"t\",\"sources\":[],\"layers\":[{\"index\":0,\"kind\":\"dense\"}]}";
  ts_desc* d3 = NULL;
  EXPECT(ts_desc_from_json(js_new, strlen(js_new), &d3) == TS_OK);
  if (d3) {
    char* j3 = NULL;
    EXPECT(ts_desc_to_json(d3, &j3, NULL) == TS_OK);
    EXPECT(j3 && strstr(j3, "\"strix-stream/desc-v1\"") != NULL);
    ts_free(j3);
  }
  ts_desc_free(d3);

  /* an error through either name sets the same last error */
  EXPECT(ts_desc_from_json("{", 1, &d3) != TS_OK);
  const char* e1 = ts_last_error();
  EXPECT(e1 && *e1);

  printf("%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
