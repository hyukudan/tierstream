"""Negative cases for validate_v1.py, derived (read-only) from the runnable MiniMax lab recipe. Run from examples/:
python -I tests/make_tests.py"""
import sys

s = open("minimax-m3.1-flash.lab.toml", encoding="utf-8").read()

cases = {
    "unknown-key": s.replace("[limits]\n", "[limits]\nbogus = 1\n", 1),
    "exact-key": s.replace("[match]\n", "[match]\nexact = true\n", 1),
    "string-bool": s.replace("reader_threads = 1", "reader_threads = 1\npin_planned = \"yes\""),
    "nan-pred": s.replace("value = 16.1", "value = nan"),
    "uncited-threshold": s.replace('"routes.prefill_long_rtx.when.prompt_tokens_min" = ["mm-route-rule", "mm-t7-1k", "mm-t7-8k"]\n', ""),
    "bad-quote": s.replace('quote = "tiers device 32, pinned 16', 'quote = "tiers device 32, pinned 17'),
    "dangling-cite": s.replace("[x-cites]\n", '[x-cites]\n"limits.nothing" = "mm-watchdog"\n'),
    "no-catchall": s.replace('id = "prefill_pair"\nphase = "prefill"\ndevices = ["pair"]\nexecute = "tp"\nhandoff_route = "decode_pair"\n',
                             'id = "prefill_pair"\nphase = "prefill"\ndevices = ["pair"]\nexecute = "tp"\nhandoff_route = "decode_pair"\nwhen = { prompt_tokens_max = 4095 }\n'),
    "catchall-with-batch": s.replace('id = "prefill_pair"\nphase = "prefill"\ndevices = ["pair"]\nexecute = "tp"\nhandoff_route = "decode_pair"\n',
                                     'id = "prefill_pair"\nphase = "prefill"\ndevices = ["pair"]\nexecute = "tp"\nhandoff_route = "decode_pair"\nwhen = { batch_max = 1 }\n'),
    "unlinked-handoff": s.replace('links = ["pc-nodes"]\nfallback = "prefill_pair"', 'fallback = "prefill_pair"'),
    "handoff-to-prefill": s.replace('handoff_route = "decode_pair"\nlinks = ["pc-nodes"]', 'handoff_route = "prefill_pair"\nlinks = ["pc-nodes"]'),
    "missing-max-batch": s.replace("max_context_tokens = 9216\nmax_batch = 1\nworkspace_gib", "max_context_tokens = 9216\nworkspace_gib"),
    "over-cap": s.replace("device_gib = 93.0", "device_gib = 99.0"),
    "zero-device": s.replace("device_gib = 93.0", "device_gib = 0"),
    "workspace-eq-cap": s.replace("workspace_gib = 0.0", "workspace_gib = 93.0", 1),
    "missing-context": s.replace("device_gib = 93.0\nhost_pinned_gib = 44.3\nmax_context_tokens = 9216", "device_gib = 93.0\nhost_pinned_gib = 44.3"),
    "single-on-pair": s.replace('id = "decode_pair"\nphase = "decode"\ndevices = ["pair"]\nexecute = "tp"', 'id = "decode_pair"\nphase = "decode"\ndevices = ["pair"]\nexecute = "single"'),
    "fallback-cycle": s.replace('handoff_route = "decode_pair"\n[routes.placement.pair]', 'handoff_route = "decode_pair"\nfallback = "prefill_long_rtx"\n[routes.placement.pair]', 1),
    "rest-pinned-fill": s.replace('pinned = 16\nrest = "disk"', 'pinned = "fill"\nrest = "pinned"'),
    "bad-name": s.replace('name = "minimax-m3.1-flash-lab"', 'name = "minimax m3.1 flash/lab"'),
    "bad-sha": s.replace('model = "minimax-m3.1-flash"', 'model = "minimax-m3.1-flash"\nmodel_sha256 = "XYZ"', 1),
    "bad-pin": s.replace('x-status', 'pin = ["routes.nope.when.prompt_tokens_min"]\nx-status', 1),
    "bad-layer-range": s.replace('rank = "list"', 'layers = [[5, 2], -1]\nrank = "list"', 1),
    "empty-evidence-list": s.replace('evidence = "mm-t7-8k"', "evidence = []", 1),
    "link-between-string": s.replace('between = ["rtx", "pair"]', 'between = "rtx"', 1),
    "when-not-table": s.replace("when = { prompt_tokens_min = 4096 }", "when = 4096", 1),
    "units-not-array": s.replace("[[routes.placement.rtx.units]]", "[routes.placement.rtx.units]", 1),
    "peer-outside-hetero": s.replace('pinned = 16\nrest = "disk"', 'pinned = 16\nrest = "peer"\npeer = "pair"'),
    # Sol-3 probes
    "pin-not-settable": s.replace('x-status', 'pin = ["match.devices.pair.count"]\nx-status', 1),
    "pin-duplicate": s.replace('x-status', 'pin = ["routes.prefill_long_rtx", "routes.prefill_long_rtx"]\nx-status', 1),
    "engine-int": s.replace('model = "minimax-m3.1-flash"', 'model = "minimax-m3.1-flash"\nengine = 5', 1),
    "description-table": s.replace('description = "', 'description.x = "', 1),
    "layers-uncited": s.replace('rank = "list"', 'layers = [[0, 59]]\nrank = "list"', 1),
    "between-not-ids": s.replace('between = ["rtx", "pair"]', 'between = [{}, "pair"]', 1),
    "handoff-not-string": s.replace('handoff_route = "decode_pair"\nlinks', 'handoff_route = []\nlinks', 1),
    # Sol-4 probes
    "fallback-not-string": s.replace('fallback = "prefill_pair"', "fallback = []", 1),
    "devices-not-ids": s.replace('id = "decode_pair"\nphase = "decode"\ndevices = ["pair"]', 'id = "decode_pair"\nphase = "decode"\ndevices = [{}]', 1),
    "pin-wrong-set": s.replace('x-status', 'pin = ["routes.decode_pair.placement.rtx.device_gib"]\nx-status', 1),
    "pin-missing-group": s.replace('x-status', 'pin = ["routes.decode_pair.placement.pair.units.layers.resident"]\nx-status', 1),
    "duplicate-route-link": s.replace('links = ["pc-nodes"]', 'links = ["pc-nodes", "pc-nodes"]', 1),
    "two-default-groups": s.replace('[[routes.placement.rtx.units]]\nid = "layers"', '[[routes.placement.rtx.units]]\nid = "extra"\nclass = "blob"\nresident = "fill"\n\n[[routes.placement.rtx.units]]\nid = "layers"', 1),
    "pinned-without-host": s.replace("host_pinned_gib = 44.3\n", "", 1),
    # final-core fields
    "enabled-string": s.replace("\nenabled = false\n", "\nenabled = \"no\"\n", 1),
    "h2d-on-hip": s.replace('internal_link = { kind = "usb4", gbps_min = 30.0 }', 'internal_link = { kind = "usb4", gbps_min = 30.0 }\nh2d_gbytes_ps_min = 25.0', 1),
    "tier-mode-bad": s.replace("workspace_gib = 0.0", 'workspace_gib = 0.0\ntier_mode = "both"', 1),
    "uncited-idle-preload": s.replace("workspace_gib = 0.0", "workspace_gib = 0.0\nidle_preload_layers = 2", 1),
}
n = 0
for k, v in cases.items():
    if v == s:
        sys.exit(f"case {k}: replacement did not apply")
    open(f"tests/bad-{k}.toml", "w", encoding="utf-8").write(v)
    n += 1
print(n, "cases")
