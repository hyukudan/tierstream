# Re-runs the spec's prefill planner (wave4/wstream/plan/planner.py, the model behind spec-rtx-prefill.md §4.1) for the
# card32 examples: the table's 32 decimal GB column and the recipes' actual room (device_gib 31 - default margin
# max(2, 3 %) = 2 GiB -> 29 GiB = 31.14 GB), with the PC's 70 GB pinned budget, B = 30 GB/s.
# Usage (from wave4/wstream/plan): python -I ../../recipes/examples/derived/card32_planner.py
import json, sys
sys.path.insert(0, '.')
import planner as pl
pl.COLD.update(json.load(open('../routes/ds41-coldtouch.json')))
pl.B_PCIE = 30e9
ROOM = (31.0 - 2.0) * 2**30          # recipe: device_gib 31, default margin max(2 GiB, 3 % x 31 = 0.93) = 2 GiB
for name, m, Ps in (("ds-exact", pl.ds('exact'), (8192, 16384, 32768)), ("minimax", pl.minimax(), (8192,))):
    for label, V in (("col32GB", 32e9), ("room%.2fGB" % (ROOM / 1e9), ROOM)):
        for P in Ps:
            b = pl.plan(m, P, V, 70e9)
            print(f"{name} {label} P={P} compute_phase_s={'none' if b is None else round(b[1][0], 3)}")
