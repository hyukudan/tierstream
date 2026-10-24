"""Planner parity: strix-stream's ss-plan against hyperloom vramplan.plan() on the same descriptor.

    python tools/vramplan_parity.py --ss-plan build/ss-plan.exe --hyperloom D:/tmp/tm-hl-vramplan

The vramplan Model is built from the descriptor exactly as docs/design.md section 2 maps it (fixed bytes, one Unit per
(layer, unit) with the pool stride, order, tiers, a stream() callable sized like strix-stream's per-size rings).
Synthetic descriptors with DS V4.1 and MiniMax shapes; every cap / host budget pair must give the same tier per unit
and the same totals, or the same refusal.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

GIB = 1 << 30


def ds_like():
    # 40 MoE layers x 384 experts x 9,953,280 B (three GGUF parts), 10.07 GB of dense/shared/globals.
    layers = []
    for L in range(40):
        layers.append({"index": L, "kind": "moe", "n_units": 384, "top_k": 6, "fixed_bytes": 200_000_000,
                       "parts": [{"format": "IQ2_XXS", "offset": 0, "stride": 3_041_280, "bytes": 3_041_280},
                                 {"format": "IQ2_XXS", "offset": 0, "stride": 3_041_280, "bytes": 3_041_280},
                                 {"format": "Q2_K", "offset": 0, "stride": 3_870_720, "bytes": 3_870_720}]})
    return {"schema": "strix-stream/desc-v1", "engine": "ds-like", "order": "ranked", "tiers": "pinned|disk",
            "fixed_device": [{"name": "globals", "bytes": 2_070_000_000}],
            "sources": [{"path": "none.bin"}], "layers": layers}


def mm_like():
    # 3 dense + 57 blob layers of 2,864,462,336 B, prefix order (--resident N).
    layers = [{"index": L, "kind": "dense", "fixed_bytes": 237_834_240} for L in range(3)]
    for L in range(3, 60):
        layers.append({"index": L, "kind": "blob", "parts": [{"format": "mixed", "offset": 0,
                                                              "bytes": 2_864_462_336}]})
    return {"schema": "strix-stream/desc-v1", "engine": "mm-like", "order": "prefix", "tiers": "pinned|disk",
            "fixed_device": [{"name": "embed+head", "bytes": 3_690_000_000}],
            "sources": [{"path": "none.bin"}], "layers": layers}


def to_model(vp, desc, ring_slots):
    align = desc.get("unit_align", 256)
    fixed = [(n["name"], n["bytes"]) for n in desc.get("fixed_device", [])]
    units, max_units = [], {}
    for l in desc["layers"]:
        fixed.append((f"L{l['index']}.fixed", l.get("fixed_bytes", 0)))
        n = l.get("n_units", 1 if l["kind"] == "blob" else 0)
        if not n:
            continue
        ub = sum(p["bytes"] for p in l["parts"])
        stride = (ub + align - 1) // align * align
        max_units[stride] = max(max_units.get(stride, 0), n)
        for u in range(n):
            units.append(vp.Unit(f"L{l['index']}.U{u}", stride, 1.0, desc.get("tiers", "pinned|disk")))

    def stream(streamed):
        sizes = {u.bytes for u in streamed}
        return sum(s * (ring_slots or 2 * max_units[s]) for s in sizes)

    return vp.Model(desc["engine"], fixed, units, desc.get("order", "ranked"), lambda keys, plan: {},
                    stream=stream)


def ss_plan(exe, path, cap_gib, host_gib, ring):
    args = [exe, path, "--device-gib", repr(cap_gib), "--host-gib", repr(host_gib)]
    if ring:
        args += ["--ring", str(ring)]
    r = subprocess.run(args, capture_output=True, text=True)
    return (json.loads(r.stdout), None) if r.returncode == 0 else (None, r.stderr.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ss-plan", required=True)
    ap.add_argument("--hyperloom", required=True)
    a = ap.parse_args()
    sys.path.insert(0, a.hyperloom)
    from hyperloom import vramplan as vp

    cases = [(ds_like(), 0, [(48, 63), (90, 63), (90, 20), (96, 0), (200, 0), (30, 63)]),
             (ds_like(), 48, [(48, 63), (90, 63)]),
             (mm_like(), 0, [(48, 80), (90, 63), (90, 80), (95, 80), (20, 80)])]
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for desc, ring, points in cases:
            path = os.path.join(tmp, desc["engine"] + ".json")
            with open(path, "w") as f:
                json.dump(desc, f)
            model = to_model(vp, desc, ring)
            for cap, host in points:
                # ss-plan takes GiB as floats too; vramplan sizes the cap as int(cap * GIB) and the margin from it.
                ours, err = ss_plan(a.ss_plan, path, cap, host, ring)
                try:
                    theirs = vp.plan(model, vp.Inputs(vram_cap_gib=cap, host_gib=host))
                    terr = None
                except vp.PlanError as e:
                    theirs, terr = None, str(e)
                tag = f"{desc['engine']} ring={ring or 'default'} cap={cap} host={host}"
                if (ours is None) != (theirs is None):
                    print(f"MISMATCH {tag}: ours={'refused: ' + err if ours is None else 'plan'} "
                          f"vramplan={'refused: ' + terr if theirs is None else 'plan'}")
                    bad += 1
                    continue
                if ours is None:
                    print(f"same     {tag}: both refuse")
                    continue
                letters = "".join(ours["tiers"])
                mine = {}
                k = 0
                for l in desc["layers"]:
                    n = l.get("n_units", 1 if l["kind"] == "blob" else 0)
                    for u in range(n):
                        mine[f"L{l['index']}.U{u}"] = "RPDX"[ "RPDX".index(letters[k])]
                        k += 1
                res = set(theirs.resident)
                want = {key: ("R" if key in res else {"pinned": "P", "disk": "D", "pair": "X"}[theirs.tiers[key]])
                        for key in mine}
                diff = [key for key in mine if mine[key] != want[key]]
                # Pinned bytes: vramplan bills unit bytes; strix-stream bills the 4 KiB strides its pinned tier
                # allocates. Compare after converting vramplan's pinned units to 4 KiB strides.
                strides = {}
                for l in desc["layers"]:
                    n = l.get("n_units", 1 if l["kind"] == "blob" else 0)
                    ub = sum(p["bytes"] for p in l["parts"]) if n else 0
                    for u in range(n):
                        strides[f"L{l['index']}.U{u}"] = (ub + 4095) // 4096 * 4096
                want_pinned = sum(strides[k] for k, t in theirs.tiers.items() if t == "pinned")
                same_tot = (ours["resident_bytes"] == theirs.resident_bytes and ours["ring_bytes"] == theirs.stream_bytes
                            and ours["disk_bytes"] == theirs.disk_bytes and ours["peak_bytes"] == theirs.peak_bytes
                            and ours["pinned_bytes"] == want_pinned)
                if diff or not same_tot:
                    bad += 1
                    print(f"MISMATCH {tag}: {len(diff)} units differ (first {diff[:3]}); totals ours "
                          f"res {ours['resident_bytes']} ring {ours['ring_bytes']} disk {ours['disk_bytes']} peak "
                          f"{ours['peak_bytes']} vs res {theirs.resident_bytes} ring {theirs.stream_bytes} disk "
                          f"{theirs.disk_bytes} peak {theirs.peak_bytes}")
                else:
                    print(f"same     {tag}: R {ours['n_resident']} P {ours['n_pinned']} D {ours['n_disk']} "
                          f"peak {ours['peak_bytes'] / GIB:.3f} GiB")
    print("PARITY OK" if not bad else f"{bad} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
