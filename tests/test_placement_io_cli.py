"""CPU integration for explicit content scans, saved-plan admission and measured costs."""
import copy
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP

exe = Path(sys.argv[1]).resolve()
root = Path(sys.argv[2]).resolve()
data = root / "tests/data/placement"


def run(args, rc=0):
    p = subprocess.run([str(exe), *map(str, args)], cwd=root, capture_output=True, text=True)
    assert p.returncode == rc, (args, p.returncode, p.stderr)
    return p


def canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"))


def write(path, doc):
    path.write_text(canonical(doc), encoding="utf-8", newline="\n")


def digest(doc):
    return hashlib.sha256(canonical(doc).encode()).hexdigest()


def ggstr(s):
    s = s.encode()
    return struct.pack("<Q", len(s)) + s


with tempfile.TemporaryDirectory(prefix="placement-io-", dir=exe.parent) as tmp:
    tmp = Path(tmp)
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 1)
    header += ggstr("general.name") + struct.pack("<I", 8) + ggstr("fixture")
    header += ggstr("weight") + struct.pack("<I Q I Q", 1, 4, 0, 0)
    body = b"tensor-data" * 100000
    model, draft, overlay = [tmp / n for n in ("model.gguf", "draft.gguf", "overlay.gguf")]
    for p in (model, draft, overlay):
        p.write_bytes(header + body)
    manifest = tmp / "manifest.json"
    args = ["manifest", "--model", model, "--draft", draft, "--overlay", overlay]
    text = run([*args, "--json-out", manifest]).stdout
    assert manifest.read_text() == text
    assert text == canonical(json.loads(text)) + "\n"
    doc = json.loads(text)
    assert [f["path"] for f in doc["files"]] == sorted(str(p).replace("\\", "/") for p in (model, draft, overlay))
    for f in doc["files"]:
        p = Path(f["path"])
        assert f["size"] == len(header + body)
        assert f["mtime_ns"] == p.stat().st_mtime_ns
        assert f["header_sha256"] == hashlib.sha256(header).hexdigest()
        assert "sha256" not in f
    assert run(["manifest", "--verify", manifest]).stdout == text
    full = run([*args, "--full", "--json-out", manifest]).stdout
    assert all(f["sha256"] == hashlib.sha256(header + body).hexdigest() for f in json.loads(full)["files"])
    assert run(["manifest", "--verify", manifest]).stdout == full
    assert run(["manifest", "--verify", manifest, "--full"]).stdout == full
    assert "files" in run(["manifest", "--verify", manifest, model], 1).stderr

    # A Windows byte-range lock makes any actual body read fail, including CRT read-ahead.
    if sys.platform == "win32":
        import ctypes.wintypes as w
        class OVERLAPPED(ctypes.Structure):
            _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                        ("Offset", w.DWORD), ("OffsetHigh", w.DWORD), ("hEvent", w.HANDLE)]
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
        k.CreateFileW.restype = w.HANDLE
        k.LockFileEx.argtypes = [w.HANDLE, w.DWORD, w.DWORD, w.DWORD, w.DWORD, ctypes.POINTER(OVERLAPPED)]
        k.UnlockFileEx.argtypes = [w.HANDLE, w.DWORD, w.DWORD, w.DWORD, ctypes.POINTER(OVERLAPPED)]
        k.CloseHandle.argtypes = [w.HANDLE]
        handle = k.CreateFileW(str(model), 0x80000000, 7, None, 3, 0, None)
        assert handle != w.HANDLE(-1).value
        lock = OVERLAPPED()
        lock.Offset = len(header)
        assert k.LockFileEx(handle, 3, 0, len(body), 0, ctypes.byref(lock)), ctypes.get_last_error()
        try:
            run(["manifest", model])
            run(["manifest", "--verify", manifest])
            assert "full read failed" in run(["manifest", "--full", model], 1).stderr
            assert "full read failed" in run(["manifest", "--verify", manifest, "--full"], 1).stderr
        finally:
            assert k.UnlockFileEx(handle, 0, len(body), 0, ctypes.byref(lock))
            k.CloseHandle(handle)

    original = model.stat()
    model.write_bytes(header + b"X" + body[1:])
    os.utime(model, ns=(original.st_atime_ns, original.st_mtime_ns + 1000000))
    assert "mtime_ns" in run(["manifest", "--verify", manifest], 1).stderr
    os.utime(model, ns=(original.st_atime_ns, original.st_mtime_ns))
    # Documented trust boundary: an unchanged header and forged mtime require explicit full verification.
    run(["manifest", "--verify", manifest])
    assert "sha256" in run(["manifest", "--verify", manifest, "--full"], 1).stderr
    model.write_bytes(b"X" + header[1:] + body)
    os.utime(model, ns=(original.st_atime_ns, original.st_mtime_ns))
    run(["manifest", "--verify", manifest], 1)
    # Recognised-format header perturbation also retains parseability.
    model.write_bytes(header.replace(b"fixture", b"changed") + body)
    os.utime(model, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert "header_sha256" in run(["manifest", "--verify", manifest], 1).stderr
    model.write_bytes(header + body + b"X")
    assert ".size" in run(["manifest", "--verify", manifest], 1).stderr
    run(["manifest", model, model], 1)
    run(["manifest", "--header-bytes", "-1", model], 2)

    safe_header = canonical({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    safe = tmp / "draft.safetensors"
    safe.write_bytes(struct.pack("<Q", len(safe_header)) + safe_header + b"1234")
    assert json.loads(run(["manifest", safe]).stdout)["files"][0]["header_sha256"] == hashlib.sha256(
        struct.pack("<Q", len(safe_header)) + safe_header).hexdigest()
    raw = tmp / "overlay.bin"
    raw.write_bytes(b"HEADER" + body)
    raw_manifest = tmp / "raw.json"
    run(["manifest", "--header-bytes", 6, raw, "--json-out", raw_manifest])
    run(["manifest", "--verify", raw_manifest])
    assert "header_sha256" in run(["manifest", "--header-bytes", 6, raw]).stdout

    probe = json.loads((data / "J2-probe.json").read_text())
    probe.update(model="DeepSeek-V4.1-Flash", engine="strix-cuda-ds41-replica")
    probe_path, plan_path = tmp / "probe.json", tmp / "plan.json"
    write(probe_path, probe)
    base = [data / "J2-recipe.toml", "--route", "decode_rtx", "--probe", probe_path,
            "--components", data / "J2-components.json", "--usage", data / "usage-all.json"]
    plan = json.loads(run(["place", *base, "--json-out", plan_path]).stdout)
    verify = ["verify", *base, "--placement", plan_path]
    assert json.loads(run(verify).stdout) == plan
    # Whitespace/key order do not affect canonical placement identity.
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    assert json.loads(run(verify).stdout) == plan
    for field in ("seed", "units", "devices", "host", "effective_sha256", "usage_sha256", "context_sha256"):
        changed = copy.deepcopy(plan)
        if field == "devices": changed[field][0]["lines"][0]["bytes"] += 1
        elif field == "host": changed[field]["lines"][0]["bytes"] += 1
        elif field in ("seed", "units"): changed[field] = []
        else: changed[field] = "0" * 64
        changed.pop("placement_sha256")
        changed["placement_sha256"] = digest(changed)
        write(plan_path, changed)
        err = run(verify, 1).stderr
        assert "PL_E_STALE " + field in err, err
    changed = copy.deepcopy(plan)
    changed["devices"][0]["lines"][0]["bytes"] += 1
    write(plan_path, changed)
    assert "PL_E_STALE placement_sha256" in run(verify, 1).stderr
    write(plan_path, plan)
    assert "PL_E_STALE effective_sha256" in run([*verify, "--set", "routes.decode_rtx.placement.rtx.device_gib=89"], 1).stderr
    for field in ("driver", "platform", "arch", "uuid", "pci"):
        changed_probe = copy.deepcopy(probe)
        changed_probe["devices"][0][field] = {"driver": "changed", "platform": "linux", "arch": "sm_89",
                                              "uuid": "GPU-00000000-0000-0000-0000-000000000099",
                                              "pci": "0000:99:00.0"}[field]
        write(probe_path, changed_probe)
        assert "PL_E_STALE devices" in run(verify, 1).stderr
    write(probe_path, probe)
    changed_components = json.loads((data / "J2-components.json").read_text())
    changed_components["build_sha256"] = "f" * 64
    component_path = tmp / "components.json"
    write(component_path, changed_components)
    changed_base = [component_path if a == data / "J2-components.json" else a for a in base]
    assert "PL_E_STALE build_sha256" in run(["verify", *changed_base, "--placement", plan_path], 1).stderr
    # Optional inputs are part of the saved identity; omitting a used usage document is stale.
    no_usage = base[:-2]
    assert "PL_E_STALE usage_sha256" in run(["verify", *no_usage, "--placement", plan_path], 1).stderr

    values = [0, 0.0005, 0.0015, 1.2345, -0.5, 5e-324, 1e-12, 0.0004999999999999999,
              0.0005000000000000001, 123456789.0005, 1.2344999999999999]
    rng = random.Random(21)
    values += [rng.random() * 10 ** rng.randrange(-8, 9) for _ in range(1000)]
    report = {"schema": "wdsrtx/rtx-unit-costs-v1", "allhit_ms": {str(i): v for i, v in enumerate(values)},
              "miss_file_ms": None, "miss_pinned_ms": -1, "verify_ms": {"8192": {"6": 0.0005}},
              "accepted": {"code/R6": 3.4565}, "provenance": {"allhit_ms": {"source": "fixture", "estimate": True}}}
    cost_path, out_path = tmp / "costs.json", tmp / "converted.json"
    write(cost_path, report)
    converted = run(["costs", "--from", "rtx-unit-costs-v1", cost_path, "--json-out", out_path]).stdout
    assert out_path.read_text() == converted
    assert converted == canonical(json.loads(converted)) + "\n"
    out = json.loads(converted)
    for i, v in enumerate(values):
        if v < 0: assert str(i) not in out["allhit_us"]
        else: assert out["allhit_us"][str(i)] == int((Decimal(repr(v)) * 1000).quantize(Decimal(1), rounding=ROUND_HALF_UP)), (i, v)
    assert "miss_file_us" not in out and "miss_pinned_us" not in out
    assert out["verify_us"]["8192"]["6"] == 1
    assert out["accepted_milli"]["code/R6"] == 3457
    assert out["provenance"]["allhit_us"]["source"] == "fixture"

print("manifest bounded/full/stale scans, placement verify perturbations, and 1011 decimal cost conversions passed")
