"""CPU CLI integration: parallel JSON/text output and persisted solver picks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib

exe = str(Path(sys.argv[1]).resolve())
root = Path(sys.argv[2]).resolve()
data = root / "tests/data/placement"


def run(args, rc=0):
    p = subprocess.run([exe, *map(str, args)], cwd=root, capture_output=True, text=True)
    assert p.returncode == rc, (args, p.returncode, p.stderr)
    return p


def material(doc):
    out = {k: doc[k] for k in ("devices", "host", "units", "seed")}
    for owner in [*out["devices"], out["host"]]:
        for line in owner["lines"]:
            line.pop("source", None)
            line.pop("spilled", None)
    return out


with tempfile.TemporaryDirectory(prefix="place-cli-", dir=Path(exe).parent) as tmp:
    tmp = Path(tmp)
    probe = json.loads((data / "J2-probe.json").read_text())
    probe.update(model="DeepSeek-V4.1-Flash", engine="strix-cuda-ds41-replica")
    probe_path = tmp / "probe.json"
    probe_path.write_text(json.dumps(probe), encoding="utf-8")
    base = [data / "J2-recipe.toml", "--route", "decode_rtx", "--probe", probe_path]
    place = ["place", *base, "--components", data / "J2-components.json"]
    default = run(place).stdout
    jout = tmp / "place.json"
    assert run([*place, "--json-out", jout]).stdout == default
    assert jout.read_text() == default
    explanation = run([*place, "--explain", "--json-out", jout]).stdout
    assert explanation == run([*place, "--explain"]).stdout
    assert jout.read_text() == default
    assert "placement_sha256" in json.loads(jout.read_text())

    inputs = {"costs": {}, "choices": [{"id": "J2-B", "mode": "spec", "R": 6,
              "ctx": 131072, "lang": "code", "components": str(data / "J2-components.json")} ]}
    ipath = tmp / "inputs.json"
    ipath.write_text(json.dumps(inputs), encoding="utf-8")
    solve = ["solve", *base, "--inputs", ipath]
    default_solver = run(solve).stdout
    jout = tmp / "solve.json"
    assert run([*solve, "--json-out", jout]).stdout == default_solver
    assert jout.read_text() == default_solver
    table = run([*solve, "--table", "--json-out", jout]).stdout
    assert table == run([*solve, "--table"]).stdout
    assert "J2-B" in table and "unpredicted" in table
    assert jout.read_text() == default_solver

    overrides = tmp / "picked.toml"
    assert run([*solve, "--pick", "J2-B", "--emit-overrides", overrides,
                "--table", "--json-out", jout]).stdout == table
    assert jout.read_text() == default_solver
    assert b"\r" not in overrides.read_bytes()
    saved = tomllib.loads(overrides.read_text())
    assert saved["schema"] == "tierstream/recipe-overrides-v1"
    assert saved["set"]["routes.decode_rtx.components.arena.units"] == 7395
    assert saved["set"]["routes.decode_rtx.components.expert_file.units"] == 0
    assert saved["set"]["routes.decode_rtx.max_rows"] == 6
    before = json.loads(default)
    after = json.loads(run([*place, "--overrides", overrides]).stdout)
    assert before["effective_sha256"] != after["effective_sha256"]
    assert material(before) == material(after)
    assert "PL_E_SCHEMA" in run([*solve, "--pick", "missing", "--emit-overrides", tmp / "missing.toml"], 1).stderr
    assert not (tmp / "missing.toml").exists()
    run([*solve, "--pick", "J2-B"], 2)
    run([*solve, "--emit-overrides", overrides], 2)
    assert "cannot write" in run([*place, "--json-out", tmp / "no-dir" / "out.json"], 1).stderr

    # CLI --context reaches every choice, matching an ordinary place with the same measurements.
    context = {"schema": "tierstream/context-v1", "entries": [{"build_sha256": "0" * 64,
        "arch": "sm_120", "device_name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        "driver": "617.14", "platform": "windows-wddm", "samples": [
            {"bytes": 744245094, "method": "pdh-process-dedicated"} for _ in range(3)]}]}
    context_path = tmp / "context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    direct = json.loads(run([*place, "--context", context_path]).stdout)
    rows = json.loads(run([*solve, "--context", context_path, "--pick", "J2-B",
                          "--emit-overrides", overrides]).stdout)["rows"]
    assert rows[0]["placement"]["placement_sha256"] == direct["placement_sha256"]

print("place/solve JSON + text equality, J2/B pick persistence, context passthrough and refusals passed")
