#!/usr/bin/env python3
"""The rename's symbol gate.

The expected names come from tools/rename_abi_baseline.txt, captured once from main 4dc53b0's archive (the last commit
before the rename) and never regenerated from the current tree: every strix-stream C ABI function and supported test
hook (ss_test_*) is still DEFINED, by name, in the library archive, and so is each ABI function's ts_ name. SS_API
functions added to ss.h after the baseline are checked the same way. include/strix_stream/ss.h still has the baseline's
SHA-256 (byte identity), or the SHA-256 of a
reviewed additive version listed after it (ss.h-sha256 lines). Internal C++ symbols are not ABI and are not compared.
Usage: check_old_symbols.py ARCHIVE
"""
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "include" / "strix_stream" / "ss.h"


def expected():
    rows = [l.split() for l in (ROOT / "tools" / "rename_abi_baseline.txt").read_text(encoding="utf-8").splitlines()
            if l and not l.startswith("#")]
    shas = {r[1] for r in rows if r[0] == "ss.h-sha256"}
    abi = {r[1] for r in rows if r[0] == "abi"}
    hooks = {r[1] for r in rows if r[0] == "hook"}
    hdr = re.sub(r"/\*.*?\*/", "", HEADER.read_text(encoding="utf-8"), flags=re.S)
    abi |= set(re.findall(r"\bSS_API\s+[^;{}]*?\b(ss_\w+)\s*\(", hdr, flags=re.S))
    return abi, hooks, shas


def defined(archive):
    """Names of the defined external symbols of a static library (nm, or dumpbin on MSVC)."""
    names = set()
    if shutil.which("nm") and not archive.lower().endswith(".lib"):
        out = subprocess.run(["nm", "-g", "--defined-only", archive], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[1] in ("T", "W"):
                names.add(parts[2])
        return names
    if shutil.which("dumpbin"):
        out = subprocess.run(["dumpbin", "/symbols", archive], capture_output=True, text=True, check=True).stdout
        for line in out.splitlines():
            # "... SECT3  notype ()    External     | ss_open" (defined) vs "UNDEF" (a reference)
            if "External" in line and "UNDEF" not in line and "|" in line:
                names.add(line.split("|", 1)[1].strip().split(" ")[0])
        return names
    print("check_old_symbols: neither nm nor dumpbin is available")
    sys.exit(2)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    abi, hooks, shas = expected()
    have_sha = hashlib.sha256(HEADER.read_bytes()).hexdigest()
    if have_sha not in shas:
        print("check_old_symbols: include/strix_stream/ss.h is not a reviewed version (sha256 %s)" % have_sha)
        return 1
    have = defined(sys.argv[1])
    missing = sorted(n for n in abi | hooks if n not in have)
    missing += sorted("ts_" + n[3:] for n in abi if ("ts_" + n[3:]) not in have)
    print("check_old_symbols: %d ABI functions, %d test hooks, %d defined names in %s"
          % (len(abi), len(hooks), len(have), sys.argv[1]))
    if missing:
        print("check_old_symbols: MISSING: " + ", ".join(missing))
        return 1
    print("check_old_symbols: ss.h unchanged; every baseline ss_ ABI function and test hook, and every ts_ ABI "
          "function, is defined")
    return 0


if __name__ == "__main__":
    sys.exit(main())
