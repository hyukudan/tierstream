#!/usr/bin/env python3
"""The rename's symbol gate (wave4/wstream/rename/design.md §2.1): every strix-stream C ABI function (each SS_API
function of include/strix_stream/ss.h) and every supported test hook (the extern "C" ss_test_* functions of
src/capi.cpp) is still DEFINED, by name, in the library archive, and so is its new ts_ name for the ABI functions.
Internal C++ symbols are not ABI and are not compared.
Usage: check_old_symbols.py ARCHIVE
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def expected():
    hdr = (ROOT / "include" / "strix_stream" / "ss.h").read_text(encoding="utf-8")
    hdr = re.sub(r"/\*.*?\*/", "", hdr, flags=re.S)
    abi = set(re.findall(r"\bSS_API\s+[^;{}]*?\b(ss_\w+)\s*\(", hdr, flags=re.S))
    capi = (ROOT / "src" / "capi.cpp").read_text(encoding="utf-8")
    hooks = set(re.findall(r"^(?:ss_status|uint64_t|int|void)\s+(ss_test_\w+)\s*\(", capi, flags=re.M))
    return abi, hooks


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
    abi, hooks = expected()
    have = defined(sys.argv[1])
    missing = sorted(n for n in abi | hooks if n not in have)
    missing += sorted("ts_" + n[3:] for n in abi if ("ts_" + n[3:]) not in have)
    print("check_old_symbols: %d ABI functions, %d test hooks, %d defined names in %s"
          % (len(abi), len(hooks), len(have), sys.argv[1]))
    if missing:
        print("check_old_symbols: MISSING: " + ", ".join(missing))
        return 1
    print("check_old_symbols: every ss_ ABI function and test hook, and every ts_ ABI function, is defined")
    return 0


if __name__ == "__main__":
    sys.exit(main())
