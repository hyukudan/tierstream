#!/usr/bin/env python3
"""Interim validator for strix-stream/recipe-v1 (wave4/recipes/core/schema-draft-5.md, Sol GO) plus the examples'
citation rules. Use `ss-recipe validate` for the schema once the core exists; the citation checks stay here.

Usage: python -I validate_v1.py [--root SPEC_ROOT] FILE_OR_GLOB...
Exit 0 when every file is valid.

Scope: structure (keys, types, ranges, references, route rules of §5) and citations. Descriptor-dependent compilation
(tiering, capacity, rings, derived workspace, list ids vs n_units) is the core's `ss-recipe plan`; whether a citation
actually supports its field is review, not this script.

Citation rules (x- keys, which the core keeps but never reads):
  [x-evidence.<key>] ref = "path:line" (relative to the lab spec root), quote (>= 12 chars, verbatim on that line), note
  [x-cites] "<settable path>" = "<key>" or ["<key>", ...]
  [[x-predicted]] route, metric, value, basis, evidence (key or non-empty list), optional scope fields
Every threshold field present in the recipe (CITED_LEAVES, see needs_cite) must be cited, every cite must name a
present field, every prediction needs evidence, and every evidence entry must be used.
"""
import glob
import math
import os
import re
import sys
import tomllib

SCHEMA = "strix-stream/recipe-v1"
TOP = {"schema", "name", "description", "generated_by", "pin", "match", "limits", "routes"}
MATCH = {"model", "model_sha256", "engine", "devices", "links"}
DEVICE = {"id", "kind", "count", "name", "memory_min_gib", "host_ram_min_gib", "disk_read_gbytes_ps_min",
          "h2d_gbytes_ps_min", "internal_link"}
LINK = {"id", "between", "kind", "gbps_min"}
LIMITS = {"host_free_ram_floor_gib", "host_free_commit_floor_gib"}
ROUTE = {"id", "phase", "enabled", "devices", "execute", "handoff_route", "links", "when", "fallback", "placement"}
WHEN = {"prompt_tokens_min", "prompt_tokens_max", "batch_min", "batch_max"}
BUDGET = {"device_gib", "host_pinned_gib", "workspace_gib", "max_context_tokens", "max_batch", "margin_gib",
          "margin_fraction", "dynamic_slots_per_layer", "ring_slots", "pinned_backing", "tier_mode"}
OPTIONS = {"policy", "lookahead_units", "reader_threads", "staging_buffers", "staging_mib", "copy_piece_mib",
           "score_window", "pin_planned", "fill_residents", "direct_io", "verify_hash", "group_units",
           "spec_persistent"}
ENGINE = {"prefetch_layers", "chunk_tokens", "routed_only_max_tokens", "idle_preload_layers", "max_copies_per_token"}
PLACEMENT = BUDGET | OPTIONS | ENGINE | {"units"}
UNITS = {"id", "class", "layers", "rank", "list", "resident", "pinned", "rest", "peer"}
SETTABLE = ([("limits", k) for k in LIMITS] +
            [("match", "devices", "*", k) for k in ("name", "memory_min_gib", "host_ram_min_gib",
                                                   "disk_read_gbytes_ps_min", "h2d_gbytes_ps_min")] +
            [("match", "links", "*", "gbps_min"), ("routes", "*", "fallback"), ("routes", "*", "enabled")] +
            [("routes", "*", "when", k) for k in WHEN] +
            [("routes", "*", "placement", "*", k) for k in BUDGET | OPTIONS | ENGINE] +
            [("routes", "*", "placement", "*", "units", "*", k) for k in ("layers", "rank", "list", "resident", "pinned")])
ID_RE = re.compile(r"[a-z0-9_-]{1,32}\Z")
NAME_RE = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")

CITED_LEAVES = {"prompt_tokens_min", "prompt_tokens_max", "batch_min", "batch_max", "max_batch", "memory_min_gib",
                "host_ram_min_gib", "disk_read_gbytes_ps_min", "gbps_min", "device_gib", "host_pinned_gib",
                "workspace_gib", "max_context_tokens", "dynamic_slots_per_layer", "ring_slots", "staging_mib",
                "staging_buffers", "prefetch_layers", "chunk_tokens", "routed_only_max_tokens", "lookahead_units",
                "resident", "pinned", "list", "host_free_ram_floor_gib", "margin_gib", "margin_fraction",
                "reader_threads", "copy_piece_mib", "score_window", "group_units", "layers", "h2d_gbytes_ps_min",
                "host_free_commit_floor_gib", "idle_preload_layers", "max_copies_per_token"}
ZERO_EXEMPT = {"prompt_tokens_min", "host_pinned_gib", "dynamic_slots_per_layer", "ring_slots", "chunk_tokens",
               "routed_only_max_tokens", "pinned", "resident", "prefetch_layers", "lookahead_units", "group_units",
               "idle_preload_layers", "max_copies_per_token", "host_free_commit_floor_gib", "h2d_gbytes_ps_min"}
BASIS = {"measured", "derived", "projected", "upper_bound", "lower_bound", "target"}
PRED = {"route", "metric", "value", "basis", "evidence", "at_prompt_tokens", "context_tokens", "concurrency",
        "batch", "scope", "date"}
EVID = {"ref", "quote", "note"}


class Checker:
    def __init__(self, path, root):
        self.path, self.root = path, root
        self.errors, self.leaves = [], {}

    def err(self, where, msg):
        self.errors.append(f"{self.path}: {where}: {msg}")

    def table(self, obj, where):
        if not isinstance(obj, dict):
            self.err(where, f"expected a table, got {type(obj).__name__}")
            return False
        return True

    def array(self, obj, where):
        if not isinstance(obj, list):
            self.err(where, f"expected an array, got {type(obj).__name__}")
            return []
        return obj

    def keys(self, obj, allowed, where):
        if not self.table(obj, where):
            return False
        for k in obj:
            if k.startswith("x-"):
                continue
            if k in ("exact", "exactness", "bit_exact"):
                self.err(where, f"'{k}': exactness is not a recipe option (every placement is bit-identical)")
            elif k not in allowed:
                self.err(where, f'unknown key "{k}"')
        return True

    def num(self, obj, k, where, lo=None, hi=None, integer=False, required=False, lo_open=False, hi_open=False):
        if k not in obj:
            if required:
                self.err(where, f"missing '{k}'")
            return None
        v = obj[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or (integer and not isinstance(v, int)):
            self.err(f"{where}.{k}", f"expected {'an integer' if integer else 'a number'}, got {v!r}")
            return None
        bad = (not math.isfinite(v) or (lo is not None and (v <= lo if lo_open else v < lo))
               or (hi is not None and (v >= hi if hi_open else v > hi)))
        if bad:
            self.err(f"{where}.{k}", f"{v} out of range {'(' if lo_open else '['}{lo}, {hi}{')' if hi_open else ']'}")
            return None
        self.leaves[f"{where}.{k}"] = v
        return v

    def boolean(self, obj, k, where):
        if k in obj and not isinstance(obj[k], bool):
            self.err(f"{where}.{k}", f"expected a boolean, got {obj[k]!r}")

    def ident(self, obj, where):
        i = obj.get("id") if isinstance(obj, dict) else None
        if not isinstance(i, str) or not ID_RE.match(i):
            self.err(where, f"bad or missing id {i!r}")
            return None
        return i


def validate(path, root):
    c = Checker(path, root)
    try:
        with open(path, "rb") as f:
            r = tomllib.load(f)
    except Exception as e:  # noqa: BLE001
        c.err("file", f"parse error: {e}")
        return c.errors
    c.keys(r, TOP, "top")
    if r.get("schema") != SCHEMA:
        c.err("schema", f"must be {SCHEMA!r}")
    if not isinstance(r.get("name"), str) or not NAME_RE.match(r["name"]):
        c.err("name", "required, [A-Za-z0-9._-]{1,64}")
    if r.get("generated_by", "human") not in ("human", "autotuner"):
        c.err("generated_by", "human | autotuner")
    if "description" in r and not isinstance(r["description"], str):
        c.err("description", "string")

    sets, links = {}, {}
    m = r.get("match")
    if m is None:
        c.err("match", "required")
    elif c.keys(m, MATCH, "match"):
        if not isinstance(m.get("model"), str) or not m.get("model"):
            c.err("match.model", "required")
        if "model_sha256" in m and not (isinstance(m["model_sha256"], str) and HEX64.match(m["model_sha256"])):
            c.err("match.model_sha256", "64 lowercase hex")
        if "engine" in m and not isinstance(m["engine"], str):
            c.err("match.engine", "string")
        for d in c.array(m.get("devices", []), "match.devices"):
            i = c.ident(d, "match.devices")
            w = f"match.devices.{i}"
            if i is None or not c.keys(d, DEVICE, w):
                continue
            if d.get("kind") not in ("cuda", "hip", "cpu"):
                c.err(f"{w}.kind", "cuda | hip | cpu")
            cnt = c.num(d, "count", w, 1, 64, integer=True) or 1
            c.num(d, "memory_min_gib", w, 0, 1048576, required=True, lo_open=True)
            c.num(d, "host_ram_min_gib", w, 0, 1048576)
            c.num(d, "disk_read_gbytes_ps_min", w, 0)
            if c.num(d, "h2d_gbytes_ps_min", w, 0) is not None and d.get("kind") != "cuda":
                c.err(f"{w}.h2d_gbytes_ps_min", "cuda sets only")
            if "name" in d and not isinstance(d["name"], str):
                c.err(f"{w}.name", "string")
            if "internal_link" in d and c.keys(d["internal_link"], {"kind", "gbps_min"}, f"{w}.internal_link"):
                if d["internal_link"].get("kind") not in ("usb4", "ethernet", "pcie", "other"):
                    c.err(f"{w}.internal_link.kind", "usb4 | ethernet | pcie | other")
                c.num(d["internal_link"], "gbps_min", f"{w}.internal_link", 0)
            if i in sets:
                c.err(w, "duplicate set id")
            sets[i] = dict(d, count=cnt)
        if not sets:
            c.err("match.devices", "at least one device set")
        for l in c.array(m.get("links", []), "match.links"):
            i = c.ident(l, "match.links")
            w = f"match.links.{i}"
            if i is None or not c.keys(l, LINK, w):
                continue
            b = l.get("between")
            if not (isinstance(b, list) and len(b) == 2 and all(isinstance(x, str) and x in sets for x in b)
                    and b[0] != b[1]):
                c.err(f"{w}.between", "two distinct declared set ids")
            if l.get("kind", "other") not in ("usb4", "ethernet", "pcie", "other"):
                c.err(f"{w}.kind", "usb4 | ethernet | pcie | other")
            c.num(l, "gbps_min", w, 0)
            if i in links:
                c.err(w, "duplicate link id")
            links[i] = l

    if "limits" in r and c.keys(r["limits"], LIMITS, "limits"):
        c.num(r["limits"], "host_free_ram_floor_gib", "limits", 0, 1048576)
        c.num(r["limits"], "host_free_commit_floor_gib", "limits", 0, 1048576)

    routes = [rt for rt in c.array(r.get("routes", []), "routes") if c.table(rt, "routes[]")]
    if not routes:
        c.err("routes", "required")
    byid = {}
    for rt in routes:
        i = c.ident(rt, "routes")
        if i in byid:
            c.err(f"routes.{i}", "duplicate route id")
        if i is not None:
            byid[i] = rt
    for i, rt in byid.items():
        route_checks(c, i, rt, byid, sets, links)
    for ph in ("prefill", "decode"):
        rs = [rt for rt in routes if rt.get("phase") == ph]
        if ph == "decode" and not rs:
            c.err("routes", "at least one decode route")
        if rs and rs[-1].get("when"):
            c.err(f"routes.{rs[-1].get('id')}", f"the last {ph} route must be a catch-all (empty when)")
    for i in byid:
        seen, cur = set(), i
        while isinstance(cur, str) and cur not in seen:
            seen.add(cur)
            cur = byid.get(cur, {}).get("fallback")
        if isinstance(cur, str):
            c.err(f"routes.{i}.fallback", "fallback chain cycles")
    pins = c.array(r.get("pin", []), "pin")
    for n, pth in enumerate(pins):
        if not (isinstance(pth, str) and pin_ok(pth, r)):
            c.err(f"pin[{n}]", f"{pth!r} is not a settable path (§6.2) or a segment prefix of one in this recipe")
        elif pins.count(pth) > 1:
            c.err(f"pin[{n}]", f"duplicate pin {pth!r}")
    citations(c, r, byid)
    return c.errors


def route_checks(c, i, rt, byid, sets, links):
    w = f"routes.{i}"
    if not c.keys(rt, ROUTE, w):
        return
    ph = rt.get("phase")
    if ph not in ("prefill", "decode"):
        c.err(f"{w}.phase", "prefill | decode")
    devs = rt.get("devices")
    if not (isinstance(devs, list) and devs and all(isinstance(d, str) and d in sets for d in devs)
            and len(set(devs)) == len(devs)):
        c.err(f"{w}.devices", "non-empty, distinct, declared set ids")
        devs = list(dict.fromkeys(d for d in devs if isinstance(d, str) and d in sets)) if isinstance(devs, list) else []
    ex = rt.get("execute")
    cnt = [sets[d]["count"] for d in devs]
    if ex == "single" and not (len(devs) == 1 and cnt[0] == 1):
        c.err(f"{w}.execute", "single = one set with count 1")
    elif ex in ("tp", "replicas") and not (len(devs) == 1 and cnt[0] >= 2):
        c.err(f"{w}.execute", f"{ex} = one set with count >= 2")
    elif ex == "hetero" and len(devs) < 2:
        c.err(f"{w}.execute", "hetero = >= 2 sets")
    elif ex not in ("single", "tp", "replicas", "hetero"):
        c.err(f"{w}.execute", "single | tp | replicas | hetero")
    c.boolean(rt, "enabled", w)
    hr = rt.get("handoff_route")
    if hr is not None and not isinstance(hr, str):
        c.err(f"{w}.handoff_route", "a route id string")
        hr = None
    if hr is not None and (ph != "prefill" or byid.get(hr, {}).get("phase") != "decode"):
        c.err(f"{w}.handoff_route", "prefill only, must name a decode route")
    rl = c.array(rt.get("links", []), f"{w}.links")
    for n, l in enumerate(rl):
        if not isinstance(l, str) or l not in links:
            c.err(f"{w}.links", f"unknown link {l!r}")
        elif l in rl[:n]:
            c.err(f"{w}.links", f"duplicate link {l!r}")
    hdevs = byid.get(hr, {}).get("devices", []) if hr else []
    nodes = set(devs) | {d for d in hdevs if isinstance(d, str) and d in sets} if isinstance(hdevs, list) else set(devs)
    if len(nodes) > 1:
        adj = {n: set() for n in nodes}
        for l in rl:
            b = links.get(l, {}).get("between", []) if isinstance(l, str) else []
            if isinstance(b, list) and len(b) == 2 and all(isinstance(x, str) for x in b) and b[0] in adj and b[1] in adj:
                adj[b[0]].add(b[1])
                adj[b[1]].add(b[0])
        seen, todo = set(), [next(iter(nodes))]
        while todo:
            n = todo.pop()
            if n not in seen:
                seen.add(n)
                todo += adj[n]
        if seen != nodes:
            c.err(f"{w}.links", f"do not connect {sorted(nodes)}")
    wh = rt.get("when", {})
    if c.keys(wh, WHEN, f"{w}.when"):
        lo = c.num(wh, "prompt_tokens_min", f"{w}.when", 0, integer=True)
        c.num(wh, "prompt_tokens_max", f"{w}.when", lo or 0, integer=True)
        bl = c.num(wh, "batch_min", f"{w}.when", 1, integer=True)
        c.num(wh, "batch_max", f"{w}.when", bl or 1, integer=True)
    fb = rt.get("fallback")
    if fb is not None and not isinstance(fb, str):
        c.err(f"{w}.fallback", "a route id string")
        fb = None
    if fb is not None and (fb == i or byid.get(fb, {}).get("phase") != ph):
        c.err(f"{w}.fallback", "a route of the same phase, not itself")
    pl = rt.get("placement", {})
    if not c.table(pl, f"{w}.placement"):
        return
    for s in devs:
        if s not in pl:
            c.err(f"{w}.placement", f"set {s!r} has no placement")
    for s, p in pl.items():
        placement_checks(c, f"{w}.placement.{s}", s, p, sets, devs, ex)


def placement_checks(c, pw, s, p, sets, devs, ex):
    if s not in devs:
        c.err(pw, "set not in the route's devices")
    if not c.keys(p, PLACEMENT, pw):
        return
    mem = sets.get(s, {}).get("memory_min_gib")
    dg = c.num(p, "device_gib", pw, 0, mem, lo_open=True)
    hr = sets.get(s, {}).get("host_ram_min_gib")
    hp = c.num(p, "host_pinned_gib", pw, 0, hr if hr else None)
    c.num(p, "workspace_gib", pw, 0, dg if dg is not None else mem, hi_open=True)
    c.num(p, "max_context_tokens", pw, 1, 2 ** 40, integer=True, required=True)
    c.num(p, "max_batch", pw, 1, 65536, integer=True, required=True)
    c.num(p, "margin_gib", pw, 0, 64)
    c.num(p, "margin_fraction", pw, 0, 0.5)
    for k, lo, hi in (("dynamic_slots_per_layer", 0, 65536), ("ring_slots", 0, 65536), ("lookahead_units", 0, 64),
                      ("reader_threads", 1, 64), ("staging_buffers", 2, 64), ("staging_mib", 1, 16384),
                      ("copy_piece_mib", 1, 4096), ("score_window", 1, 4096), ("group_units", 0, 65536),
                      ("prefetch_layers", 0, 8), ("chunk_tokens", 0, 1048576), ("routed_only_max_tokens", 0, 1048576),
                      ("idle_preload_layers", 0, 64), ("max_copies_per_token", 0, 65536)):
        c.num(p, k, pw, lo, hi, integer=True)
    for k in ("pinned_backing", "pin_planned", "fill_residents", "direct_io", "verify_hash", "spec_persistent"):
        c.boolean(p, k, pw)
    if p.get("policy", "lru") not in ("lru", "scored"):
        c.err(f"{pw}.policy", "lru | scored")
    if p.get("tier_mode", "inclusive") not in ("inclusive", "exclusive"):
        c.err(f"{pw}.tier_mode", "inclusive | exclusive")
    groups = c.array(p.get("units", []), f"{pw}.units")
    if ex == "hetero" and not groups:
        c.err(pw, "hetero placements need unit groups")
    gids, seen_layers, all_layers = set(), {}, {}
    for g in groups:
        gi = c.ident(g, f"{pw}.units")
        gw = f"{pw}.units.{gi}"
        if gi is None or not c.keys(g, UNITS, gw):
            continue
        if gi in gids:
            c.err(gw, "duplicate group id")
        gids.add(gi)
        cls = g.get("class")
        if cls not in ("moe", "blob"):
            c.err(f"{gw}.class", "moe | blob")
        rank = g.get("rank", "model")
        if rank not in ("model", "list", "frequency"):
            c.err(f"{gw}.rank", "model | list | frequency")
        lst = g.get("list")
        if (rank == "list") != (lst is not None):
            c.err(f"{gw}.list", "required iff rank = 'list'")
        if lst is not None:
            if not (isinstance(lst, list) and lst and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                                                          for x in lst) and len(set(lst)) == len(lst)):
                c.err(f"{gw}.list", "non-empty, distinct integers >= 0")
            else:
                c.leaves[f"{gw}.list"] = lst
        for k in ("resident", "pinned"):
            if g.get(k) != "fill":
                c.num(g, k, gw, 0, integer=True)
        if (g.get("pinned") == "fill" or (isinstance(g.get("pinned"), int) and g["pinned"] > 0)
                or g.get("rest") == "pinned") and not hp:
            c.err(gw, "pinned units need host_pinned_gib > 0")
        rest = g.get("rest", "disk")
        if rest not in ("disk", "pinned", "peer"):
            c.err(f"{gw}.rest", "disk | pinned | peer")
        if rest == "pinned" and g.get("pinned", 0) == "fill":
            c.err(gw, "rest = 'pinned' with pinned = 'fill' is refused")
        if (rest == "peer") != ("peer" in g):
            c.err(f"{gw}.peer", "required iff rest = 'peer'")
        if rest == "peer":
            if ex != "hetero":
                c.err(f"{gw}.rest", "peer only in hetero routes")
            elif g["peer"] not in devs or devs.index(g["peer"]) <= devs.index(s):
                c.err(f"{gw}.peer", "a set after this one in the route's devices")
        if "layers" not in g:
            if cls in all_layers or any(k[0] == cls for k in seen_layers):
                c.err(f"{gw}.layers", f"default (every layer) overlaps another {cls} group")
            all_layers[cls] = gi
        else:
            if cls in all_layers:
                c.err(f"{gw}.layers", f"overlaps group {all_layers[cls]} (every layer)")
            idx = []
            lay = g["layers"]
            ok = isinstance(lay, list) and bool(lay)
            for e in lay if ok else []:
                if isinstance(e, int) and not isinstance(e, bool) and e >= 0:
                    idx.append((e, e))
                elif (isinstance(e, list) and len(e) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                                  and x >= 0 for x in e) and e[0] <= e[1]):
                    idx.append((e[0], e[1]))
                else:
                    ok = False
            if not ok:
                c.err(f"{gw}.layers", "non-empty array of indices >= 0 and [first, last] ranges, first <= last")
            else:
                c.leaves[f"{gw}.layers"] = lay
            for a, b in idx:
                for L in range(a, b + 1):
                    if (cls, L) in seen_layers:
                        c.err(f"{gw}.layers", f"layer {L} already in group {seen_layers[(cls, L)]}")
                    seen_layers[(cls, L)] = gi


def pin_ok(pth, r):
    """A pin is a settable path (§6.2) or a segment prefix of one, resolved in context: the route, its placement set
    (one of the route's devices) and the group of that placement must exist."""
    segs = pth.split(".")
    m = r.get("match", {}) if isinstance(r.get("match"), dict) else {}
    routes = {rt.get("id"): rt for rt in r.get("routes", []) if isinstance(rt, dict)}

    def ids(v):
        return {d.get("id") for d in v if isinstance(d, dict)} if isinstance(v, list) else set()

    for pat in SETTABLE:
        if len(segs) > len(pat):
            continue
        ok, rt, pl = True, None, None
        for i, (sg, pt) in enumerate(zip(segs, pat)):
            if pt != "*":
                ok = sg == pt
            elif pat[i - 1] == "devices":
                ok = sg in ids(m.get("devices"))
            elif pat[i - 1] == "links":
                ok = sg in ids(m.get("links"))
            elif pat[i - 1] == "routes":
                rt = routes.get(sg)
                ok = rt is not None
            elif pat[i - 1] == "placement":
                devs = rt.get("devices") if isinstance(rt, dict) else None
                ok = isinstance(devs, list) and sg in devs
                pl = (rt.get("placement") or {}).get(sg) if ok and isinstance(rt.get("placement"), dict) else None
            elif pat[i - 1] == "units":
                ok = isinstance(pl, dict) and sg in ids(pl.get("units"))
            if not ok:
                break
        if ok:
            return True
    return False


def needs_cite(leaf_path, value):
    leaf = leaf_path.rsplit(".", 1)[-1]
    if leaf not in CITED_LEAVES:
        return False
    if leaf in ("max_batch", "batch_max", "batch_min") and value == 1:
        return False  # one sequence = the engines' one-prompt contract, not a calibrated threshold
    if leaf in ZERO_EXEMPT and value == 0:
        return False
    return True


def citations(c, r, byid):
    ev = r.get("x-evidence", {})
    cites = r.get("x-cites", {})
    if not (c.table(ev, "x-evidence") and c.table(cites, "x-cites")):
        return
    preds = c.array(r.get("x-predicted", []), "x-predicted")
    used = set()
    for key, e in ev.items():
        w = f"x-evidence.{key}"
        if not c.keys(e, EVID, w):
            continue
        ref, quote = e.get("ref", ""), e.get("quote", "")
        fpath, _, line = ref.rpartition(":") if isinstance(ref, str) else ("", "", "")
        if not fpath or not line.isdigit():
            c.err(w, f"ref must be 'path:line', got {ref!r}")
            continue
        try:
            with open(os.path.join(c.root, fpath), encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError:
            c.err(w, f"file not found: {fpath}")
            continue
        n = int(line)
        if not 1 <= n <= len(lines):
            c.err(w, f"line {n} outside 1..{len(lines)}")
        elif not isinstance(quote, str) or len(quote) < 12 or quote not in lines[n - 1]:
            c.err(w, f"quote not found verbatim (>= 12 chars) on {ref}")

    def keys_of(v, w):
        ks = v if isinstance(v, list) else [v]
        if not ks:
            c.err(w, "empty evidence list")
        for k in ks:
            if not isinstance(k, str) or k not in ev:
                c.err(w, f"evidence {k!r} not in [x-evidence]")
            else:
                used.add(k)

    for field, v in cites.items():
        keys_of(v, f"x-cites.'{field}'")
        if field not in c.leaves:
            c.err(f"x-cites.'{field}'", "no such field in the recipe")
    for leaf, v in c.leaves.items():
        if needs_cite(leaf, v) and leaf not in cites:
            c.err(leaf, "threshold without a citation in [x-cites]")
    for n, p in enumerate(preds):
        w = f"x-predicted[{n}]"
        if not c.keys(p, PRED, w):
            continue
        if p.get("route") not in byid:
            c.err(f"{w}.route", f"{p.get('route')!r} is not a route")
        if not isinstance(p.get("metric"), str) or not p.get("metric"):
            c.err(f"{w}.metric", "required")
        v = p.get("value")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            c.err(f"{w}.value", "finite number > 0")
        if p.get("basis") not in BASIS:
            c.err(f"{w}.basis", f"one of {sorted(BASIS)}")
        if "evidence" not in p:
            c.err(w, "evidence required")
        else:
            keys_of(p["evidence"], f"{w}.evidence")
    if not preds:
        c.err("x-predicted", "at least one prediction")
    for key in ev:
        if key not in used:
            c.err(f"x-evidence.{key}", "not used by any cite or prediction")


def main(argv):
    root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    pats = []
    it = iter(argv)
    for a in it:
        if a == "--root":
            root = next(it)
        else:
            pats.append(a)
    files = []
    for p in pats:
        hits = sorted(glob.glob(p))
        files += hits if hits else [p]
    bad = 0
    for f in files:
        errs = validate(f, root)
        print("\n".join(errs) if errs else f"{f}: OK")
        bad += bool(errs)
    return 1 if bad or not files else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
