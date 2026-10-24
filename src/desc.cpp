#include "desc.hpp"

#include <cstdio>
#include <set>

#include "errors.hpp"

namespace ss {
namespace {

constexpr const char* kSchema = "strix-stream/desc-v1";
constexpr int kMaxLayers = 4096;
constexpr int kMaxUnits = 1 << 16;

std::string at(const std::string& base, size_t i) { return base + "[" + std::to_string(i) + "]"; }

const Json& need(const Json& o, const std::string& key, const std::string& where) {
  const Json* v = o.find(key);
  require(v != nullptr, SS_E_DESC, where + ": missing \"" + key + "\"");
  return *v;
}

uint64_t opt_u64(const Json& o, const std::string& key, const std::string& where, uint64_t dflt) {
  const Json* v = o.find(key);
  return v ? json_u64(*v, where + "." + key) : dflt;
}

std::string opt_str(const Json& o, const std::string& key, const std::string& where, const std::string& dflt = "") {
  const Json* v = o.find(key);
  return v ? json_str(*v, where + "." + key) : dflt;
}

void check_keys(const Json& o, const std::string& where, std::initializer_list<const char*> allowed) {
  require(o.is_object(), SS_E_DESC, where + ": expected an object");
  std::set<std::string> ok;
  for (const char* k : allowed) ok.insert(k);
  for (const auto& kv : o.obj) {
    // "x-" keys are free-form adapter notes; anything else unknown is a typo or a newer schema.
    if (kv.first.rfind("x-", 0) == 0) continue;
    require(ok.count(kv.first) != 0, SS_E_DESC, where + ": unknown key \"" + kv.first + "\"");
  }
}

std::vector<Named> named_list(const Json* v, const std::string& where) {
  std::vector<Named> out;
  if (!v) return out;
  require(v->is_array(), SS_E_DESC, where + ": expected an array");
  for (size_t i = 0; i < v->arr.size(); ++i) {
    const Json& e = v->arr[i];
    std::string w = at(where, i);
    check_keys(e, w, {"name", "bytes"});
    out.push_back({json_str(need(e, "name", w), w + ".name"), json_u64(need(e, "bytes", w), w + ".bytes")});
  }
  return out;
}

bool format_known(const std::string& f) {
  static const std::set<std::string> known = {
      "F32", "F16", "BF16", "FP8_E4M3", "FP8_E5M2", "EXL3", "MXFP4", "NVFP4", "Q2_K", "Q3_K", "Q4_K", "Q5_K",
      "Q6_K", "Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS",
      "IQ3_S", "IQ4_NL", "IQ4_XS", "TQ1_0", "TQ2_0", "I8", "U8", "mixed"};
  return known.count(f) != 0 || f.rfind("x-", 0) == 0;
}

std::string hex16(uint64_t v) {
  char buf[17];
  std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

uint64_t parse_hex16(const std::string& s, const std::string& where) {
  require(s.size() == 16, SS_E_DESC, where + ": expected 16 hex digits");
  uint64_t v = 0;
  for (char c : s) {
    v <<= 4;
    if (c >= '0' && c <= '9') v |= uint64_t(c - '0');
    else if (c >= 'a' && c <= 'f') v |= uint64_t(c - 'a' + 10);
    else fail(SS_E_DESC, where + ": expected lowercase hex digits");
  }
  return v;
}

}  // namespace

const char* tiers_name(Tiers t) {
  switch (t) {
    case Tiers::Pinned: return "pinned";
    case Tiers::Disk: return "disk";
    case Tiers::PinnedDisk: return "pinned|disk";
    case Tiers::Pair: return "pair";
  }
  return "?";
}

const char* order_name(Order o) { return o == Order::Prefix ? "prefix" : "ranked"; }

uint64_t Desc::unit_stride(int layer) const {
  uint64_t b = layers[size_t(layer)].unit_bytes;
  return (b + unit_align - 1) / unit_align * unit_align;
}

uint64_t Desc::fixed_total() const {
  uint64_t t = 0;
  for (const auto& n : fixed_device) t += n.bytes;
  for (const auto& l : layers) {
    t += l.fixed_bytes;
    if (!l.streamable) t += uint64_t(l.n_units) * l.unit_bytes;
  }
  return t;
}

Desc desc_from_json(const Json& j) {
  const std::string w = "desc";
  check_keys(j, w,
             {"schema", "model", "engine", "fixed_device", "host_fixed", "state_bytes_per_position",
              "workspace_bytes_per_row", "unit_align", "sources", "order", "tiers", "ring_slots", "routes", "layers",
              "lookup_tables"});
  require(json_str(need(j, "schema", w), "desc.schema") == kSchema, SS_E_DESC,
          std::string("desc.schema: expected \"") + kSchema + "\"");
  Desc d;
  d.model = opt_str(j, "model", w);
  d.engine = opt_str(j, "engine", w);
  d.fixed_device = named_list(j.find("fixed_device"), "desc.fixed_device");
  d.host_fixed = named_list(j.find("host_fixed"), "desc.host_fixed");
  d.state_bytes_per_position = opt_u64(j, "state_bytes_per_position", w, 0);
  d.workspace_bytes_per_row = opt_u64(j, "workspace_bytes_per_row", w, 0);
  d.unit_align = opt_u64(j, "unit_align", w, 256);
  d.ring_slots = uint32_t(opt_u64(j, "ring_slots", w, 0));
  d.routes = opt_str(j, "routes", w);

  std::string order = opt_str(j, "order", w, "ranked");
  if (order == "prefix") d.order = Order::Prefix;
  else if (order == "ranked") d.order = Order::Ranked;
  else fail(SS_E_DESC, "desc.order: expected prefix | ranked");

  std::string tiers = opt_str(j, "tiers", w, "pinned|disk");
  if (tiers == "pinned") d.tiers = Tiers::Pinned;
  else if (tiers == "disk") d.tiers = Tiers::Disk;
  else if (tiers == "pinned|disk") d.tiers = Tiers::PinnedDisk;
  else if (tiers == "pair") d.tiers = Tiers::Pair;
  else fail(SS_E_DESC, "desc.tiers: expected pinned | disk | pinned|disk | pair");

  if (const Json* s = j.find("sources")) {
    require(s->is_array(), SS_E_DESC, "desc.sources: expected an array");
    for (size_t i = 0; i < s->arr.size(); ++i) {
      std::string sw = at("desc.sources", i);
      check_keys(s->arr[i], sw, {"name", "path", "align"});
      Source src;
      src.name = opt_str(s->arr[i], "name", sw);
      src.path = json_str(need(s->arr[i], "path", sw), sw + ".path");
      src.align = opt_u64(s->arr[i], "align", sw, 4096);
      d.sources.push_back(src);
    }
  }

  const Json layers = need(j, "layers", w);
  require(layers.is_array(), SS_E_DESC, "desc.layers: expected an array");
  for (size_t i = 0; i < layers.arr.size(); ++i) {
    const Json& lj = layers.arr[i];
    std::string lw = at("desc.layers", i);
    check_keys(lj, lw,
               {"index", "kind", "fixed_bytes", "n_units", "top_k", "n_shared", "unit_bytes", "parts", "router",
                "hash", "benefit", "streamable"});
    Layer l;
    l.index = int(json_i64(need(lj, "index", lw), lw + ".index"));
    std::string kind = json_str(need(lj, "kind", lw), lw + ".kind");
    if (kind == "dense") l.kind = LayerKind::Dense;
    else if (kind == "moe") l.kind = LayerKind::Moe;
    else if (kind == "blob") l.kind = LayerKind::Blob;
    else fail(SS_E_DESC, lw + ".kind: expected dense | moe | blob");
    l.fixed_bytes = opt_u64(lj, "fixed_bytes", lw, 0);
    l.n_units = int(opt_u64(lj, "n_units", lw, l.kind == LayerKind::Blob ? 1 : 0));
    l.top_k = int(opt_u64(lj, "top_k", lw, 0));
    l.n_shared = int(opt_u64(lj, "n_shared", lw, 0));
    l.unit_bytes = opt_u64(lj, "unit_bytes", lw, 0);
    if (const Json* st = lj.find("streamable")) l.streamable = json_bool(*st, lw + ".streamable");
    if (const Json* parts = lj.find("parts")) {
      require(parts->is_array(), SS_E_DESC, lw + ".parts: expected an array");
      for (size_t p = 0; p < parts->arr.size(); ++p) {
        const Json& pj = parts->arr[p];
        std::string pw = at(lw + ".parts", p);
        check_keys(pj, pw, {"name", "format", "source", "offset", "stride", "bytes"});
        Part part;
        part.name = opt_str(pj, "name", pw);
        part.format = json_str(need(pj, "format", pw), pw + ".format");
        part.source = int(opt_u64(pj, "source", pw, 0));
        part.offset = json_u64(need(pj, "offset", pw), pw + ".offset");
        part.bytes = json_u64(need(pj, "bytes", pw), pw + ".bytes");
        part.stride = opt_u64(pj, "stride", pw, part.bytes);
        l.parts.push_back(part);
      }
    }
    if (const Json* r = lj.find("router")) {
      std::string rw = lw + ".router";
      check_keys(*r, rw, {"tensor", "bias", "format", "score", "rows", "cols"});
      l.has_router = true;
      l.router.tensor = json_str(need(*r, "tensor", rw), rw + ".tensor");
      l.router.bias = opt_str(*r, "bias", rw);
      l.router.format = opt_str(*r, "format", rw, "F32");
      l.router.score = opt_str(*r, "score", rw, "softmax");
      l.router.rows = uint32_t(opt_u64(*r, "rows", rw, 0));
      l.router.cols = uint32_t(opt_u64(*r, "cols", rw, 0));
    }
    if (const Json* h = lj.find("hash")) {
      require(h->is_array(), SS_E_DESC, lw + ".hash: expected an array of xxh64 hex strings");
      for (size_t k = 0; k < h->arr.size(); ++k)
        l.hash.push_back(parse_hex16(json_str(h->arr[k], at(lw + ".hash", k)), at(lw + ".hash", k)));
    }
    if (const Json* b = lj.find("benefit")) {
      require(b->is_array(), SS_E_DESC, lw + ".benefit: expected an array");
      for (size_t k = 0; k < b->arr.size(); ++k) l.benefit.push_back(json_f64(b->arr[k], at(lw + ".benefit", k)));
    }
    d.layers.push_back(std::move(l));
  }

  if (const Json* t = j.find("lookup_tables")) {
    require(t->is_array(), SS_E_DESC, "desc.lookup_tables: expected an array");
    for (size_t i = 0; i < t->arr.size(); ++i) {
      std::string tw = at("desc.lookup_tables", i);
      check_keys(t->arr[i], tw, {"name", "source", "offset", "rows", "row_bytes", "placement"});
      LookupTable lt;
      lt.name = json_str(need(t->arr[i], "name", tw), tw + ".name");
      lt.source = int(opt_u64(t->arr[i], "source", tw, 0));
      lt.offset = opt_u64(t->arr[i], "offset", tw, 0);
      lt.rows = json_u64(need(t->arr[i], "rows", tw), tw + ".rows");
      lt.row_bytes = json_u64(need(t->arr[i], "row_bytes", tw), tw + ".row_bytes");
      lt.placement = opt_str(t->arr[i], "placement", tw, "file");
      d.lookup_tables.push_back(lt);
    }
  }
  desc_validate(d);
  return d;
}

void desc_validate(Desc& d) {
  require(!d.layers.empty() && d.layers.size() <= size_t(kMaxLayers), SS_E_DESC, "desc.layers: 1..4096 layers");
  require(d.unit_align >= 1 && (d.unit_align & (d.unit_align - 1)) == 0, SS_E_DESC,
          "desc.unit_align: power of two");
  for (size_t s = 0; s < d.sources.size(); ++s) {
    uint64_t a = d.sources[s].align;
    require(a >= 1 && (a & (a - 1)) == 0 && a <= (1u << 20), SS_E_DESC,
            at("desc.sources", s) + ".align: power of two <= 1 MiB");
  }
  d.unit_base.assign(d.layers.size(), 0);
  d.total_units = 0;
  d.max_unit_bytes = 0;
  d.max_units_per_layer = 0;
  for (size_t i = 0; i < d.layers.size(); ++i) {
    Layer& l = d.layers[i];
    std::string lw = at("desc.layers", i);
    require(l.index == int(i), SS_E_DESC, lw + ".index: layers must be listed in order 0..n-1");
    require(l.n_units >= 0 && l.n_units <= kMaxUnits, SS_E_DESC, lw + ".n_units: 0..65536");
    if (l.kind == LayerKind::Dense) {
      require(l.n_units == 0 && l.parts.empty(), SS_E_DESC, lw + ": a dense layer has no units (use fixed_bytes)");
    } else if (l.kind == LayerKind::Blob) {
      require(l.n_units == 1, SS_E_DESC, lw + ": a blob layer has exactly one unit");
    } else {
      require(l.n_units >= 1, SS_E_DESC, lw + ": a moe layer needs n_units >= 1");
      require(l.top_k >= 1 && l.top_k <= l.n_units, SS_E_DESC, lw + ".top_k: 1..n_units");
    }
    if (l.n_units > 0) {
      require(!l.parts.empty(), SS_E_DESC, lw + ".parts: a layer with units needs parts");
      uint64_t sum = 0;
      for (size_t p = 0; p < l.parts.size(); ++p) {
        const Part& part = l.parts[p];
        std::string pw = at(lw + ".parts", p);
        require(part.bytes > 0, SS_E_DESC, pw + ".bytes: > 0");
        require(format_known(part.format), SS_E_DESC, pw + ".format: unknown format " + part.format);
        require(part.source >= 0 && size_t(part.source) < d.sources.size(), SS_E_DESC, pw + ".source: no such source");
        require(l.n_units == 1 || part.stride >= part.bytes, SS_E_DESC, pw + ".stride: >= bytes (units overlap)");
        sum += part.bytes;
      }
      if (l.unit_bytes == 0) l.unit_bytes = sum;
      require(l.unit_bytes == sum, SS_E_DESC,
              lw + ".unit_bytes: " + std::to_string(l.unit_bytes) + " != sum of parts " + std::to_string(sum));
      require(l.hash.empty() || l.hash.size() == size_t(l.n_units), SS_E_DESC, lw + ".hash: one per unit");
      require(l.benefit.empty() || l.benefit.size() == size_t(l.n_units), SS_E_DESC, lw + ".benefit: one per unit");
      if (l.has_router)
        require(l.router.rows == 0 || l.router.rows == uint32_t(l.n_units), SS_E_DESC,
                lw + ".router.rows: must equal n_units");
      if (l.streamable) {
        if (l.unit_bytes > d.max_unit_bytes) d.max_unit_bytes = l.unit_bytes;
        if (l.n_units > d.max_units_per_layer) d.max_units_per_layer = l.n_units;
      }
    }
    d.unit_base[i] = d.total_units;
    d.total_units += l.n_units;
  }
  for (size_t i = 0; i < d.lookup_tables.size(); ++i) {
    const LookupTable& t = d.lookup_tables[i];
    std::string tw = at("desc.lookup_tables", i);
    require(t.source >= 0 && size_t(t.source) < d.sources.size(), SS_E_DESC, tw + ".source: no such source");
    require(t.placement == "file" || t.placement == "pinned" || t.placement == "device", SS_E_DESC,
            tw + ".placement: file | pinned | device");
  }
}

Json desc_to_json(const Desc& d) {
  Json j = Json::object();
  j["schema"] = Json::string(kSchema);
  if (!d.model.empty()) j["model"] = Json::string(d.model);
  if (!d.engine.empty()) j["engine"] = Json::string(d.engine);
  auto named = [](const std::vector<Named>& v) {
    Json a = Json::array();
    for (const auto& n : v) {
      Json e = Json::object();
      e["name"] = Json::string(n.name);
      e["bytes"] = Json::number(n.bytes);
      a.push(e);
    }
    return a;
  };
  j["fixed_device"] = named(d.fixed_device);
  j["host_fixed"] = named(d.host_fixed);
  j["state_bytes_per_position"] = Json::number(d.state_bytes_per_position);
  j["workspace_bytes_per_row"] = Json::number(d.workspace_bytes_per_row);
  j["unit_align"] = Json::number(d.unit_align);
  j["order"] = Json::string(order_name(d.order));
  j["tiers"] = Json::string(tiers_name(d.tiers));
  j["ring_slots"] = Json::number(uint64_t(d.ring_slots));
  if (!d.routes.empty()) j["routes"] = Json::string(d.routes);
  Json srcs = Json::array();
  for (const auto& s : d.sources) {
    Json e = Json::object();
    if (!s.name.empty()) e["name"] = Json::string(s.name);
    e["path"] = Json::string(s.path);
    e["align"] = Json::number(s.align);
    srcs.push(e);
  }
  j["sources"] = srcs;
  Json layers = Json::array();
  for (const auto& l : d.layers) {
    Json e = Json::object();
    e["index"] = Json::number(int64_t(l.index));
    e["kind"] = Json::string(l.kind == LayerKind::Dense ? "dense" : l.kind == LayerKind::Moe ? "moe" : "blob");
    e["fixed_bytes"] = Json::number(l.fixed_bytes);
    if (l.n_units > 0) {
      e["n_units"] = Json::number(int64_t(l.n_units));
      e["unit_bytes"] = Json::number(l.unit_bytes);
      e["streamable"] = Json::boolean(l.streamable);
      Json parts = Json::array();
      for (const auto& p : l.parts) {
        Json pj = Json::object();
        if (!p.name.empty()) pj["name"] = Json::string(p.name);
        pj["format"] = Json::string(p.format);
        pj["source"] = Json::number(int64_t(p.source));
        pj["offset"] = Json::number(p.offset);
        pj["stride"] = Json::number(p.stride);
        pj["bytes"] = Json::number(p.bytes);
        parts.push(pj);
      }
      e["parts"] = parts;
      if (!l.hash.empty()) {
        Json h = Json::array();
        for (uint64_t v : l.hash) h.push(Json::string(hex16(v)));
        e["hash"] = h;
      }
      if (!l.benefit.empty()) {
        Json b = Json::array();
        for (double v : l.benefit) b.push(Json::number(v));
        e["benefit"] = b;
      }
    }
    if (l.top_k) e["top_k"] = Json::number(int64_t(l.top_k));
    if (l.n_shared) e["n_shared"] = Json::number(int64_t(l.n_shared));
    if (l.has_router) {
      Json r = Json::object();
      r["tensor"] = Json::string(l.router.tensor);
      if (!l.router.bias.empty()) r["bias"] = Json::string(l.router.bias);
      r["format"] = Json::string(l.router.format);
      r["score"] = Json::string(l.router.score);
      r["rows"] = Json::number(uint64_t(l.router.rows));
      r["cols"] = Json::number(uint64_t(l.router.cols));
      e["router"] = r;
    }
    layers.push(e);
  }
  j["layers"] = layers;
  Json tables = Json::array();
  for (const auto& t : d.lookup_tables) {
    Json e = Json::object();
    e["name"] = Json::string(t.name);
    e["source"] = Json::number(int64_t(t.source));
    e["offset"] = Json::number(t.offset);
    e["rows"] = Json::number(t.rows);
    e["row_bytes"] = Json::number(t.row_bytes);
    e["placement"] = Json::string(t.placement);
    tables.push(e);
  }
  j["lookup_tables"] = tables;
  return j;
}

}  // namespace ss
