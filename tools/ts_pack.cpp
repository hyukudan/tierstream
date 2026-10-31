// ts-pack: build, verify and inspect tierstream/pack-v1 packs (docs/pack.md).
//   ts-pack build LAYOUT.json OUT_DIR [--dry-run] [--split-gib N] [--trust-source-sha256] [--layouts A,B|none]
//   ts-pack verify DIR [--pin SHA256] [--full]
//   ts-pack show DIR [--records]
//   ts-pack cat DIR LIST OUT [--verify cached|always|off]   LIST: lines "layer expert name [row0 row1]"
//   ts-pack desc DIR BASE_DESC.json OUT.json
// build reads an engine-provided layout (tierstream/pack-layout-v1): which byte ranges of which source files make
// each record, optionally as a row gather (rows and byte pieces of each source row), so an engine whose device format
// is a byte rearrangement of its model file needs no code here; any other transform runs in the engine through the
// writer API (ss_pack_writer_*).
#include <algorithm>
#include <climits>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <future>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "errors.hpp"
#include "json.hpp"
#include "pack.hpp"
#include "raw_file.hpp"
#include "sha256.hpp"

using namespace ts;

namespace {

constexpr const char* kLayoutSchema = "tierstream/pack-layout-v1";

std::string read_text(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  require(bool(f), SS_E_IO, "cannot open " + path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

struct Gather {
  uint64_t src_row_bytes = 0;
  std::vector<std::pair<uint32_t, uint32_t>> rows;    // source row ranges, in output order
  std::vector<std::pair<uint64_t, uint64_t>> pieces;  // (offset, bytes) of each source row, in output order
  uint32_t out_rows() const {
    uint64_t n = 0;
    for (const auto& r : rows) n += r.second - r.first;
    return uint32_t(n);
  }
  uint64_t out_row_bytes() const {
    uint64_t n = 0;
    for (const auto& p : pieces) n += p.second;
    return n;
  }
};

struct Tensor {
  int32_t layer = -1;
  std::string name, format;
  uint32_t source = 0;
  uint64_t offset = 0, stride = 0, bytes = 0;  // source unit u: [offset + u * stride, + bytes)
  uint32_t units = 0, first_unit = 0;          // 0 units = one dense record (expert -1)
  uint32_t rows = 0;
  uint64_t row_bytes = 0;
  bool rows_given = false;
  std::string layout;  // optional layout this tensor belongs to ("" = always built)
  std::vector<uint32_t> seg_rows;
  bool gather = false;
  Gather g;
  uint64_t out_bytes() const { return gather ? uint64_t(g.out_rows()) * g.out_row_bytes() : bytes; }
};

struct LayoutSource {
  std::string name, path, sha256;
};

struct Layout {
  std::string model;
  std::vector<LayoutSource> sources;
  std::map<std::string, PackFormatEntry> formats;
  uint64_t split_bytes = 0;
  Json meta = Json::object(), receipt = Json::object();
  std::vector<Tensor> tensors;
};

uint32_t u32_of(const Json& v, const std::string& w) {
  const uint64_t x = json_u64(v, w);
  require(x <= INT32_MAX, SS_E_DESC, w + ": out of range");
  return uint32_t(x);
}
uint64_t add_checked(uint64_t a, uint64_t b, const std::string& w) {
  require(b <= UINT64_MAX - a, SS_E_DESC, w + ": out of range");
  return a + b;
}
uint64_t mul_checked(uint64_t a, uint64_t b, const std::string& w) {
  require(a == 0 || b <= UINT64_MAX / a, SS_E_DESC, w + ": out of range");
  return a * b;
}

const Json& key(const Json& o, const char* k, const std::string& w) {
  const Json* v = o.find(k);
  require(v != nullptr, SS_E_DESC, w + ": missing " + k);
  return *v;
}

Layout parse_layout(const std::string& text) {
  const Json j = json_parse(text);
  require(j.is_object(), SS_E_DESC, "layout: expected an object");
  for (const auto& kv : j.obj)
    require(kv.first.rfind("x-", 0) == 0 || kv.first == "schema" || kv.first == "model" || kv.first == "sources" ||
                kv.first == "formats" || kv.first == "split_bytes" || kv.first == "meta" || kv.first == "receipt" ||
                kv.first == "tensors",
            SS_E_DESC, "layout: unknown key " + kv.first);
  require(json_str(key(j, "schema", "layout"), "schema") == kLayoutSchema, SS_E_DESC,
          std::string("layout: schema is not ") + kLayoutSchema);
  Layout L;
  const std::string top = "layout";
  L.model = json_str(key(j, "model", top), "model");
  if (const Json* v = j.find("split_bytes")) L.split_bytes = json_u64(*v, "split_bytes");
  if (const Json* v = j.find("meta")) L.meta = *v;
  if (const Json* v = j.find("receipt")) L.receipt = *v;
  const Json& src = key(j, "sources", top);
  require(src.is_array() && !src.arr.empty(), SS_E_DESC, "layout: sources");
  for (size_t i = 0; i < src.arr.size(); ++i) {
    const std::string w = "sources[" + std::to_string(i) + "]";
    LayoutSource s;
    s.name = json_str(key(src.arr[i], "name", w), w + ".name");
    s.path = json_str(key(src.arr[i], "path", w), w + ".path");
    if (const Json* v = src.arr[i].find("sha256")) s.sha256 = json_str(*v, w + ".sha256");
    L.sources.push_back(s);
  }
  const Json& fm = key(j, "formats", top);
  require(fm.is_object(), SS_E_DESC, "layout: formats");
  for (const auto& kv : fm.obj) {
    const std::string w = "formats." + kv.first;
    PackFormatEntry e;
    e.transform = json_str(key(kv.second, "transform", w), w + ".transform");
    e.version = json_str(key(kv.second, "version", w), w + ".version");
    e.tool = json_str(key(kv.second, "tool", w), w + ".tool");
    L.formats[kv.first] = e;
  }
  const Json& ts_ = key(j, "tensors", top);
  require(ts_.is_array() && !ts_.arr.empty(), SS_E_DESC, "layout: tensors");
  for (size_t i = 0; i < ts_.arr.size(); ++i) {
    const Json& t = ts_.arr[i];
    const std::string w = "tensors[" + std::to_string(i) + "]";
    require(t.is_object(), SS_E_DESC, w + ": expected an object");
    for (const auto& kv : t.obj)
      require(kv.first.rfind("x-", 0) == 0 || kv.first == "layer" || kv.first == "name" || kv.first == "format" ||
                  kv.first == "source" || kv.first == "offset" || kv.first == "bytes" || kv.first == "units" ||
                  kv.first == "first_unit" || kv.first == "stride" || kv.first == "rows" || kv.first == "row_bytes" ||
                  kv.first == "segments" || kv.first == "gather" || kv.first == "layout",
              SS_E_DESC, w + ": unknown key " + kv.first);
    Tensor x;
    {
      const int64_t layer = json_i64(key(t, "layer", w), w + ".layer");
      require(layer >= -1 && layer <= INT32_MAX, SS_E_DESC, w + ".layer: out of range");
      x.layer = int32_t(layer);
    }
    x.name = json_str(key(t, "name", w), w + ".name");
    x.format = json_str(key(t, "format", w), w + ".format");
    x.source = u32_of(key(t, "source", w), w + ".source");
    x.offset = json_u64(key(t, "offset", w), w + ".offset");
    x.bytes = json_u64(key(t, "bytes", w), w + ".bytes");
    if (const Json* v = t.find("units")) x.units = u32_of(*v, w + ".units");
    if (const Json* v = t.find("first_unit")) x.first_unit = u32_of(*v, w + ".first_unit");
    if (const Json* v = t.find("stride")) x.stride = json_u64(*v, w + ".stride");
    if (const Json* v = t.find("rows")) {
      x.rows = u32_of(*v, w + ".rows");
      x.rows_given = true;
    }
    if (const Json* v = t.find("row_bytes")) x.row_bytes = json_u64(*v, w + ".row_bytes");
    if (const Json* v = t.find("layout")) {
      x.layout = json_str(*v, w + ".layout");
      require(!x.layout.empty() && x.layout != "none" && x.layout.find(',') == std::string::npos, SS_E_DESC,
              w + ".layout: a name without commas");
    }
    if (const Json* v = t.find("segments")) {
      require(v->is_array(), SS_E_DESC, w + ".segments: expected boundaries");
      for (const Json& b : v->arr) x.seg_rows.push_back(u32_of(b, w + ".segments"));
    }
    if (const Json* g = t.find("gather")) {
      x.gather = true;
      const std::string wg = w + ".gather";
      require(g->is_object(), SS_E_DESC, wg + ": expected an object");
      for (const auto& kv : g->obj)
        require(kv.first == "row_bytes" || kv.first == "rows" || kv.first == "pieces", SS_E_DESC,
                wg + ": unknown key " + kv.first);
      x.g.src_row_bytes = json_u64(key(*g, "row_bytes", wg), w + ".gather.row_bytes");
      for (const Json& r : key(*g, "rows", wg).arr) {
        require(r.is_array() && r.arr.size() == 2, SS_E_DESC, w + ".gather.rows: [first, end) pairs");
        x.g.rows.push_back({u32_of(r.arr[0], wg + ".rows"), u32_of(r.arr[1], wg + ".rows")});
      }
      for (const Json& p : key(*g, "pieces", wg).arr) {
        require(p.is_array() && p.arr.size() == 2, SS_E_DESC, w + ".gather.pieces: [offset, bytes] pairs");
        x.g.pieces.push_back({json_u64(p.arr[0], w), json_u64(p.arr[1], w)});
      }
      require(x.g.src_row_bytes > 0 && !x.g.rows.empty() && !x.g.pieces.empty(), SS_E_DESC, w + ".gather: empty");
      uint64_t out_rows = 0, out_row_bytes = 0;
      for (const auto& r : x.g.rows) {
        require(r.second > r.first && mul_checked(r.second, x.g.src_row_bytes, wg + ".rows") <= x.bytes, SS_E_DESC,
                wg + ".rows: outside the source unit");
        out_rows += r.second - r.first;
      }
      for (const auto& p : x.g.pieces) {
        require(p.second > 0 && p.first < x.g.src_row_bytes && p.second <= x.g.src_row_bytes - p.first, SS_E_DESC,
                wg + ".pieces: outside the source row");
        out_row_bytes = add_checked(out_row_bytes, p.second, wg + ".pieces");
      }
      require(out_rows <= INT32_MAX, SS_E_DESC, wg + ".rows: too many rows");
      mul_checked(out_rows, out_row_bytes, wg);
      if (!x.rows_given) {
        x.rows = x.g.out_rows();
        x.row_bytes = x.g.out_row_bytes();
      }
    }
    require(x.source < L.sources.size(), SS_E_DESC, w + ".source: no such source");
    require(L.formats.count(x.format) == 1, SS_E_DESC, w + ".format: not in formats");
    require(x.units == 0 || x.stride >= x.bytes || x.units == 1, SS_E_DESC, w + ".stride: units overlap");
    require(x.bytes > 0, SS_E_DESC, w + ".bytes: empty");
    require(uint64_t(x.first_unit) + x.units <= uint64_t(INT32_MAX) + 1, SS_E_DESC, w + ".units: out of range");
    if (x.units) add_checked(add_checked(x.offset, mul_checked(x.units - 1, x.stride, w + ".stride"), w + ".offset"),
                             x.bytes, w + ".bytes");
    else add_checked(x.offset, x.bytes, w + ".bytes");
    L.tensors.push_back(x);
  }
  return L;
}

// A record of the build plan, in unit-major order.
struct Item {
  const Tensor* t;
  int32_t expert;   // -1 = dense
  uint64_t src_off;
};

std::vector<Item> plan_items(const Layout& L) {
  std::map<int32_t, std::vector<const Tensor*>> by_layer;
  for (const Tensor& t : L.tensors) by_layer[t.layer].push_back(&t);
  std::vector<Item> items;
  for (const auto& kv : by_layer) {
    std::set<uint32_t> experts;
    for (const Tensor* t : kv.second) {
      if (t->units == 0) items.push_back({t, -1, t->offset});
      for (uint32_t u = 0; u < t->units; ++u) experts.insert(t->first_unit + u);
    }
    for (uint32_t e : experts)
      for (const Tensor* t : kv.second)
        if (t->units && e >= t->first_unit && e < t->first_unit + t->units)
          items.push_back({t, int32_t(e), t->offset + uint64_t(e - t->first_unit) * t->stride});
  }
  return items;
}

uint64_t pad4k(uint64_t v) { return (v + kPackAlign - 1) / kPackAlign * kPackAlign; }

// Covering aligned read of [off, off + n) from a source into an aligned buffer; returns the pointer to off.
struct SrcBuf {
  std::vector<char> raw;
  char* base = nullptr;
  uint64_t cap = 0;
  const char* read(const RawFile& f, uint64_t off, uint64_t n) {
    const uint64_t al = f.direct() ? kPackAlign : 1;
    const uint64_t a0 = off / al * al, a1 = (off + n + al - 1) / al * al;
    if (a1 - a0 + kPackAlign > cap) {
      cap = a1 - a0 + kPackAlign;
      raw.assign(size_t(cap + kPackAlign), 0);
      base = reinterpret_cast<char*>((uintptr_t(raw.data()) + kPackAlign - 1) / kPackAlign * kPackAlign);
    }
    const uint64_t want = std::min(a1, f.size()) - a0;
    const uint64_t got = f.read(base, f.direct() ? a1 - a0 : want, a0);
    require(got >= off + n - a0, SS_E_IO, "short read of " + f.path());
    return base + (off - a0);
  }
};

void produce(const Tensor& t, const char* src, std::vector<char>* out) {
  out->resize(size_t(t.out_bytes()));
  if (!t.gather) {
    std::memcpy(out->data(), src, size_t(t.bytes));
    return;
  }
  char* o = out->data();
  for (const auto& r : t.g.rows)
    for (uint32_t row = r.first; row < r.second; ++row) {
      const char* s = src + uint64_t(row) * t.g.src_row_bytes;
      for (const auto& p : t.g.pieces) {
        std::memcpy(o, s + p.first, size_t(p.second));
        o += p.second;
      }
    }
}

PackRecordSpec spec_of(const Item& it) {
  PackRecordSpec s;
  s.layer = it.t->layer;
  s.expert = it.expert;
  s.name = it.t->name;
  s.format = it.t->format;
  s.bytes = it.t->out_bytes();
  s.rows = it.t->rows;
  s.row_bytes = it.t->rows ? it.t->row_bytes : 0;
  s.seg_rows = it.t->seg_rows;
  return s;
}

int cmd_build(int argc, char** argv) {
  require(argc >= 4, SS_E_INVALID, "usage: ts-pack build LAYOUT.json OUT_DIR [--dry-run] [--split-gib N] "
                                   "[--trust-source-sha256] [--layouts A,B|none]");
  bool dry = false, trust = false, pick = false;
  std::set<std::string> picked;
  double split_gib = -1;
  for (int i = 4; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--dry-run") dry = true;
    else if (a == "--trust-source-sha256") trust = true;
    else if (a == "--split-gib" && i + 1 < argc) split_gib = std::stod(argv[++i]);
    else if (a == "--layouts" && i + 1 < argc) {
      pick = true;
      std::stringstream ls(argv[++i]);
      for (std::string l; std::getline(ls, l, ',');)
        if (!l.empty() && l != "none") picked.insert(l);
    }
    else fail(SS_E_INVALID, "unknown option " + a);
  }
  const std::string layout_text = read_text(argv[2]);
  Layout L = parse_layout(layout_text);
  if (split_gib >= 0) L.split_bytes = uint64_t(split_gib * double(uint64_t(1) << 30));
  // optional layouts: tensors tagged with a layout are built only when it is picked (default: every tagged layout)
  std::set<std::string> have;
  for (const Tensor& t : L.tensors)
    if (!t.layout.empty()) have.insert(t.layout);
  if (!pick) picked = have;
  for (const std::string& l : picked) require(have.count(l) == 1, SS_E_INVALID, "--layouts: the layout has no " + l);
  L.tensors.erase(std::remove_if(L.tensors.begin(), L.tensors.end(),
                                 [&](const Tensor& t) { return !t.layout.empty() && picked.count(t.layout) == 0; }),
                  L.tensors.end());
  const std::vector<Item> items = plan_items(L);
  // the sizes, before anything is read or written
  std::map<std::string, std::pair<uint64_t, uint64_t>> per;
  uint64_t payload = 0, data = 0, file_bytes = 0, files = 1;
  int32_t last_layer = INT32_MIN;
  for (const Item& it : items) {
    if (it.t->layer != last_layer) {
      if (L.split_bytes && file_bytes >= L.split_bytes) {
        ++files;
        file_bytes = 0;
      }
      last_layer = it.t->layer;
    }
    const uint64_t b = it.t->out_bytes();
    payload += b;
    data += pad4k(b);
    file_bytes += pad4k(b);
    auto& e = per[it.t->format];
    ++e.first;
    e.second += b;
  }
  Json sz = Json::object();
  sz["records"] = Json::number(uint64_t(items.size()));
  sz["payload_bytes"] = Json::number(payload);
  sz["data_bytes"] = Json::number(data);
  sz["padding_bytes"] = Json::number(data - payload);
  sz["data_gib"] = Json::number(double(data) / double(uint64_t(1) << 30));
  sz["data_files"] = Json::number(files);
  Json lays = Json::array();
  for (const std::string& l : picked) lays.push(Json::string(l));
  sz["layouts"] = lays;
  Json fm = Json::object();
  for (const auto& kv : per) {
    Json e = Json::object();
    e["records"] = Json::number(kv.second.first);
    e["bytes"] = Json::number(kv.second.second);
    e["gib"] = Json::number(double(kv.second.second) / double(uint64_t(1) << 30));
    fm[kv.first] = e;
  }
  sz["formats"] = fm;
  std::printf("%s\n", json_write(sz).c_str());
  if (dry) return 0;

  // sources: bound opens, held for the build; identity before and after; SHA-256 (cached by identity) unless trusted
  std::vector<std::unique_ptr<RawFile>> src;
  std::vector<FileIdentity> ids;
  PackWriterOptions o;
  o.model = L.model;
  o.formats = L.formats;
  o.split_bytes = L.split_bytes;
  o.layouts.assign(picked.begin(), picked.end());
  o.meta = L.meta;
  for (const LayoutSource& s : L.sources) {
    src.push_back(std::make_unique<RawFile>(s.path, true, true));
    ids.push_back(src.back()->identity());
    PackSourceEntry e;
    e.name = s.name;
    e.size = src.back()->size();
    if (trust && !s.sha256.empty()) {
      e.sha256 = s.sha256;
    } else {
      std::fprintf(stderr, "ts-pack: SHA-256 of %s (cached by file identity)\n", s.path.c_str());
      e.sha256 = file_sha256_cached(*src.back());
      require(s.sha256.empty() || s.sha256 == e.sha256, SS_E_DESC,
              "source " + s.name + ": SHA-256 " + e.sha256 + " differs from the layout's " + s.sha256);
    }
    o.sources.push_back(e);
  }
  for (const Item& it : items)
    require(it.src_off + it.t->bytes <= src[it.t->source]->size(), SS_E_DESC,
            "tensor " + it.t->name + ": source range past the end of " + L.sources[it.t->source].path);
  o.receipt = L.receipt.is_object() ? L.receipt : Json::object();
  o.receipt["tool"] = Json::string("ts-pack build");
  o.receipt["layout_sha256"] = Json::string(sha256_hex(layout_text.data(), layout_text.size()));
  PackWriter w(argv[3], o);
  const auto t0 = std::chrono::steady_clock::now();
  // the next record's source read overlaps this one's transform and write
  struct Slot {
    SrcBuf buf;
    const char* p = nullptr;
  };
  Slot slots[2];
  auto load = [&](size_t i) {
    Slot& s = slots[i % 2];
    s.p = s.buf.read(*src[items[i].t->source], items[i].src_off, items[i].t->bytes);
  };
  std::future<void> next = std::async(std::launch::async, load, 0);
  std::vector<char> out;
  for (size_t i = 0; i < items.size(); ++i) {
    next.get();
    if (i + 1 < items.size()) next = std::async(std::launch::async, load, i + 1);  // the other slot
    produce(*items[i].t, slots[i % 2].p, &out);
    w.add(spec_of(items[i]), out.data());
  }
  for (size_t i = 0; i < src.size(); ++i)
    require(identity_of(L.sources[i].path) == ids[i], SS_E_IO, "source " + L.sources[i].path + " changed during the build");
  const std::string pin = w.finish();
  const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("{\"pack\":\"%s\",\"manifest_sha256\":\"%s\",\"seconds\":%.1f}\n", argv[3], pin.c_str(), s);
  return 0;
}

PackOptions opts_from(int argc, char** argv, int from) {
  PackOptions o;
  for (int i = from; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--pin" && i + 1 < argc) o.pin_sha256 = argv[++i];
    else if (a == "--weight-dirs" && i + 1 < argc) o.weight_dirs = argv[++i];
    else if (a == "--buffered") o.direct = false;
    else if (a == "--layout" && i + 1 < argc) o.require_layouts.push_back(argv[++i]);
    else if (a == "--verify" && i + 1 < argc) {
      const std::string m = argv[++i];
      o.verify = m == "always" ? PackVerify::Always : m == "off" ? PackVerify::Off : PackVerify::Cached;
    }
  }
  return o;
}

int cmd_verify(int argc, char** argv) {
  require(argc >= 3, SS_E_INVALID, "usage: ts-pack verify DIR [--pin SHA256] [--full]");
  bool full = false;
  for (int i = 3; i < argc; ++i) full = full || std::string(argv[i]) == "--full";
  auto p = Pack::open(argv[2], opts_from(argc, argv, 3));
  if (!full) {
    std::printf("{\"ok\":true,\"manifest_sha256\":\"%s\",\"checked\":\"manifest+index\"}\n", p->manifest_sha256().c_str());
    return 0;
  }
  const Json r = p->verify_full();
  std::printf("%s\n", json_write(r).c_str());
  return r.find("ok")->b ? 0 : 3;
}

int cmd_show(int argc, char** argv) {
  require(argc >= 3, SS_E_INVALID, "usage: ts-pack show DIR [--records]");
  auto p = Pack::open(argv[2], opts_from(argc, argv, 3));
  std::printf("%s\n", json_write(p->info()).c_str());
  for (int i = 3; i < argc; ++i)
    if (std::string(argv[i]) == "--records")
      for (const PackRecord& r : p->index().records)
        std::printf("%d %d %s %s file %u offset %llu bytes %llu rows %u row_bytes %llu segments %u xxh64 %016llx\n",
                    r.layer, r.expert, p->name(r).c_str(), p->format(r).c_str(), unsigned(r.file),
                    (unsigned long long)r.offset, (unsigned long long)r.bytes, r.rows,
                    (unsigned long long)r.row_bytes, r.seg_count, (unsigned long long)r.xxh64);
  return 0;
}

int cmd_cat(int argc, char** argv) {
  require(argc >= 5, SS_E_INVALID, "usage: ts-pack cat DIR LIST OUT [--verify cached|always|off]");
  auto p = Pack::open(argv[2], opts_from(argc, argv, 5));
  std::istringstream list(read_text(argv[3]));
  struct Want {
    uint64_t rec;
    uint32_t r0, r1;
    uint64_t bytes;
  };
  std::vector<Want> wants;
  for (std::string line; std::getline(list, line);) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    int32_t layer, expert;
    std::string name;
    uint32_t r0 = 0, r1 = 0;
    require(bool(ls >> layer >> expert >> name), SS_E_INVALID, "cat: bad line: " + line);
    ls >> r0 >> r1;
    const int64_t i = p->find(layer, expert, name);
    require(i >= 0, SS_E_INVALID, "cat: no record for: " + line);
    uint64_t off, bytes;
    p->row_span(p->index().records[size_t(i)], r0, r1, &off, &bytes);
    wants.push_back({uint64_t(i), r0, r1, bytes});
  }
  std::FILE* f = std::fopen(argv[4], "wb");
  require(f != nullptr, SS_E_IO, std::string("cannot create ") + argv[4]);
  constexpr size_t kBatch = 64;
  std::vector<std::vector<char>> bufs(kBatch);  // 4 KiB aligned destinations: whole records land in place
  uint64_t total = 0;
  for (size_t at = 0; at < wants.size(); at += kBatch) {
    const size_t n = std::min(kBatch, wants.size() - at);
    std::vector<PackRead> rs;
    std::vector<char*> dst(n);
    for (size_t k = 0; k < n; ++k) {
      bufs[k].resize(size_t(pad4k(wants[at + k].bytes) + kPackAlign));
      dst[k] = reinterpret_cast<char*>((uintptr_t(bufs[k].data()) + kPackAlign - 1) / kPackAlign * kPackAlign);
      rs.push_back({wants[at + k].rec, wants[at + k].r0, wants[at + k].r1, dst[k]});
    }
    p->readv(rs.data(), rs.size());
    for (size_t k = 0; k < n; ++k) {
      const size_t b = size_t(wants[at + k].bytes);
      require(std::fwrite(dst[k], 1, b, f) == b, SS_E_IO, "write failed");
      total += b;
    }
  }
  require(std::fclose(f) == 0, SS_E_IO, "close failed");
  const PackStats s = p->stats();
  std::printf("{\"reads\":%llu,\"bytes\":%llu,\"bounce_bytes\":%llu,\"verified_records\":%llu,\"verified_segments\":%llu}\n",
              (unsigned long long)s.reads, (unsigned long long)total, (unsigned long long)s.bounce_bytes,
              (unsigned long long)s.verified_records, (unsigned long long)s.verified_segments);
  return 0;
}

int cmd_desc(int argc, char** argv) {
  require(argc >= 5, SS_E_INVALID, "usage: ts-pack desc DIR BASE_DESC.json OUT.json");
  auto p = Pack::open(argv[2], opts_from(argc, argv, 5));
  const std::string out = json_write(p->dg_descriptor(json_parse(read_text(argv[3]))));
  std::ofstream f(argv[4], std::ios::binary);
  f << out;
  require(bool(f), SS_E_IO, std::string("cannot write ") + argv[4]);
  return 0;
}

}  // namespace

int pack_main(int argc, char** argv) {
  try {
    const std::string c = argc > 1 ? argv[1] : "";
    if (c == "build") return cmd_build(argc, argv);
    if (c == "verify") return cmd_verify(argc, argv);
    if (c == "show") return cmd_show(argc, argv);
    if (c == "cat") return cmd_cat(argc, argv);
    if (c == "desc") return cmd_desc(argc, argv);
    std::fprintf(stderr, "usage: ts-pack build|verify|show|cat|desc ... (docs/pack.md)\n");
    return 2;
  } catch (const Error& e) {
    std::fprintf(stderr, "ts-pack: %s\n", e.what());
    return 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ts-pack: %s\n", e.what());
    return 1;
  }
}

#if !defined(TS_PACK_TESTING)
int main(int argc, char** argv) { return pack_main(argc, argv); }
#endif
