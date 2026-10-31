// C ABI of tierstream/pack-v1 (ss.h): every entry point catches, maps to ss_status and keeps the message per thread
// through the same guard as capi.cpp.
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "errors.hpp"
#include "pack.hpp"
#include "strix_stream/ss.h"

struct ss_pack {
  std::unique_ptr<ts::Pack> p;
};
struct ss_pack_writer {
  std::unique_ptr<ts::PackWriter> w;
};

namespace ts {
// capi.cpp
ss_status capi_guard(const std::function<void()>& f);
ss_status capi_give_text(const std::string& s, char** out, size_t* len);
}  // namespace ts

namespace {

void need(bool ok, const char* what) { ts::require(ok, SS_E_INVALID, what); }

template <class T>
void copy_sized(const T& full, T* out, uint32_t size) {
  need(out && size >= sizeof(uint32_t), "null or undersized output");
  std::memcpy(out, &full, std::min<size_t>(size, sizeof full));
  out->struct_size = uint32_t(std::min<size_t>(size, sizeof full));
}

ss_pack_entry entry_of(const ts::Pack& p, uint64_t i) {
  const ts::PackRecord& r = p.index().records[size_t(i)];
  ss_pack_entry e{};
  e.struct_size = sizeof e;
  e.layer = r.layer;
  e.expert = r.expert;
  e.file = r.file;
  e.index = i;
  e.name = p.name(r).c_str();
  e.format = p.format(r).c_str();
  e.offset = r.offset;
  e.bytes = r.bytes;
  e.xxh64 = r.xxh64;
  e.rows = r.rows;
  e.segments = r.seg_count;
  e.row_bytes = r.row_bytes;
  return e;
}

// The sizes of the structs as first released (pack-v1): smaller callers are refused, fields added later get defaults.
constexpr size_t kRecordSpecV1 = offsetof(ss_pack_record_spec, reserved0) + sizeof(uint32_t);
constexpr size_t kReadReqV1 = offsetof(ss_pack_read_req, dst) + sizeof(void*);

ts::PackRecordSpec spec_of(const ss_pack_record_spec* in) {
  need(in && in->struct_size >= kRecordSpecV1, "bad record spec size");
  ss_pack_record_spec full{};  // defaults of the fields a smaller (older) struct lacks
  std::memcpy(&full, in, std::min<size_t>(in->struct_size, sizeof full));
  const ss_pack_record_spec* r = &full;
  need(r->name && r->format, "bad record spec");
  ts::PackRecordSpec s;
  s.layer = r->layer;
  s.expert = r->expert;
  s.name = r->name;
  s.format = r->format;
  s.bytes = r->bytes;
  s.rows = r->rows;
  s.row_bytes = r->row_bytes;
  if (r->n_seg) {
    need(r->seg_rows != nullptr, "seg_rows is NULL");
    s.seg_rows.assign(r->seg_rows, r->seg_rows + r->n_seg + 1);
    if (r->seg_bytes) s.seg_bytes.assign(r->seg_bytes, r->seg_bytes + r->n_seg);
  }
  return s;
}

}  // namespace

extern "C" {

void ss_pack_options_defaults_sized(ss_pack_options* o, uint32_t size) {
  if (!o || size < sizeof(uint32_t)) return;
  ss_pack_options d{};
  d.struct_size = sizeof d;
  d.verify = SS_PACK_VERIFY_CACHED;
  d.drive_readers = 4;
  d.direct_io = 1;
  std::memcpy(o, &d, std::min<size_t>(size, sizeof d));
  o->struct_size = uint32_t(std::min<size_t>(size, sizeof d));
}

ss_status ss_pack_open(const char* dir, const ss_pack_options* opt, ss_pack** out) {
  return ts::capi_guard([&] {
    need(dir && out, "null argument");
    *out = nullptr;
    ss_pack_options o;
    ss_pack_options_defaults_sized(&o, sizeof o);
    if (opt) {
      need(opt->struct_size >= sizeof(uint32_t), "bad options size");
      std::memcpy(&o, opt, std::min<size_t>(opt->struct_size, sizeof o));
    }
    need(o.verify >= SS_PACK_VERIFY_CACHED && o.verify <= SS_PACK_VERIFY_OFF, "bad verify mode");
    need(o.drive_readers >= 1 && o.drive_readers <= 32, "drive_readers must be 1..32");
    ts::PackOptions po;
    po.verify = ts::PackVerify(o.verify);
    if (o.manifest_sha256) po.pin_sha256 = o.manifest_sha256;
    if (o.weight_dirs) po.weight_dirs = o.weight_dirs;
    po.drive_readers = o.drive_readers;
    po.direct = o.direct_io != 0;
    for (uint32_t i = 0; i < o.source_sha256_n; ++i) {
      need(o.source_sha256 && o.source_sha256[i], "source_sha256 entry is NULL");
      po.source_sha256.push_back(o.source_sha256[i]);
    }
    for (uint32_t i = 0; i < o.layouts_n; ++i) {
      need(o.layouts && o.layouts[i], "layouts entry is NULL");
      po.require_layouts.push_back(o.layouts[i]);
    }
    auto h = std::make_unique<ss_pack>();
    h->p = ts::Pack::open(dir, po);
    *out = h.release();
  });
}

void ss_pack_close(ss_pack* p) { delete p; }

uint64_t ss_pack_records(const ss_pack* p) { return p ? p->p->index().records.size() : 0; }

ss_status ss_pack_entry_at_sized(const ss_pack* p, uint64_t index, ss_pack_entry* out, uint32_t size) {
  return ts::capi_guard([&] {
    need(p && index < p->p->index().records.size(), "bad pack or record index");
    copy_sized(entry_of(*p->p, index), out, size);
  });
}

ss_status ss_pack_find_sized(const ss_pack* p, int layer, int expert, const char* name, ss_pack_entry* out,
                             uint32_t size) {
  return ts::capi_guard([&] {
    need(p && name, "null argument");
    const int64_t i = p->p->find(layer, expert, name);
    ts::require(i >= 0, SS_E_INVALID, "pack: no record (" + std::to_string(layer) + ", " + std::to_string(expert) +
                                          ", " + name + ")");
    copy_sized(entry_of(*p->p, uint64_t(i)), out, size);
  });
}

ss_status ss_pack_segment_at(const ss_pack* p, uint64_t record, uint32_t k, ss_pack_segment* out) {
  return ts::capi_guard([&] {
    need(p && out && record < p->p->index().records.size(), "bad argument");
    const ts::PackRecord& r = p->p->index().records[size_t(record)];
    need(k < r.seg_count, "no such segment");
    const ts::PackSegment& g = p->p->index().segments[r.seg_first + k];
    *out = {g.row0, g.row1, g.offset, g.bytes, g.xxh64};
  });
}

ss_status ss_pack_unit_find(const ss_pack* p, int layer, int expert, ss_pack_unit* out) {
  return ts::capi_guard([&] {
    need(p && out, "null argument");
    ts::PackUnit u;
    ts::require(p->p->unit(layer, expert, &u), SS_E_INVALID,
                "pack: no unit (" + std::to_string(layer) + ", " + std::to_string(expert) + ")");
    *out = {u.file, 0, u.offset, u.bytes, u.first, u.count};
  });
}

ss_status ss_pack_read(ss_pack* p, uint64_t record, uint32_t row0, uint32_t row1, void* dst) {
  return ts::capi_guard([&] {
    need(p && dst, "null argument");
    p->p->read(record, row0, row1, dst);
  });
}

ss_status ss_pack_readv(ss_pack* p, const ss_pack_read_req* reads, uint32_t n) {
  return ts::capi_guard([&] {
    need(p && (reads || n == 0), "null argument");
    std::vector<ts::PackRead> rs(n);
    // one element size per call (the first's): the array is walked with it, fields past the caller's size default
    const size_t stride = n ? reads[0].struct_size : 0;
    need(n == 0 || (stride >= kReadReqV1 && stride % alignof(ss_pack_read_req) == 0), "bad read request size");
    for (uint32_t i = 0; i < n; ++i) {
      ss_pack_read_req q{};
      const char* at = reinterpret_cast<const char*>(reads) + size_t(i) * stride;
      std::memcpy(&q, at, std::min<size_t>(stride, sizeof q));
      need(q.struct_size == stride, "read requests of one call must have one struct_size");
      rs[i] = {q.record, q.row0, q.row1, q.dst};
    }
    p->p->readv(rs.data(), rs.size());
  });
}

ss_status ss_pack_read_unit(ss_pack* p, int layer, int expert, void* dst) {
  return ts::capi_guard([&] {
    need(p && dst, "null argument");
    p->p->read_unit(layer, expert, dst);
  });
}

ss_status ss_pack_dg_descriptor(ss_pack* p, const char* base_json, size_t len, char** out_json,
                                size_t* out_len) {
  return ts::capi_guard([&] {
    need(p && base_json, "null argument");
    const ts::Json d = p->p->dg_descriptor(ts::json_parse(std::string(base_json, len)));
    ts::require(ts::capi_give_text(ts::json_write(d), out_json, out_len) == SS_OK, SS_E_NOMEM, "out of host memory");
  });
}

uint32_t ss_pack_files(const ss_pack* p) { return p ? uint32_t(p->p->manifest().files.size()) : 0; }

int ss_pack_has_layout(const ss_pack* p, const char* layout) {
  if (!p || !layout) return 0;
  const auto& l = p->p->manifest().layouts;
  return std::binary_search(l.begin(), l.end(), std::string(layout)) ? 1 : 0;
}

ss_status ss_pack_file_identity(const ss_pack* p, uint32_t file, ss_file_identity* out) {
  return ts::capi_guard([&] {
    need(p && out && file < p->p->manifest().files.size(), "bad argument");
    const ts::FileIdentity id = p->p->file_identity(file);
    *out = {id.volume, id.index, id.size};
  });
}

ss_status ss_pack_info_json(const ss_pack* p, char** out_json, size_t* out_len) {
  return ts::capi_guard([&] {
    need(p, "null pack");
    ts::require(ts::capi_give_text(ts::json_write(p->p->info()), out_json, out_len) == SS_OK, SS_E_NOMEM,
                "out of host memory");
  });
}

ss_status ss_pack_verify_full(ss_pack* p, char** out_json, size_t* out_len) {
  bool ok = false;
  const ss_status st = ts::capi_guard([&] {
    need(p, "null pack");
    const ts::Json rep = p->p->verify_full();
    ok = rep.find("ok")->b;
    if (out_json)
      ts::require(ts::capi_give_text(ts::json_write(rep), out_json, out_len) == SS_OK, SS_E_NOMEM, "out of host memory");
  });
  if (st != SS_OK) return st;
  return ts::capi_guard([&] { ts::require(ok, SS_E_HASH, "pack verification failed (see the report)"); });
}

ss_status ss_pack_writer_begin(const char* dir, const char* meta_json, size_t len, ss_pack_writer** out) {
  return ts::capi_guard([&] {
    need(dir && meta_json && out, "null argument");
    *out = nullptr;
    const ts::Json j = ts::json_parse(std::string(meta_json, len));
    need(j.is_object(), "meta_json: expected an object");
    for (const auto& kv : j.obj)
      ts::require(kv.first == "model" || kv.first == "sources" || kv.first == "formats" || kv.first == "split_bytes" ||
                      kv.first == "layouts" || kv.first == "meta" || kv.first == "receipt",
                  SS_E_INVALID, "meta_json: unknown key " + kv.first);
    ts::PackWriterOptions o;
    if (const ts::Json* v = j.find("model")) o.model = ts::json_str(*v, "model");
    if (const ts::Json* v = j.find("split_bytes")) o.split_bytes = ts::json_u64(*v, "split_bytes");
    if (const ts::Json* v = j.find("meta")) o.meta = *v;
    if (const ts::Json* v = j.find("layouts")) {
      need(v->is_array(), "layouts: expected an array");
      for (const ts::Json& l : v->arr) o.layouts.push_back(ts::json_str(l, "layouts"));
    }
    if (const ts::Json* v = j.find("receipt")) o.receipt = *v;
    if (const ts::Json* v = j.find("sources")) {
      need(v->is_array(), "sources: expected an array");
      for (const ts::Json& s : v->arr) {
        need(s.is_object() && s.find("name") && s.find("size") && s.find("sha256"),
             "sources: every entry needs name, size and sha256");
        ts::PackSourceEntry e;
        e.name = ts::json_str(*s.find("name"), "sources.name");
        e.size = ts::json_u64(*s.find("size"), "sources.size");
        e.sha256 = ts::json_str(*s.find("sha256"), "sources.sha256");
        o.sources.push_back(e);
      }
    }
    if (const ts::Json* v = j.find("formats")) {
      need(v->is_object(), "formats: expected an object");
      for (const auto& kv : v->obj) {
        need(kv.second.is_object(), "formats: every entry must be an object");
        ts::PackFormatEntry e;
        const ts::Json* t = kv.second.find("transform");
        const ts::Json* ver = kv.second.find("version");
        const ts::Json* tool = kv.second.find("tool");
        need(t && ver && tool, "formats: transform, version and tool are required");
        e.transform = ts::json_str(*t, "transform");
        e.version = ts::json_str(*ver, "version");
        e.tool = ts::json_str(*tool, "tool");
        o.formats[kv.first] = e;
      }
    }
    auto h = std::make_unique<ss_pack_writer>();
    h->w = std::make_unique<ts::PackWriter>(dir, std::move(o));
    *out = h.release();
  });
}

ss_status ss_pack_writer_record_begin(ss_pack_writer* w, const ss_pack_record_spec* r) {
  return ts::capi_guard([&] {
    need(w, "null writer");
    w->w->begin(spec_of(r));
  });
}

ss_status ss_pack_writer_record_write(ss_pack_writer* w, const void* data, uint64_t n) {
  return ts::capi_guard([&] {
    need(w && (data || n == 0), "null argument");
    w->w->write(data, n);
  });
}

ss_status ss_pack_writer_record_end(ss_pack_writer* w) {
  return ts::capi_guard([&] {
    need(w, "null writer");
    w->w->end();
  });
}

ss_status ss_pack_writer_add(ss_pack_writer* w, const ss_pack_record_spec* r, const void* data) {
  return ts::capi_guard([&] {
    need(w && data, "null argument");
    w->w->add(spec_of(r), data);
  });
}

ss_status ss_pack_writer_finish(ss_pack_writer* w, char* manifest_sha256) {
  std::unique_ptr<ss_pack_writer> own(w);
  return ts::capi_guard([&] {
    need(w, "null writer");
    const std::string pin = own->w->finish();
    if (manifest_sha256) std::memcpy(manifest_sha256, pin.c_str(), pin.size() + 1);
  });
}

void ss_pack_writer_abort(ss_pack_writer* w) { delete w; }

}  // extern "C"
