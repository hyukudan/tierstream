# Indexed prepared pack `tierstream/pack-v1`

A pack holds a model's weights **in the engine's final device format**: the exact bytes each engine's loader produces
before its host-to-device copy, written once and then read straight into place. One pack per model (and quantization
and device-format version) serves every layout: an expert-parallel rank reads whole experts of its group, a
tensor-parallel rank reads its row range of each expert (or its own per-rank record), the device-gated cache reads
units by key. tierstream never produces device bytes: the engine's repacker writes them through the writer API, or
`ts-pack build` copies them from the model file following a layout the engine provides. An engine that finds no
matching pack, or is told not to use one, keeps its original loader.

## 1. Files

A pack is a directory:

| file | content |
|---|---|
| `pack.json` | manifest (canonical JSON, below). Operators pin its SHA-256; it pins everything else. |
| `pack.index` | binary index: one record per tensor / expert part (§2) |
| `pack-0000.data`, ... | the records back to back, each 4 KiB aligned, zero padding; sizes are multiples of 4 KiB |

`pack.json` (unknown keys are errors except `x-*`):

| field | meaning |
|---|---|
| `schema` | `"tierstream/pack-v1"` |
| `model` | name, for messages and receipts |
| `alignment` | 4096 |
| `sources` | `[{name, size, sha256}]`: the model files the records were produced from. A pack whose sources differ from the files the app authenticated is refused (`ss_pack_options.source_sha256`). |
| `formats` | `{tag: {transform, version, tool}}`: how the bytes of each format tag were produced (every record's format is listed) |
| `files` | `[{name, size, sha256}]`, named `pack-NNNN.data` in order |
| `index` | `{name: "pack.index", size, sha256, records, segments}` |
| `layouts` | the optional layouts the build included (sorted names, e.g. `tp2-packed`); a consumer that needs one the pack lacks is refused by name (`ss_pack_options.layouts`) |
| `meta`, `receipt` | engine notes and the creation receipt (objects) |

## 2. Records

A record is one tensor or one part of an expert, whole and contiguous: key `(layer, expert, name)` (layer -1 = global
tensors, expert -1 = a layer's dense tensors), its format tag, data file, offset, bytes and xxh64.

- **Rows.** A record may declare `rows` and `row_bytes` (uniform rows: any `[row0, row1)` is one contiguous read) or,
  for non-uniform rows, a segment table (read by whole segments). `rows = 0`: not row-addressable.
- **Segments** `{row0, row1, offset, bytes, xxh64}` tile a record's rows and bytes in order. They carry their own
  xxh64, so a rank's row range that is a union of segments is verified without reading the rest of the record. A
  writer declares the ranks' boundaries as segments (DeepSeek expert TP: 0 / 1152 / 1280 / 2304).
- **Per-rank records.** A format that is not row-separable (DeepSeek expert-TP down rows split inside a Q2_K block)
  stores one record per rank, named `<tensor>@<rank>/<ranks>` (variants with a suffix, e.g. `+dup`), its own format tag.
- **Order (unit-major).** Layers ascending (-1 first); inside a layer its dense records, then its units; the records of
  one unit `(layer, expert >= 0)` adjacent, so a unit is one contiguous span (its records and the zero padding between
  them) and the units of a layer with equal record sizes are evenly spaced. A layer never spans two data files.

`pack.index` (little endian, frozen for v1): `"TSPACKI1"`, u32 version 1, u32 record size 64, u32 segment size 32,
u32 string count, u64 records, u64 segments, u64 string-table bytes, 16 zero bytes; then the records
`{i32 layer, i32 expert, u32 name, u32 format, u16 file, u16 kind, u32 rows, u64 row_bytes, u64 offset, u64 bytes,
u64 xxh64, u32 seg_first, u32 seg_count}`, the segments `{u32 row0, u32 row1, u64 offset, u64 bytes, u64 xxh64}`, and
the strings (u32 length + bytes). `kind` 0 = stored bytes, the only kind of v1; other kinds are reserved for later
versions (e.g. records derived from another record on read) and a v1 reader refuses them by name. Sizes are checked before allocation (at most 2^24 records and segments, 64 MiB of
strings, 4096 data files; the writer enforces the same limits and a 16 MiB `pack.json` before it publishes) and every
rule above is validated at open, with checked arithmetic (segment sizes, rows x row bytes), every string unique.

## 3. Integrity

- `pack.json` SHA-256 pinned by the operator (`ss_pack_options.manifest_sha256`, `ts-pack verify --pin`) ->
  `pack.index` SHA-256 and the data files' sizes and SHA-256 -> every record's and segment's xxh64.
- Every file is opened through held directory handles (bound opens, as every descriptor source); `pack.json` and
  `pack.index` are read through the handles that stay open (share-read) for the pack's lifetime.
- Record verification (`ss_pack_options.verify`): `cached` (default) checks a record or segment the first time it is
  read and keeps the verified set in the user's cache directory (`tierstream/pack-verified/<pack.json sha>.bits`) for
  as long as the data files keep their identities (volume, file index, size, write and creation times); `always`
  checks every read; `off` never (the operator ran `ts-pack verify --full`). A mismatch is `SS_E_HASH`. Cached results
  vouch only for the data files as they were opened: each batch of reads checks the files' live identities (through
  a held handle per file) before and after; a change (even a metadata-only one) drops every cached result for the
  rest of the pack's life (all reads are then hashed and succeed when the bytes are intact), fails with `SS_E_IO` a
  batch that returned bytes on the strength of a cached result while the change happened, and nothing is persisted
  for a changed file (Windows denies writers while the pack is open; Linux cannot). A data file served through stripe
  copies (`weight_dirs`) is hashed on every read in `cached` mode: its copies on other drives have no live identity
  check here. A range read
  inside a segment, of a record not yet verified, reads the whole record once and copies the range.
- xxh64 is an integrity check against accidental corruption (media errors, torn or misplaced writes), not tamper
  resistance: anyone who can write the data files can craft bytes that match a record's xxh64. Only `verify_full`
  (`ts-pack verify --full`: every data file's SHA-256 against the pinned manifest) resists crafted data; otherwise the
  trust anchor is the pinned manifest and the pack directory's owner.

## 4. Reading

`ss_pack_open(dir, options)`, then `ss_pack_find` / `ss_pack_entry_at` / `ss_pack_unit_find` and:

- `ss_pack_read(pack, record, row0, row1, dst)` - a whole record (`row1 = 0`) or a row range;
- `ss_pack_readv(pack, reads, n)` - all reads of one call in flight together per data file (overlapped reads on Windows,
  one `preadv` routed over the drives on a striped file); every `ss_pack_read_req` carries `struct_size`, one size per
  call;
- `ss_pack_read_unit(pack, layer, expert, dst)` - a unit's whole span.

Unbuffered reads land in place when the destination is aligned like the file offset; only an unaligned head or tail
block (or, for a destination aligned differently, the whole span) goes through a bounce buffer. The data files are
ordinary primaries for `ts-stripe`: stripe a pack over drives and pass `weight_dirs` (§4.4 of the design).

**Device-gated cache.** `ss_pack_dg_descriptor(pack, base_desc)` takes the engine's own desc-v1, appends the pack's data
files to its sources and points every part of each routed layer (part name = record name) into the pack: offset of unit
0, one stride (refused with `SS_E_UNSUPPORTED` when units are not evenly placed in ascending order, or when the base's
sources and the pack's data files together exceed desc-v1's 1024 sources; `SS_E_DESC` when a part's bytes differ from
its records). Lookup tables and other sources stay as they were. A routed layer whose base has no per-unit `hash` gets one from the pack (verify
`cached` / `always`): each unit's records are read once with the pack's verification and the xxh64 of its parts back to
back is listed, so the device-gated cache (`verify_hash`) checks every unit it lands against verified bytes; the lists
are cached per (pack, layer, parts, data-file identities), so this full read of the selected parts happens once
(DeepSeek: every routed expert, the size of the selected parts; again after the data files change identity, under
`always`, for striped packs, or without a cache directory). An engine that already has per-unit hashes of its bytes
passes them in the base descriptor and nothing is read.
`ss_pack_file_identity` gives the identities for `ss_options.source_identities`.

## 5. Writing

In-process (`ss_pack_writer_begin` / `record_begin` / `record_write` / `record_end` / `add` / `finish` / `abort`): records
in unit-major order, streamed with their xxh64 and the data files' SHA-256 computed on the fly. The pack is built in
`DIR.incomplete` (created exclusively; never over an existing `DIR`) and published by one rename after the index and the
manifest were written and every file flushed; abort or a failure removes it. On Linux the publish is
`renameat2(RENAME_NOREPLACE)` (a `DIR` created during the build is never replaced; other systems check then rename). A
refused record spec (unit-major order, geometry, an undeclared format) leaves the writer as it was; any other error
(a write, a data-file split, finish) fails the build at once: `DIR.incomplete` is removed and every later call except
`abort` returns an error, so a retry never continues a pack with missing or duplicated bytes. `split_bytes` starts a new
data file at the first layer boundary past that size. On Linux written pages are synced and dropped from the page cache as the build
goes (unified memory).

**`ts-pack build LAYOUT.json OUT_DIR`** builds from a layout (`tierstream/pack-layout-v1`) the engine provides:

```json
{"schema": "tierstream/pack-layout-v1", "model": "...",
 "sources": [{"name": "model", "path": "/models/x.gguf"}],
 "formats": {"Q2_K": {"transform": "copy", "version": "1", "tool": "..."}},
 "split_bytes": 0, "meta": {}, "receipt": {},
 "tensors": [
   {"layer": 3, "name": "blk.3.ffn_down_exps.weight", "format": "Q2_K", "source": 0,
    "offset": 123, "stride": 3870720, "bytes": 3870720, "units": 384, "rows": 5120, "row_bytes": 756},
   {"layer": 3, "name": "blk.3.ffn_down_exps.weight@1/2", "format": "x-tp2-down", "source": 0,
    "offset": 123, "stride": 3870720, "bytes": 3870720, "units": 384,
    "gather": {"row_bytes": 756, "rows": [[0, 5120]], "pieces": [[420, 336], [344, 8], [384, 32], [416, 4]]}}]}
```

Unit `u` of a tensor is `bytes` at `offset + u x stride` of its source (`units` 0 = one dense record). A tensor may
name an optional `layout` (e.g. the per-rank records of one tensor-parallel split): `--layouts A,B` builds only those
(`none` = none; default every layout of the file) and pack.json records which were built, so one layout file gives an
RTX-only pack of whole experts or a node pack with the per-rank records. Without `gather`
the record is those bytes; with it, the record is, for each source row of `rows` (ranges in output order), the listed
byte `pieces` of that row - row selections and byte rearrangements need no engine code. Any other transform runs in the
engine through the writer API. The build prints the sizes first (`--dry-run` stops there), hashes each source (SHA-256
cached by file identity; `--trust-source-sha256` takes the layout's), reads the next record while it writes the current
one, checks that no source changed, and prints the pin.

Other commands: `ts-pack verify DIR [--pin SHA] [--full]`, `ts-pack show DIR [--records]`,
`ts-pack cat DIR LIST OUT` (concatenates records / row ranges listed as `layer expert name [row0 row1]`),
`ts-pack desc DIR BASE_DESC.json OUT.json`.
