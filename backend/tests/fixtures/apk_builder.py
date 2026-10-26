"""TEST-ONLY builder for tiny, benign, synthetic APK files.

Purpose: let unit tests exercise the real Androguard-based extractor without the
Android SDK and without handling any malware. The APKs produced here:

* contain a hand-encoded binary AndroidManifest.xml (AXML) and a hand-encoded
  classes.dex that declare only the names the test asks for;
* contain no executable code (class definitions have no code items);
* are NOT signed and would not install on a device;
* must never be used as training data or as evidence of detection quality.
"""

from __future__ import annotations

import hashlib
import struct
import zipfile
import zlib
from pathlib import Path
from typing import Iterable, Sequence

ANDROID_NS = "http://schemas.android.com/apk/res/android"
NO_INDEX = 0xFFFFFFFF
DEFAULT_PACKAGE = "com.example.guardian.testapp"
MAIN_ACTIVITY = DEFAULT_PACKAGE + ".MainActivity"

# (class descriptor, method name, return descriptor, parameter descriptors)
MethodRef = tuple[str, str, str, Sequence[str]]


# --------------------------------------------------------------------------
# Binary XML (AXML)
# --------------------------------------------------------------------------
def _axml_string_pool(strings: Sequence[str]) -> bytes:
    data = b""
    offsets = []
    for s in strings:
        if len(s) >= 0x7FFF:
            raise ValueError("string too long for this minimal encoder")
        offsets.append(len(data))
        data += struct.pack("<H", len(s)) + s.encode("utf-16-le") + b"\x00\x00"
    while len(data) % 4:
        data += b"\x00"
    header_size = 28
    strings_start = header_size + 4 * len(strings)
    size = strings_start + len(data)
    out = struct.pack("<HHIIIIII", 0x0001, header_size, size, len(strings), 0, 0, strings_start, 0)
    out += b"".join(struct.pack("<I", o) for o in offsets)
    return out + data


def build_axml_manifest(
    package: str,
    permissions: Iterable[str],
    activities: Iterable[str],
    services: Iterable[str],
    receivers: Iterable[str],
    activity_aliases: Iterable[str] = (),
) -> bytes:
    permissions, activities = list(permissions), list(activities)
    services, receivers = list(services), list(receivers)
    activity_aliases = list(activity_aliases)

    pool: list[str] = []
    index: dict[str, int] = {}

    def sid(s: str) -> int:
        if s not in index:
            index[s] = len(pool)
            pool.append(s)
        return index[s]

    sid("name")  # index 0 must line up with the resource map below
    ns_prefix, ns_uri = sid("android"), sid(ANDROID_NS)
    for s in ("manifest", "package", package, "uses-permission", "application",
              "activity", "service", "receiver", "activity-alias"):
        sid(s)
    for s in [*permissions, *activities, *services, *receivers, *activity_aliases]:
        sid(s)

    chunks: list[bytes] = []
    line = [1]

    def next_line() -> int:
        line[0] += 1
        return line[0]

    def start_ns() -> bytes:
        return struct.pack("<HHIIIII", 0x0100, 16, 24, next_line(), NO_INDEX, ns_prefix, ns_uri)

    def end_ns() -> bytes:
        return struct.pack("<HHIIIII", 0x0101, 16, 24, next_line(), NO_INDEX, ns_prefix, ns_uri)

    def start_el(name: str, attrs: list[tuple[int, str, str]]) -> bytes:
        size = 36 + 20 * len(attrs)
        out = struct.pack("<HHIIIIIHHHHHH", 0x0102, 16, size, next_line(), NO_INDEX,
                          NO_INDEX, sid(name), 0x14, 0x14, len(attrs), 0, 0, 0)
        for ns, attr_name, value in attrs:
            v = sid(value)
            out += struct.pack("<IIIHBBI", ns, sid(attr_name), v, 8, 0, 0x03, v)
        return out

    def end_el(name: str) -> bytes:
        return struct.pack("<HHIIIII", 0x0103, 16, 24, next_line(), NO_INDEX, NO_INDEX, sid(name))

    chunks.append(start_ns())
    chunks.append(start_el("manifest", [(NO_INDEX, "package", package)]))
    for perm in permissions:
        chunks.append(start_el("uses-permission", [(ns_uri, "name", perm)]))
        chunks.append(end_el("uses-permission"))
    chunks.append(start_el("application", []))
    for tag, names in (("activity", activities), ("activity-alias", activity_aliases),
                       ("service", services), ("receiver", receivers)):
        for n in names:
            chunks.append(start_el(tag, [(ns_uri, "name", n)]))
            chunks.append(end_el(tag))
    chunks.append(end_el("application"))
    chunks.append(end_el("manifest"))
    chunks.append(end_ns())

    pool_chunk = _axml_string_pool(pool)
    resmap = struct.pack("<HHII", 0x0180, 8, 12, 0x01010003)
    body = pool_chunk + resmap + b"".join(chunks)
    return struct.pack("<HHI", 0x0003, 8, 8 + len(body)) + body


# --------------------------------------------------------------------------
# DEX (version 035, no code items)
# --------------------------------------------------------------------------
def _uleb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _shorty_char(descriptor: str) -> str:
    return "L" if descriptor[0] in "L[" else descriptor[0]


def build_dex(
    defined_classes: Sequence[str],
    method_refs: Sequence[MethodRef] = (),
    extra_strings: Sequence[str] = (),
    superclass: str = "Landroid/app/Activity;",
) -> bytes:
    method_refs = list(dict.fromkeys((c, n, r, tuple(p)) for c, n, r, p in method_refs))

    types: set[str] = set(defined_classes)
    if defined_classes:
        types.add(superclass)
    strings: set[str] = set(extra_strings)
    protos: set[tuple[str, str, tuple[str, ...]]] = set()
    for cls, name, ret, params in method_refs:
        types.add(cls)
        types.add(ret)
        types.update(params)
        strings.add(name)
        protos.add((_shorty_char(ret) + "".join(_shorty_char(p) for p in params), ret, tuple(params)))
    for _, ret, params in protos:
        pass
    strings.update(types)
    strings.update(shorty for shorty, _, _ in protos)
    for s in strings:
        if not s.isascii() or not s:
            raise ValueError("this minimal encoder only supports non-empty ASCII strings")

    sorted_strings = sorted(strings, key=lambda s: s.encode("utf-16-be"))
    sidx = {s: i for i, s in enumerate(sorted_strings)}
    sorted_types = sorted(types, key=lambda t: sidx[t])
    tidx = {t: i for i, t in enumerate(sorted_types)}
    sorted_protos = sorted(protos, key=lambda p: (tidx[p[1]], tuple(tidx[x] for x in p[2])))
    pidx = {p: i for i, p in enumerate(sorted_protos)}

    method_ids = sorted(
        {
            (tidx[c], pidx[(_shorty_char(r) + "".join(_shorty_char(x) for x in p), r, tuple(p))], sidx[n])
            for c, n, r, p in method_refs
        }
    )

    n_str, n_type, n_proto = len(sorted_strings), len(sorted_types), len(sorted_protos)
    n_method, n_class = len(method_ids), len(defined_classes)

    off_string_ids = 0x70
    off_type_ids = off_string_ids + 4 * n_str
    off_proto_ids = off_type_ids + 4 * n_type
    off_method_ids = off_proto_ids + 12 * n_proto
    off_class_defs = off_method_ids + 8 * n_method
    off_data = off_class_defs + 32 * n_class

    data = bytearray()
    string_offsets = []
    for s in sorted_strings:
        string_offsets.append(off_data + len(data))
        data += _uleb128(len(s)) + s.encode("ascii") + b"\x00"
    string_data_end = len(data)

    while (off_data + len(data)) % 4:
        data += b"\x00"
    type_list_start = len(data)
    param_offsets: dict[tuple[str, ...], int] = {}
    for _, _, params in sorted_protos:
        if params and params not in param_offsets:
            param_offsets[params] = off_data + len(data)
            data += struct.pack("<I", len(params)) + b"".join(struct.pack("<H", tidx[p]) for p in params)
            while (off_data + len(data)) % 4:
                data += b"\x00"
    has_type_lists = len(data) > type_list_start

    off_map = off_data + len(data)
    map_entries = [
        (0x0000, 1, 0),
        (0x0001, n_str, off_string_ids),
        (0x0002, n_type, off_type_ids),
        (0x0003, n_proto, off_proto_ids),
    ]
    if n_method:
        map_entries.append((0x0005, n_method, off_method_ids))
    if n_class:
        map_entries.append((0x0006, n_class, off_class_defs))
    map_entries.append((0x2002, n_str, off_data))
    if has_type_lists:
        map_entries.append((0x1001, len(param_offsets), off_data + type_list_start))
    map_entries.append((0x1000, 1, off_map))
    data += struct.pack("<I", len(map_entries))
    for t, size, off in map_entries:
        data += struct.pack("<HHII", t, 0, size, off)

    body = bytearray()
    for off in string_offsets:
        body += struct.pack("<I", off)
    for t in sorted_types:
        body += struct.pack("<I", sidx[t])
    for shorty, ret, params in sorted_protos:
        body += struct.pack("<III", sidx[shorty], tidx[ret], param_offsets.get(params, 0))
    for c, p, n in method_ids:
        body += struct.pack("<HHI", c, p, n)
    for cls in defined_classes:
        body += struct.pack("<IIIIIIII", tidx[cls], 0x1, tidx[superclass], 0, NO_INDEX, 0, 0, 0)

    total = off_data + len(data)
    header = bytearray(0x70)
    header[0:8] = b"dex\n035\x00"
    fields = [
        total, 0x70, 0x12345678, 0, 0, off_map,
        n_str, off_string_ids if n_str else 0,
        n_type, off_type_ids if n_type else 0,
        n_proto, off_proto_ids if n_proto else 0,
        0, 0,
        n_method, off_method_ids if n_method else 0,
        n_class, off_class_defs if n_class else 0,
        total - off_data, off_data,
    ]
    struct.pack_into("<20I", header, 32, *fields)
    blob = bytes(header) + bytes(body) + bytes(data)
    assert len(blob) == total, (len(blob), total)
    blob_ba = bytearray(blob)
    blob_ba[12:32] = hashlib.sha1(bytes(blob_ba[32:])).digest()
    struct.pack_into("<I", blob_ba, 8, zlib.adler32(bytes(blob_ba[12:])) & 0xFFFFFFFF)
    return bytes(blob_ba)


# --------------------------------------------------------------------------
# APK (ZIP) assembly
# --------------------------------------------------------------------------
DEFAULT_METHOD_REFS: tuple[MethodRef, ...] = (
    ("Landroid/app/Activity;", "onCreate", "V", ("Landroid/os/Bundle;",)),
    ("Landroid/util/Log;", "d", "I", ("Ljava/lang/String;", "Ljava/lang/String;")),
    ("Lcom/example/guardian/testapp/MainActivity;", "helper", "V", ()),  # internal reference
)


def build_apk(
    path: str | Path,
    *,
    package: str = DEFAULT_PACKAGE,
    permissions: Sequence[str] = ("android.permission.INTERNET",),
    activities: Sequence[str] = (MAIN_ACTIVITY,),
    services: Sequence[str] = (),
    receivers: Sequence[str] = (),
    activity_aliases: Sequence[str] = (),
    method_refs: Sequence[MethodRef] = DEFAULT_METHOD_REFS,
    extra_strings: Sequence[str] = ("hello guardian",),
    defined_classes: Sequence[str] = ("L" + MAIN_ACTIVITY.replace(".", "/") + ";",),
    include_manifest: bool = True,
    include_dex: bool = True,
    extra_dex_files: Sequence[bytes] = (),
) -> Path:
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        if include_manifest:
            zf.writestr(
                "AndroidManifest.xml",
                build_axml_manifest(package, permissions, activities, services, receivers, activity_aliases),
            )
        if include_dex:
            zf.writestr("classes.dex", build_dex(defined_classes, method_refs, extra_strings))
        for i, blob in enumerate(extra_dex_files, start=2):
            zf.writestr(f"classes{i}.dex", blob)
    return path
