"""Read Mavic 3 Classic timed telemetry directly from its MP4 (no ML dependency).

Field paths/units follow the public DJI-family schema documented by ExifTool:
https://exiftool.org/TagNames/DJI.html . wm260l is checked explicitly; it is NOT
claimed to be an officially supported ExifTool protocol. This decoder validates
this recording's field layout, packet times, coordinate ranges and frame count.
"""
from __future__ import annotations
import datetime as dt
import struct
from pathlib import Path
import numpy as np


def boxes(data):
    pos = 0
    while pos < len(data):
        if len(data) - pos < 8:
            raise ValueError('Truncated MP4 box')
        size, kind = struct.unpack_from('>I4s', data, pos)
        header = 8
        if size == 1:
            size = struct.unpack_from('>Q', data, pos + 8)[0]
            header = 16
        if size == 0:
            size = len(data) - pos
        if size < header or pos + size > len(data):
            raise ValueError(f'Invalid MP4 box {kind!r}')
        yield kind, data[pos + header:pos + size]
        pos += size


def child(data, name):
    return next(value for key, value in boxes(data) if key == name)


def optional(data, name):
    return next((value for key, value in boxes(data) if key == name), None)


def read_moov(path):
    with Path(path).open('rb') as f:
        length = f.seek(0, 2)
        f.seek(0)
        while f.tell() < length:
            start = f.tell()
            size, kind = struct.unpack('>I4s', f.read(8))
            header = 8
            if size == 1:
                size = struct.unpack('>Q', f.read(8))[0]
                header = 16
            if size == 0:
                size = length - start
            if size < header or start + size > length:
                raise ValueError('Invalid top-level MP4 box')
            if kind == b'moov':
                return f.read(size - header)
            f.seek(start + size)
    raise ValueError('No moov box')


def track_info(trak):
    mdia = child(trak, b'mdia')
    mdhd = child(mdia, b'mdhd')
    timescale = struct.unpack_from('>I', mdhd, 20 if mdhd[0] else 12)[0]
    stbl = child(child(mdia, b'minf'), b'stbl')
    stsd = child(stbl, b'stsd')
    codec = stsd[12:16].decode('ascii')
    stsz = child(stbl, b'stsz')
    fixed, count = struct.unpack_from('>II', stsz, 4)
    sizes = [fixed] * count if fixed else list(struct.unpack_from(f'>{count}I', stsz, 12))
    stts = child(stbl, b'stts')
    nruns = struct.unpack_from('>I', stts, 4)[0]
    ticks = np.concatenate([np.full(n, d, dtype=np.int64) for n, d in
                            struct.iter_unpack('>II', stts[8:8 + 8 * nruns])])
    if len(ticks) != count or (ticks <= 0).any():
        raise ValueError('Invalid sample timing')
    pts = np.r_[0, np.cumsum(ticks[:-1])].astype(np.int64)
    ctts = optional(stbl, b'ctts')
    if ctts:
        offsets = np.concatenate([np.full(n, v, dtype=np.int64) for n, v in
                                  struct.iter_unpack('>Ii' if ctts[0] else '>II', ctts[8:])])
        if len(offsets) != count:
            raise ValueError('Composition timing count mismatch')
        pts += offsets
    edts = optional(trak, b'edts')
    if edts:
        elst = child(edts, b'elst')
        edits = struct.unpack_from('>I', elst, 4)[0]
        if edits != 1:
            raise ValueError('Multi-edit timeline unsupported; do not silently misalign')
        _, media_time, rate, fraction = struct.unpack_from('>Qqhh' if elst[0] else '>Iihh', elst, 8)
        if media_time < 0 or rate != 1 or fraction != 0:
            raise ValueError('Nontrivial edit timeline unsupported')
        pts -= media_time
    offsets_box = optional(stbl, b'co64')
    if offsets_box is not None:
        chunks = [v[0] for v in struct.iter_unpack('>Q', offsets_box[8:])]
    else:
        chunks = [v[0] for v in struct.iter_unpack('>I', child(stbl, b'stco')[8:])]
    sc = list(struct.iter_unpack('>III', child(stbl, b'stsc')[8:]))
    sample_offsets = []
    n, rule = 0, 0
    for chunk_number, offset in enumerate(chunks, 1):
        while rule + 1 < len(sc) and chunk_number >= sc[rule + 1][0]:
            rule += 1
        for _ in range(sc[rule][1]):
            sample_offsets.append(offset)
            offset += sizes[n]
            n += 1
    if n != count:
        raise ValueError('Chunk/sample count mismatch')
    return dict(codec=codec, pts=pts / timescale, durations=ticks / timescale,
                sizes=sizes, offsets=sample_offsets, timescale=timescale)


def varint(buf, pos):
    value = 0
    for shift in range(0, 70, 7):
        if pos >= len(buf):
            raise ValueError('Truncated varint')
        c = buf[pos]
        pos += 1
        value |= (c & 127) << shift
        if c < 128:
            return value, pos
    raise ValueError('Varint overflow')


def protobuf(buf):
    result, pos = {}, 0
    while pos < len(buf):
        key, pos = varint(buf, pos)
        field, wire = key >> 3, key & 7
        if not field or field in result:
            raise ValueError('Unexpected repeated/zero protobuf field')
        if wire == 0:
            value, pos = varint(buf, pos)
            if value >= 2**63:
                value -= 2**64
        elif wire in (1, 5):
            width = 8 if wire == 1 else 4
            if pos + width > len(buf):
                raise ValueError('Truncated protobuf float')
            value = struct.unpack_from('<d' if width == 8 else '<f', buf, pos)[0]
            pos += width
        elif wire == 2:
            size, pos = varint(buf, pos)
            if pos + size > len(buf):
                raise ValueError('Truncated protobuf message')
            value = buf[pos:pos + size]
            pos += size
        else:
            raise ValueError(f'Unsupported protobuf wire type: {wire}')
        result[field] = value
    return result


def field_at(record, *path, default=None):
    for i, key in enumerate(path):
        if key not in record:
            return default
        value = record[key]
        if i == len(path) - 1:
            return value
        if not isinstance(value, bytes):
            raise ValueError(f'Expected nested message: {path}')
        record = protobuf(value)


def read_telemetry(path):
    moov = read_moov(path)
    tracks = [track_info(v) for k, v in boxes(moov) if k == b'trak']
    video = next(t for t in tracks if t['codec'] in ('avc1', 'hvc1', 'hev1'))
    meta = next(t for t in tracks if t['codec'] == 'djmd')
    mvhd = child(moov, b'mvhd')
    created = struct.unpack_from('>Q' if mvhd[0] else '>I', mvhd, 4)[0]
    creation = dt.datetime(1904, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=created)
    packets, protocol, model = [], None, None
    with Path(path).open('rb') as f:
        for i, (offset, size) in enumerate(zip(meta['offsets'], meta['sizes'])):
            f.seek(offset)
            raw = f.read(size)
            if len(raw) != size:
                raise ValueError('Truncated telemetry packet')
            p = protobuf(raw)
            name = field_at(p, 1, 1, 1)
            if name:
                protocol = name.decode('ascii')
                model = field_at(p, 1, 1, 10).decode('ascii')
            if protocol != 'dvtm_wm260l.proto' or model != 'DJIMavic3Classic':
                raise ValueError(f'Unsupported camera/protocol: {model}/{protocol}')
            units = field_at(p, 3, 3, 4, 1, 1, default=0)
            if units not in (0, 1):
                raise ValueError(f'Unknown coordinate units {units}')
            lat, lon = [field_at(p, 3, 3, 4, 1, k) for k in (2, 3)]
            if lat is None or lon is None:
                raise ValueError(f'Packet {i} has no GPS')
            if units == 0:
                lat, lon = np.rad2deg([lat, lon])
            height = field_at(p, 3, 3, 5, 1)
            if height is None:
                raise ValueError(f'Packet {i} has no relative altitude')
            row = dict(sample_index=i, time_s=float(meta['pts'][i]),
                       embedded_frame_id=field_at(p, 3, 1, 1, default=0),
                       device_timestamp_us=field_at(p, 3, 1, 2),
                       lat=float(lat), lon=float(lon), height_m=height / 1000,
                       yaw_deg=field_at(p, 3, 4, 3, 3, default=0) / 10,
                       pitch_deg=field_at(p, 3, 4, 3, 1, default=0) / 10,
                       roll_deg=field_at(p, 3, 4, 3, 2, default=0) / 10)
            if not (-90 < row['lat'] < 90 and -180 < row['lon'] < 180 and row['height_m'] > 0):
                raise ValueError(f'Invalid GPS/height at packet {i}')
            if not all(np.isfinite(row[k]) for k in ('lat', 'lon', 'height_m', 'yaw_deg', 'pitch_deg', 'roll_deg')):
                raise ValueError(f'Non-finite telemetry at packet {i}')
            packets.append(row)
    times = np.sort(video['pts'])
    if np.any(np.diff(times) <= 0) or np.any(np.diff(meta['pts']) <= 0):
        raise ValueError('Nonmonotonic track timing')
    period = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), period, atol=1e-7):
        raise ValueError('VFR input needs a timestamp-aware encoder')
    if len(packets) != len(times):
        raise ValueError('Video/telemetry sample count mismatch')
    timing_error = float(np.max(np.abs(meta['pts'] - times)))
    if timing_error > period:
        raise ValueError(f'Telemetry timing discrepancy exceeds one frame: {timing_error}')
    # The raw embedded ID can skip; the display frame index is always 0..N-1.
    info = dict(protocol=protocol, camera=model, frame_count=len(times),
                fps=1 / period, frame_period_s=period,
                duration_s=float(times[-1] + period - times[0]),
                creation_time_utc=creation.isoformat(),
                telemetry_samples=len(packets), max_track_time_delta_s=timing_error,
                embedded_frame_id_first=packets[0]['embedded_frame_id'],
                embedded_frame_id_last=packets[-1]['embedded_frame_id'])
    return info, times - times[0], packets
