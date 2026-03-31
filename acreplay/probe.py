"""
ac_format_probe.py
------------------
Diagnostic utility: reads the raw bytes of an .acreplay file and prints
structured hex/field dumps to help verify that the struct layouts in
ac_replay_parser.py are correctly calibrated against a real file.

Since the .acreplay format is undocumented, this tool exists to quickly
catch misalignment:  if a field value looks physically impossible (e.g.
frame_dt_ms = 2.3e+25) the struct layout needs adjustment.

Usage
~~~~~
    python ac_format_probe.py replay.acreplay

What it prints
~~~~~~~~~~~~~~
  1. Raw hex dump of the first 256 bytes
  2. Attempted FileHeader parse with field values
  3. Attempted first CarHeader parse
  4. Attempted first FrameHeader + first CarFrameData
  5. Sanity checks (are field values physically plausible?)
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path


def hexdump(data: bytes, width: int = 16, label: str = "") -> None:
    if label:
        print(f"\n{'─' * 60}")
        print(f"  {label}")
        print(f"{'─' * 60}")
    for i in range(0, len(data), width):
        chunk = data[i : i + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        print(f"  {i:04x}  {hex_part:<{width * 3}}  {ascii_part}")


def _try_read_string(data: bytes, offset: int) -> tuple[str, int]:
    """Try to read a 4-byte length-prefixed string from data[offset:]."""
    if offset + 4 > len(data):
        return "<EOF>", offset
    (length,) = struct.unpack_from("<I", data, offset)
    offset += 4
    if length > 4096 or offset + length > len(data):
        return f"<length={length} suspicious or truncated>", offset
    s = data[offset : offset + length].decode("utf-8", errors="replace")
    return s, offset + length


def probe(path: str | Path) -> None:
    path = Path(path)
    print(f"\n{'═' * 60}")
    print(f"  ACREPLAY FORMAT PROBE:  {path.name}")
    print(f"  file size: {path.stat().st_size:,} bytes")
    print(f"{'═' * 60}")

    raw = path.read_bytes()

    # 1. Raw hex dump
    hexdump(raw[:256], label="First 256 bytes (raw hex)")

    # 2. FileHeader
    print(f"\n{'─' * 60}")
    print("  FileHeader parse attempt")
    print(f"{'─' * 60}")
    offset = 0
    fmt = "<IIf I"
    size = struct.calcsize(fmt)
    if len(raw) < size:
        print("  ERROR: file too small for FileHeader")
        return
    version, num_frames, frame_dt_ms, num_cars = struct.unpack_from(fmt, raw, offset)
    offset += size
    track_name, offset = _try_read_string(raw, offset)

    print(f"  version        = {version}")
    print(f"  num_frames     = {num_frames}")
    print(f"  frame_dt_ms    = {frame_dt_ms:.4f}")
    print(f"  num_cars       = {num_cars}")
    print(f"  track_name     = {track_name!r}")

    # Sanity
    checks = {
        "version in [1..20]": 1 <= version <= 20,
        "num_frames in [1..1_000_000]": 1 <= num_frames <= 1_000_000,
        "frame_dt_ms in [1..100]": 1.0 <= frame_dt_ms <= 100.0,
        "num_cars in [1..100]": 1 <= num_cars <= 100,
        "track_name non-empty": bool(track_name.strip()),
    }
    print("\n  Sanity checks:")
    all_ok = True
    for desc, ok in checks.items():
        mark = "✓" if ok else "✗ FAIL"
        if not ok:
            all_ok = False
        print(f"    [{mark}]  {desc}")
    if not all_ok:
        print("\n  ⚠  One or more sanity checks failed.")
        print("     The struct layout in ac_replay_parser.py likely needs adjustment.")
        print(
            "     Compare the hex dump above with the field descriptions in the source."
        )

    # 3. First CarHeader
    print(f"\n{'─' * 60}")
    print("  CarHeader[0] parse attempt")
    print(f"{'─' * 60}")
    string_fields = [
        "car_model",
        "driver_name",
        "nation_code",
        "team_name",
        "car_skin",
        "setup_name",
    ]
    for field_name in string_fields:
        val, offset = _try_read_string(raw, offset)
        print(f"  {field_name:<16} = {val!r}")

    tail_fmt = "<II4x"
    tail_size = struct.calcsize(tail_fmt)
    if offset + tail_size <= len(raw):
        is_human, number = struct.unpack_from(tail_fmt, raw, offset)
        offset += tail_size
        print(f"  {'is_human':<16} = {bool(is_human)}")
        print(f"  {'number_plate':<16} = {number}")
    else:
        print("  ERROR: file too small for CarHeader tail")
        return

    # 4. First FrameHeader
    print(f"\n{'─' * 60}")
    print("  FrameHeader[0] parse attempt")
    print(f"{'─' * 60}")
    fh_fmt = "<I 4f 4x"
    fh_size = struct.calcsize(fh_fmt)
    if offset + fh_size > len(raw):
        print("  ERROR: file too small for FrameHeader")
        return
    ts, ambient, road, wind_spd, wind_dir = struct.unpack_from(fh_fmt, raw, offset)
    offset += fh_size
    print(f"  timestamp_ms   = {ts}")
    print(f"  ambient_temp   = {ambient:.2f} °C")
    print(f"  road_temp      = {road:.2f} °C")
    print(f"  wind_speed     = {wind_spd:.2f} m/s")
    print(f"  wind_direction = {wind_dir:.2f}°")

    fh_checks = {
        "timestamp_ms == 0 (first frame)": ts == 0,
        "ambient_temp in [-20..60]": -20 <= ambient <= 60,
        "road_temp in [-20..80]": -20 <= road <= 80,
        "wind_speed in [0..50]": 0 <= wind_spd <= 50,
    }
    print("\n  Sanity checks:")
    for desc, ok in fh_checks.items():
        mark = "✓" if ok else "✗ FAIL"
        print(f"    [{mark}]  {desc}")

    # 5. First CarFrameData
    print(f"\n{'─' * 60}")
    print("  CarFrameData[0][0] parse attempt")
    print(f"{'─' * 60}")
    cfd_fmt = (
        "<"
        "3f 4f 3f 3f"  # pos, rot, vel, ang_vel
        "f f f f"  # steer, gas, brake, clutch
        "i f f"  # gear, rpm, speed
        "I I I I"  # has_abs, abs_act, has_tc, tc_act
        "f"  # fuel
        "I I"  # drs_act, drs_avail
        "f f"  # ers_rec, ers_dep
        "I I"  # in_pit, retired
        "f 4x"  # engine_life, pad
    )
    cfd_size = struct.calcsize(cfd_fmt)
    if offset + cfd_size > len(raw):
        print(
            f"  ERROR: file too small for CarFrameData "
            f"(need {cfd_size} bytes at offset {offset}, "
            f"file has {len(raw)} bytes)"
        )
        return

    vals = struct.unpack_from(cfd_fmt, raw, offset)
    labels = [
        "pos_x",
        "pos_y",
        "pos_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "rot_w",
        "vel_x",
        "vel_y",
        "vel_z",
        "ang_x",
        "ang_y",
        "ang_z",
        "steer",
        "gas",
        "brake",
        "clutch",
        "gear",
        "rpm",
        "speed_kmh",
        "has_abs",
        "abs_act",
        "has_tc",
        "tc_act",
        "fuel",
        "drs_act",
        "drs_avail",
        "ers_rec",
        "ers_dep",
        "in_pit",
        "retired",
        "engine_life",
    ]
    for label, val in zip(labels, vals, strict=False):
        print(f"  {label:<16} = {val}")

    offset += cfd_size

    _pos_x, pos_y, _pos_z = vals[0], vals[1], vals[2]
    speed = vals[19]
    rpm = vals[18]
    gear = vals[17]
    gas = vals[15]
    brake = vals[16]

    cfd_checks = {
        "pos_y plausible (|y| < 5000)": abs(pos_y) < 5000,
        "speed_kmh in [0..500]": 0 <= speed <= 500,
        "rpm in [0..20000]": 0 <= rpm <= 20_000,
        "gear in [-1..8]": -1 <= gear <= 8,
        "gas in [0..1]": 0.0 <= gas <= 1.0,
        "brake in [0..1]": 0.0 <= brake <= 1.0,
    }
    print("\n  Sanity checks:")
    any_fail = False
    for desc, ok in cfd_checks.items():
        mark = "✓" if ok else "✗ FAIL"
        if not ok:
            any_fail = True
        print(f"    [{mark}]  {desc}")

    if any_fail:
        print(
            "\n  ⚠  CarFrameData sanity checks failed.\n"
            "     The CarFrameData struct layout needs adjustment.\n"
            "     Check field sizes and padding at the offset shown above."
        )
    else:
        print("\n  All CarFrameData sanity checks passed.")

    print(f"\n  Offset after first frame: {offset} / {len(raw)} bytes")
    remaining = len(raw) - offset
    expected_per_frame = fh_size + cfd_size * num_cars
    remaining_frames_est = remaining / expected_per_frame if expected_per_frame else 0
    print(f"  Remaining bytes: {remaining}")
    print(
        f"  Estimated remaining frames: {remaining_frames_est:.1f} "
        f"(expected {num_frames - 1})"
    )
    ratio = remaining_frames_est / (num_frames - 1) if num_frames > 1 else float("nan")
    if abs(ratio - 1.0) < 0.02:
        print("  ✓  Frame count estimate matches – struct sizes appear correct.")
    else:
        print(
            f"  ✗  Frame count ratio = {ratio:.3f}  (expected ~1.0).\n"
            "     Struct sizes are likely wrong."
        )
    print()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <replay.acreplay>")
        sys.exit(1)
    probe(sys.argv[1])


if __name__ == "__main__":
    main()
