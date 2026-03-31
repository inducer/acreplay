"""
ac_format_probe.py
------------------
Diagnostic utility: reads the raw bytes of an .acreplay file and prints
structured hex/field dumps to help verify that the struct layouts in
acreplay/replay.py are correctly calibrated against a real file.

Since the .acreplay format is undocumented, this tool exists to quickly
catch misalignment:  if a field value looks physically impossible (e.g.
recording_interval = 2.3e+25) the struct layout needs adjustment.

Usage
~~~~~
    ac-replay-probe replay.acreplay

What it prints
~~~~~~~~~~~~~~
  1. Raw hex dump of the first 256 bytes
  2. Attempted FileHeader parse with field values
  3. Attempted first CarHeader parse
  4. Attempted first FrameHeader + first CarFrameData
  5. Sanity checks (are field values physically plausible?)
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

from .replay import (
    _GLOBAL_FRAME_BASE_BYTES,
    _TRACK_OBJECT_BYTES,
)


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
    # Layout: version(u32) + recording_interval(f64) + weather(lstr) +
    #         track_name(lstr) + track_config(lstr) +
    #         num_cars(u32) + current_recording_index(u32) +
    #         num_frames(u32) + num_track_objects(u32)
    print(f"\n{'─' * 60}")
    print("  FileHeader parse attempt")
    print(f"{'─' * 60}")
    offset = 0

    if len(raw) < 4:
        print("  ERROR: file too small for FileHeader")
        return
    (version,) = struct.unpack_from("<I", raw, offset)
    offset += 4

    if len(raw) < offset + 8:
        print("  ERROR: file too small for recording_interval")
        return
    (recording_interval,) = struct.unpack_from("<d", raw, offset)
    offset += 8

    weather, offset = _try_read_string(raw, offset)
    track_name, offset = _try_read_string(raw, offset)
    track_config, offset = _try_read_string(raw, offset)

    if len(raw) < offset + 16:
        print("  ERROR: file too small for num_cars/num_frames")
        return
    num_cars, current_recording_index, num_frames, num_track_objects = (
        struct.unpack_from("<IIII", raw, offset)
    )
    offset += 16

    print(f"  version                 = {version}")
    print(f"  recording_interval      = {recording_interval:.4f} ms")
    print(f"  weather                 = {weather!r}")
    print(f"  track_name              = {track_name!r}")
    print(f"  track_config            = {track_config!r}")
    print(f"  num_cars                = {num_cars}")
    print(f"  current_recording_index = {current_recording_index}")
    print(f"  num_frames              = {num_frames}")
    print(f"  num_track_objects       = {num_track_objects}")

    checks = {
        "version in [1..20]": 1 <= version <= 20,
        "num_frames in [1..1_000_000]": 1 <= num_frames <= 1_000_000,
        "recording_interval in [1..100]": 1.0 <= recording_interval <= 100.0,
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

    # Skip global per-frame data (sun angles + track objects)
    global_frame_bytes = (
        _GLOBAL_FRAME_BASE_BYTES + _TRACK_OBJECT_BYTES * num_track_objects
    ) * num_frames
    car_data_offset = offset + global_frame_bytes

    # 3. First CarHeader
    # Layout: 5 lstrings + num_frames(u32) + num_wings(u32)
    print(f"\n{'─' * 60}")
    print("  CarHeader[0] parse attempt")
    print(f"{'─' * 60}")
    print(f"  (car data starts at file offset {car_data_offset})")
    offset = car_data_offset

    string_fields = ["car_model", "driver_name", "nation_code", "team_name", "car_skin"]
    for field_name in string_fields:
        val, offset = _try_read_string(raw, offset)
        print(f"  {field_name:<16} = {val!r}")

    tail_fmt = "<II"
    tail_size = struct.calcsize(tail_fmt)
    car_num_frames = 0
    car_num_wings = 0
    if offset + tail_size <= len(raw):
        car_num_frames, car_num_wings = struct.unpack_from(tail_fmt, raw, offset)
        offset += tail_size
        print(f"  {'num_frames':<16} = {car_num_frames}")
        print(f"  {'num_wings':<16} = {car_num_wings}")
    else:
        print("  ERROR: file too small for CarHeader tail")
        return

    # 4. First FrameHeader (20 bytes, immediately after CarHeader)
    print(f"\n{'─' * 60}")
    print("  FrameHeader[0] parse attempt")
    print(f"{'─' * 60}")
    fh_fmt = "<I4f"
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

    # 5. First CarFrameData (256 bytes)
    print(f"\n{'─' * 60}")
    print("  CarFrameData[0][0] parse attempt")
    print(f"{'─' * 60}")

    # Use the same format string as CarFrameData._FMT in replay.py
    cfd_fmt = (
        "<"
        "3f"   # position
        "3e"   # rotation (YXZ float16)
        "2x"   # padding
        "12f"  # wheelStaticPosition
        "12e"  # wheelStaticRotation
        "12f"  # wheelPosition
        "12e"  # wheelRotation
        "3e"   # velocity
        "e"    # rpm
        "4e"   # wheelAngularVelocity
        "4e"   # slipAngle
        "4e"   # slipRatio
        "4e"   # ndSlip
        "4e"   # load
        "e"    # steerAngle
        "e"    # bodyworkNoise
        "e"    # drivetrainSpeed
        "2x"   # padding
        "3I"   # currentLapTime, lastLapTime, bestLapTime
        "3B"   # fuel, fuelPerLap, gear
        "4B"   # tireDirt[4]
        "5B"   # damage fields
        "2B"   # gas, brake
        "2B"   # currentLap, unknown
        "H"    # status
        "H"    # unknown2
        "3B"   # dirt, engineHealth, boost
        "x"    # padding
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
    px, py, pz = vals[0], vals[1], vals[2]
    rot_y, rot_x, rot_z = vals[3], vals[4], vals[5]
    # Index map (mirrors _CAR_FRAME_FMT in replay.py):
    #   0-2:  position xyz
    #   3-5:  rotation YXZ
    #   6-17: wheelStaticPosition[4][3]
    #  18-29: wheelStaticRotation[4][3]
    #  30-41: wheelPosition[4][3]
    #  42-53: wheelRotation[4][3]
    #  54-56: velocity xyz
    #  57:    rpm
    #  58-73: wheelAngVel, slipAngle, slipRatio, ndSlip  (4×4=16)
    #  74-77: load[4]
    #  78:    steerAngle
    #  79-80: bodyworkNoise, drivetrainSpeed
    #  81-83: currentLapTime, lastLapTime, bestLapTime
    #  84-86: fuel, fuelPerLap, gear
    #  87-90: tireDirt[4]
    #  91-95: damage fields
    #  96-97: gas, brake
    #  98:    currentLap
    # 100:    status
    # 103:    engineHealth
    vx, vy, vz = float(vals[54]), float(vals[55]), float(vals[56])
    rpm = float(vals[57])
    steer_angle = float(vals[78])
    current_lap_time = vals[81]
    last_lap_time = vals[82]
    best_lap_time = vals[83]
    fuel = vals[84]
    gear_raw = vals[86]
    gas_raw = vals[96]
    brake_raw = vals[97]
    current_lap = vals[98]
    engine_health = vals[103]

    speed_kmh = math.sqrt(vx * vx + vy * vy + vz * vz) * 3.6
    gear = gear_raw - 1  # 0=R→-1, 1=N→0, 2=1st→1, …

    print(f"  pos_x            = {px:.4f}")
    print(f"  pos_y            = {py:.4f}")
    print(f"  pos_z            = {pz:.4f}")
    print(f"  rot_x            = {rot_x:.4f} rad")
    print(f"  rot_y            = {rot_y:.4f} rad")
    print(f"  rot_z            = {rot_z:.4f} rad")
    print(f"  vel_x            = {vx:.4f} m/s")
    print(f"  vel_y            = {vy:.4f} m/s")
    print(f"  vel_z            = {vz:.4f} m/s")
    print(f"  speed_kmh        = {speed_kmh:.2f} (computed)")
    print(f"  rpm              = {rpm:.1f}")
    print(f"  steer_angle      = {steer_angle:.2f}°")
    print(f"  gear             = {gear}  (raw={gear_raw}: 0=R 1=N 2=1st…)")
    print(f"  gas              = {gas_raw / 255.0:.3f}  (raw={gas_raw})")
    print(f"  brake            = {brake_raw / 255.0:.3f}  (raw={brake_raw})")
    print(f"  fuel             = {fuel}")
    print(f"  current_lap      = {current_lap}")
    print(f"  current_lap_time = {current_lap_time} ms")
    print(f"  last_lap_time    = {last_lap_time} ms")
    print(f"  best_lap_time    = {best_lap_time} ms")
    print(f"  engine_health    = {engine_health}")

    offset += cfd_size

    cfd_checks = {
        "pos_y plausible (|y| < 5000)": abs(py) < 5000,
        "speed_kmh in [0..500]": 0 <= speed_kmh <= 500,
        "rpm in [0..20000]": 0 <= rpm <= 20_000,
        "gear in [-1..8]": -1 <= gear <= 8,
        "gas in [0..1]": 0.0 <= gas_raw / 255.0 <= 1.0,
        "brake in [0..1]": 0.0 <= brake_raw / 255.0 <= 1.0,
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

    # Frame size = FrameHeader + CarFrameData + wings_data_between_frames
    wings_between = car_num_wings * 4
    print(f"\n  Offset after first frame: {offset} / {len(raw)} bytes")
    # Remaining frames start after the first CarFrameData; each subsequent
    # frame is: wings_data + FrameHeader + CarFrameData.
    # The per-car footer (wing data after last frame + 4-byte CSP count) must
    # be subtracted so it does not inflate the estimated remaining-frame count.
    min_footer_bytes = wings_between + 4  # wings after last frame + count u32
    remaining = len(raw) - offset
    effective_remaining = remaining - min_footer_bytes
    if car_num_frames > 1:
        per_remaining_frame = wings_between + fh_size + cfd_size
        remaining_frames_est = (
            effective_remaining / per_remaining_frame if per_remaining_frame else 0
        )
        expected_remaining = car_num_frames - 1
    else:
        per_remaining_frame = 0
        remaining_frames_est = 0.0
        expected_remaining = 0
    print(f"  Remaining bytes: {remaining}")
    print(
        f"  Estimated remaining frames: {remaining_frames_est:.1f} "
        f"(expected {expected_remaining})"
    )
    if expected_remaining > 0:
        ratio = remaining_frames_est / expected_remaining
        if abs(ratio - 1.0) < 0.02:
            print(
                f"  ✓  Frame count estimate matches (ratio={ratio:.3f}) "
                f"– struct sizes appear correct."
            )
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
