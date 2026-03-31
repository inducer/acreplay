"""
ac_replay_parser.py
-------------------
Pure-Python parser for Assetto Corsa .acreplay binary files.

Format notes (reverse-engineered from community sources and the
abchouhan/acreplay-parser C++ project):

File layout
~~~~~~~~~~~
  [FileHeader]                    – fixed-size prefix + variable track name
  [CarHeader × num_cars]          – one per car; variable-length strings
  for each frame (num_frames total):
      [FrameHeader]               – timestamp + per-frame weather metadata
      [CarFrameData × num_cars]   – physics state for every car

All multibyte values are little-endian.  Floats are IEEE 754 32-bit.

Coordinate system
~~~~~~~~~~~~~~~~~
AC uses a left-handed, Y-up system (DirectX convention):
  +X  right  |  +Y  up  |  +Z  away from viewer

For a top-down racing-line plot, project onto the XZ plane (drop Y).

NOTE: The format is undocumented by Kunos.  Field offsets and types are
reverse-engineered; see README.md for the validation / adjustment workflow.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, ClassVar


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_exactly(fp: BinaryIO, n: int) -> bytes:
    data = fp.read(n)
    if len(data) != n:
        raise EOFError(
            f"Expected {n} bytes but got {len(data)} "
            f"(file offset ~{fp.tell() - len(data)})"
        )
    return data


def _read_lstring(fp: BinaryIO) -> str:
    """4-byte length-prefixed UTF-8 string."""
    (length,) = struct.unpack("<I", _read_exactly(fp, 4))
    return _read_exactly(fp, length).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


@dataclass
class Vec3:
    x: float
    y: float
    z: float

    @classmethod
    def read(cls, fp: BinaryIO) -> Vec3:
        x, y, z = struct.unpack("<3f", _read_exactly(fp, 12))
        return cls(x, y, z)


@dataclass
class Quat:
    x: float
    y: float
    z: float
    w: float

    @classmethod
    def read(cls, fp: BinaryIO) -> Quat:
        x, y, z, w = struct.unpack("<4f", _read_exactly(fp, 16))
        return cls(x, y, z, w)


# ---------------------------------------------------------------------------
# File Header
# ---------------------------------------------------------------------------


@dataclass
class FileHeader:
    """
    Binary layout (little-endian):
        version         : u32   4 bytes
        num_frames      : u32   4 bytes
        frame_dt_ms     : f32   4 bytes
        num_cars        : u32   4 bytes
        track_name      : lstring (variable)
    """

    _FIXED_FMT: ClassVar[str] = "<IIfI"
    _FIXED_SIZE: ClassVar[int] = struct.calcsize("<IIfI")

    version: int
    num_frames: int
    frame_dt_ms: float
    num_cars: int
    track_name: str

    @classmethod
    def read(cls, fp: BinaryIO) -> FileHeader:
        raw = _read_exactly(fp, cls._FIXED_SIZE)
        version, num_frames, frame_dt_ms, num_cars = struct.unpack(cls._FIXED_FMT, raw)
        return cls(
            version=version,
            num_frames=num_frames,
            frame_dt_ms=frame_dt_ms,
            num_cars=num_cars,
            track_name=_read_lstring(fp),
        )


# ---------------------------------------------------------------------------
# Per-car session metadata
# ---------------------------------------------------------------------------


@dataclass
class CarHeader:
    """
    Binary layout:
        car_model    lstring
        driver_name  lstring
        nation_code  lstring
        team_name    lstring
        car_skin     lstring
        setup_name   lstring
        is_human     u32
        number_plate u32
        _pad         4 bytes
    """

    _TAIL_FMT: ClassVar[str] = "<II4x"
    _TAIL_SIZE: ClassVar[int] = struct.calcsize("<II4x")

    car_model: str
    driver_name: str
    nation_code: str
    team_name: str
    car_skin: str
    setup_name: str
    is_human: bool
    number_plate: int

    @classmethod
    def read(cls, fp: BinaryIO) -> CarHeader:
        car_model = _read_lstring(fp)
        driver_name = _read_lstring(fp)
        nation_code = _read_lstring(fp)
        team_name = _read_lstring(fp)
        car_skin = _read_lstring(fp)
        setup_name = _read_lstring(fp)
        raw = _read_exactly(fp, cls._TAIL_SIZE)
        is_human_u32, number_plate = struct.unpack(cls._TAIL_FMT, raw)
        return cls(
            car_model=car_model,
            driver_name=driver_name,
            nation_code=nation_code,
            team_name=team_name,
            car_skin=car_skin,
            setup_name=setup_name,
            is_human=bool(is_human_u32),
            number_plate=number_plate,
        )


# ---------------------------------------------------------------------------
# Per-frame header
# ---------------------------------------------------------------------------


@dataclass
class FrameHeader:
    """
    Binary layout (little-endian, 24 bytes):
        timestamp_ms    u32
        ambient_temp    f32  °C
        road_temp       f32  °C
        wind_speed      f32  m/s
        wind_direction  f32  degrees
        _pad            4 bytes
    """

    _FMT: ClassVar[str] = "<I4f4x"
    _SIZE: ClassVar[int] = struct.calcsize("<I4f4x")

    timestamp_ms: int
    ambient_temp: float
    road_temp: float
    wind_speed: float
    wind_direction: float

    @classmethod
    def read(cls, fp: BinaryIO) -> FrameHeader:
        raw = _read_exactly(fp, cls._SIZE)
        ts, ambient, road, wind_spd, wind_dir = struct.unpack(cls._FMT, raw)
        return cls(
            timestamp_ms=ts,
            ambient_temp=ambient,
            road_temp=road,
            wind_speed=wind_spd,
            wind_direction=wind_dir,
        )


# ---------------------------------------------------------------------------
# Per-car physics state
# ---------------------------------------------------------------------------

# Build the format string as a module-level constant to avoid the ClassVar
# / calcsize interaction that tripped up the first version.
_CAR_FRAME_FMT = (
    "<"
    "3f"  # 000 position        (XYZ)
    "4f"  # 012 rotation        (quaternion XYZW)
    "3f"  # 028 velocity        (world m/s)
    "3f"  # 040 angular_vel     (world rad/s)
    "f"  # 052 steer_angle     (rad)
    "f"  # 056 gas             (0–1)
    "f"  # 060 brake           (0–1)
    "f"  # 064 clutch          (0–1)
    "i"  # 068 gear            (-1 R, 0 N, 1…n)
    "f"  # 072 rpm
    "f"  # 076 speed_kmh
    "IIII"  # 080 has_abs abs_active has_tc tc_active
    "f"  # 096 fuel            (litres)
    "II"  # 100 drs_active drs_available
    "ff"  # 108 ers_recovery ers_deploy
    "II"  # 116 in_pit is_retired
    "f"  # 124 engine_life     (0–1)
    "4x"  # 128 _pad
)
_CAR_FRAME_SIZE = struct.calcsize(_CAR_FRAME_FMT)


@dataclass
class CarFrameData:
    """
    Physics snapshot for one car at one frame.  Layout is documented
    in the _CAR_FRAME_FMT constant above.  Total: 132 bytes.

    NOTE: carries known uncertainty — adjust _CAR_FRAME_FMT (module-level)
    if ac_probe.py reports a non-zero size delta.
    """

    _FMT: ClassVar[str] = _CAR_FRAME_FMT
    _SIZE: ClassVar[int] = _CAR_FRAME_SIZE

    position: Vec3
    rotation: Quat
    velocity: Vec3
    angular_vel: Vec3
    steer_angle: float
    gas: float
    brake: float
    clutch: float
    gear: int
    rpm: float
    speed_kmh: float
    has_abs: bool
    abs_active: bool
    has_tc: bool
    tc_active: bool
    fuel: float
    drs_active: bool
    drs_available: bool
    ers_recovery: float
    ers_deploy: float
    in_pit: bool
    is_retired: bool
    engine_life: float

    @classmethod
    def read(cls, fp: BinaryIO) -> CarFrameData:
        raw = _read_exactly(fp, cls._SIZE)
        (
            px,
            py,
            pz,
            rx,
            ry,
            rz,
            rw,
            vx,
            vy,
            vz,
            ax,
            ay,
            az,
            steer,
            gas,
            brake,
            clutch,
            gear,
            rpm,
            speed,
            has_abs,
            abs_act,
            has_tc,
            tc_act,
            fuel,
            drs_act,
            drs_avail,
            ers_rec,
            ers_dep,
            in_pit,
            retired,
            eng_life,
        ) = struct.unpack(cls._FMT, raw)
        return cls(
            position=Vec3(px, py, pz),
            rotation=Quat(rx, ry, rz, rw),
            velocity=Vec3(vx, vy, vz),
            angular_vel=Vec3(ax, ay, az),
            steer_angle=steer,
            gas=gas,
            brake=brake,
            clutch=clutch,
            gear=gear,
            rpm=rpm,
            speed_kmh=speed,
            has_abs=bool(has_abs),
            abs_active=bool(abs_act),
            has_tc=bool(has_tc),
            tc_active=bool(tc_act),
            fuel=fuel,
            drs_active=bool(drs_act),
            drs_available=bool(drs_avail),
            ers_recovery=ers_rec,
            ers_deploy=ers_dep,
            in_pit=bool(in_pit),
            is_retired=bool(retired),
            engine_life=eng_life,
        )


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------


@dataclass
class Replay:
    """
    Complete parsed replay.

    frames[frame_index][car_index] → CarFrameData
    """

    header: FileHeader
    cars: list[CarHeader]
    frame_headers: list[FrameHeader]
    frames: list[list[CarFrameData]]

    def positions_for_car(self, car_index: int) -> list[Vec3]:
        return [frame[car_index].position for frame in self.frames]

    def timestamps_ms(self) -> list[int]:
        return [fh.timestamp_ms for fh in self.frame_headers]

    @classmethod
    def from_file(cls, path: str | Path) -> Replay:
        path = Path(path)
        with path.open("rb") as fp:
            header = FileHeader.read(fp)
            cars = [CarHeader.read(fp) for _ in range(header.num_cars)]
            frame_headers: list[FrameHeader] = []
            frames: list[list[CarFrameData]] = []
            for _ in range(header.num_frames):
                fh = FrameHeader.read(fp)
                frame_headers.append(fh)
                frames.append([CarFrameData.read(fp) for _ in range(header.num_cars)])
        return cls(
            header=header,
            cars=cars,
            frame_headers=frame_headers,
            frames=frames,
        )
