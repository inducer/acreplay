"""
ac_replay_parser.py
-------------------
Pure-Python parser for Assetto Corsa .acreplay binary files.

Format notes (reverse-engineered from community sources and the
abchouhan/acreplay-parser C++ project):

File layout
~~~~~~~~~~~
  [FileHeader]                                   – fixed prefix + lstrings
  [GlobalFrameData × num_frames]                 – sun angle + track-object
                                                   data; (4 + 12×num_track_objects)
                                                   bytes per frame
  for each car (num_cars total):
      [CarHeader]                                – variable-length lstrings
                                                   + num_frames + num_wings
      [FrameHeader]                              – timestamp + weather for
                                                   frame 0 (20 bytes)
      [CarFrameData]                             – physics state for frame 0
      for frame 1 … num_frames−1:
          [num_wings × 4 bytes]                  – per-wing aero data (skip)
          [FrameHeader]                          – timestamp + weather
          [CarFrameData]                         – physics state
      [num_wings × 4 bytes]                      – wing data after last frame
      [u32 count] [count × 8 bytes]              – optional trailing CSP data

All multibyte values are little-endian.  32-bit floats are IEEE 754;
16-bit floats use the binary16 (half-precision) representation.

Coordinate system
~~~~~~~~~~~~~~~~~
AC uses a left-handed, Y-up system (DirectX convention):
  +X  right  |  +Y  up  |  +Z  away from viewer

For a top-down racing-line plot, project onto the XZ plane (drop Y).

NOTE: The format is undocumented by Kunos.  Field offsets and types are
reverse-engineered; see README.md for the validation / adjustment workflow.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, ClassVar


# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

# Per-frame global data: 2 bytes of sun angle + 2 bytes of other data = 4 bytes
# plus 12 bytes for each track object (animated objects like flags, pit gates).
_GLOBAL_FRAME_BASE_BYTES = 4       # bytes per frame before track objects
_TRACK_OBJECT_BYTES = 12           # bytes per track object per frame

# Each CSP trailing data entry (appended after the last car frame) is 8 bytes.
_CSP_TRAILING_ENTRY_BYTES = 8


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


# ---------------------------------------------------------------------------
# File Header
# ---------------------------------------------------------------------------


@dataclass
class FileHeader:
    """
    Binary layout (little-endian):
        version                  : u32        4 bytes
        recording_interval       : f64        8 bytes  (ms between frames)
        weather                  : lstring    variable
        track_name               : lstring    variable
        track_config             : lstring    variable
        num_cars                 : u32        4 bytes
        current_recording_index  : u32        4 bytes
        num_frames               : u32        4 bytes
        num_track_objects        : u32        4 bytes
    """

    version: int
    recording_interval: float  # milliseconds between frames
    weather: str
    track_name: str
    track_config: str
    num_cars: int
    current_recording_index: int
    num_frames: int
    num_track_objects: int

    @classmethod
    def read(cls, fp: BinaryIO) -> FileHeader:
        (version,) = struct.unpack("<I", _read_exactly(fp, 4))
        (recording_interval,) = struct.unpack("<d", _read_exactly(fp, 8))
        weather = _read_lstring(fp)
        track_name = _read_lstring(fp)
        track_config = _read_lstring(fp)
        num_cars, current_recording_index, num_frames, num_track_objects = (
            struct.unpack("<IIII", _read_exactly(fp, 16))
        )
        return cls(
            version=version,
            recording_interval=recording_interval,
            weather=weather,
            track_name=track_name,
            track_config=track_config,
            num_cars=num_cars,
            current_recording_index=current_recording_index,
            num_frames=num_frames,
            num_track_objects=num_track_objects,
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
        num_frames   u32
        num_wings    u32
    """

    _TAIL_FMT: ClassVar[str] = "<II"
    _TAIL_SIZE: ClassVar[int] = struct.calcsize("<II")

    car_model: str
    driver_name: str
    nation_code: str
    team_name: str
    car_skin: str
    num_frames: int
    num_wings: int

    @classmethod
    def read(cls, fp: BinaryIO) -> CarHeader:
        car_model = _read_lstring(fp)
        driver_name = _read_lstring(fp)
        nation_code = _read_lstring(fp)
        team_name = _read_lstring(fp)
        car_skin = _read_lstring(fp)
        raw = _read_exactly(fp, cls._TAIL_SIZE)
        num_frames, num_wings = struct.unpack(cls._TAIL_FMT, raw)
        return cls(
            car_model=car_model,
            driver_name=driver_name,
            nation_code=nation_code,
            team_name=team_name,
            car_skin=car_skin,
            num_frames=num_frames,
            num_wings=num_wings,
        )


# ---------------------------------------------------------------------------
# Per-frame header
# ---------------------------------------------------------------------------


@dataclass
class FrameHeader:
    """
    Binary layout (little-endian, 20 bytes):
        timestamp_ms    u32
        ambient_temp    f32  °C
        road_temp       f32  °C
        wind_speed      f32  m/s
        wind_direction  f32  degrees
    """

    _FMT: ClassVar[str] = "<I4f"
    _SIZE: ClassVar[int] = struct.calcsize("<I4f")

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

# CarFrame binary layout – 256 bytes total.
# Fields that use IEEE 754 binary16 (half-precision) are denoted 'e' in the
# struct format string.  All others are 32-bit ('f') or integer types.
#
# Memory offsets (confirmed against abchouhan/acreplay-parser):
#
#   000  position              float32 × 3          12 B
#   012  rotation (YXZ order)  float16 × 3           6 B
#   018  [padding]                                   2 B
#   020  wheelStaticPos[4][3]  float32 × 12         48 B
#   068  wheelStaticRot[4][3]  float16 × 12         24 B  (YXZ per wheel)
#   092  wheelPos[4][3]        float32 × 12         48 B
#   140  wheelRot[4][3]        float16 × 12         24 B  (YXZ per wheel)
#   164  velocity              float16 × 3           6 B
#   170  rpm                   float16               2 B
#   172  wheelAngVel[4]        float16 × 4           8 B
#   180  slipAngle[4]          float16 × 4           8 B
#   188  slipRatio[4]          float16 × 4           8 B
#   196  ndSlip[4]             float16 × 4           8 B
#   204  load[4]               float16 × 4           8 B
#   212  steerAngle            float16               2 B  (degrees)
#   214  bodyworkNoise         float16               2 B
#   216  drivetrainSpeed       float16               2 B
#   218  [padding]                                   2 B
#   220  currentLapTime        uint32                4 B  (ms)
#   224  lastLapTime           uint32                4 B  (ms)
#   228  bestLapTime           uint32                4 B  (ms)
#   232  fuel                  uint8                 1 B  (0–255)
#   233  fuelPerLap            uint8                 1 B
#   234  gear                  uint8                 1 B  (0=R, 1=N, 2=1st…)
#   235  tireDirt[4]           uint8 × 4             4 B
#   239  damageFrontDeform     uint8                 1 B
#   240  damageRear            uint8                 1 B
#   241  damageLeft            uint8                 1 B
#   242  damageRight           uint8                 1 B
#   243  damageFront           uint8                 1 B
#   244  gas                   uint8                 1 B  (0–255)
#   245  brake                 uint8                 1 B  (0–255)
#   246  currentLap            uint8                 1 B
#   247  [unknown]             uint8                 1 B
#   248  status                uint16                2 B  (bit flags)
#   250  [unknown2]            uint16                2 B
#   252  dirt                  uint8                 1 B
#   253  engineHealth          uint8                 1 B  (0–255)
#   254  boost                 uint8                 1 B
#   255  [padding]                                   1 B
#                                               Total: 256 B
_CAR_FRAME_FMT = (
    "<"
    "3f"   # 000 position XYZ           (float32)
    "3e"   # 012 rotation YXZ           (float16, Euler angles in radians)
    "2x"   # 018 padding
    "12f"  # 020 wheelStaticPosition    (float32, 4 wheels × xyz)
    "12e"  # 068 wheelStaticRotation    (float16, 4 wheels × YXZ)
    "12f"  # 092 wheelPosition          (float32, 4 wheels × xyz)
    "12e"  # 140 wheelRotation          (float16, 4 wheels × YXZ)
    "3e"   # 164 velocity XYZ           (float16, m/s)
    "e"    # 170 rpm                    (float16)
    "4e"   # 172 wheelAngularVelocity   (float16, 4 wheels)
    "4e"   # 180 slipAngle              (float16, 4 wheels)
    "4e"   # 188 slipRatio              (float16, 4 wheels)
    "4e"   # 196 ndSlip                 (float16, 4 wheels)
    "4e"   # 204 load                   (float16, 4 wheels, Newtons)
    "e"    # 212 steerAngle             (float16, degrees)
    "e"    # 214 bodyworkNoise          (float16)
    "e"    # 216 drivetrainSpeed        (float16)
    "2x"   # 218 padding (align u32)
    "3I"   # 220 currentLapTime, lastLapTime, bestLapTime  (uint32, ms)
    "3B"   # 232 fuel, fuelPerLap, gear (uint8; gear: 0=R 1=N 2=1st…)
    "4B"   # 235 tireDirt[4]            (uint8)
    "5B"   # 239 damageFrontDeform, damageRear, damageLeft, damageRight, damageFront
    "2B"   # 244 gas, brake             (uint8, 0–255)
    "2B"   # 246 currentLap, unknown    (uint8)
    "H"    # 248 status                 (uint16 bit flags)
    "H"    # 250 unknown2               (uint16)
    "3B"   # 252 dirt, engineHealth, boost  (uint8)
    "x"    # 255 padding
)
_CAR_FRAME_SIZE = struct.calcsize(_CAR_FRAME_FMT)


@dataclass
class CarFrameData:
    """
    Physics snapshot for one car at one frame.

    Layout: 256 bytes – see _CAR_FRAME_FMT for full offset map.

    Notes
    -----
    * ``gas`` and ``brake`` are normalised to [0, 1] (raw uint8 ÷ 255).
    * ``speed_kmh`` is computed from the velocity vector magnitude.
    * ``gear`` follows the convention -1=R, 0=N, 1=1st, … (raw uint8 is
      0=R, 1=N, 2=1st, …; we subtract 1).
    * ``steer_angle`` is in degrees (stored as float16).
    * ``rpm`` is stored as float16.
    """

    _FMT: ClassVar[str] = _CAR_FRAME_FMT
    _SIZE: ClassVar[int] = _CAR_FRAME_SIZE

    position: Vec3
    rotation: Vec3  # Euler angles (rad): x, y, z  (stored YXZ in file)
    wheel_static_positions: list[Vec3]   # [FL, FR, RL, RR]
    wheel_positions: list[Vec3]          # [FL, FR, RL, RR]
    velocity: Vec3  # world m/s (float16 precision)
    rpm: float
    wheel_angular_velocity: list[float]  # [FL, FR, RL, RR]
    slip_angle: list[float]              # [FL, FR, RL, RR]
    slip_ratio: list[float]              # [FL, FR, RL, RR]
    load: list[float]                    # [FL, FR, RL, RR] Newtons
    steer_angle: float                   # degrees
    current_lap_time: int                # ms
    last_lap_time: int                   # ms
    best_lap_time: int                   # ms
    fuel: int                            # 0–255
    gear: int                            # -1=R, 0=N, 1=1st, …
    tire_dirt: list[int]                 # [FL, FR, RL, RR] 0–255
    damage_front_deform: int             # 0–255
    damage_rear: int
    damage_left: int
    damage_right: int
    damage_front: int
    gas: float                           # 0–1 normalised
    brake: float                         # 0–1 normalised
    current_lap: int
    status: int                          # raw bit flags (see C++ header)
    engine_health: int                   # 0–255
    boost: int                           # 0–255
    speed_kmh: float                     # computed from velocity magnitude

    @classmethod
    def read(cls, fp: BinaryIO) -> CarFrameData:
        raw = _read_exactly(fp, cls._SIZE)
        v = struct.unpack(cls._FMT, raw)
        # Unpack positional fields by index (see _CAR_FRAME_FMT for order)
        (
            px, py, pz,
            rot_y, rot_x, rot_z,  # vectorYXZ order in file
            # wheelStaticPosition[4][3]: FL FR RL RR × xyz
            wsp0x, wsp0y, wsp0z, wsp1x, wsp1y, wsp1z,
            wsp2x, wsp2y, wsp2z, wsp3x, wsp3y, wsp3z,
            # wheelStaticRotation[4][3] (float16, skipped)
            _wsr00, _wsr01, _wsr02, _wsr10, _wsr11, _wsr12,
            _wsr20, _wsr21, _wsr22, _wsr30, _wsr31, _wsr32,
            # wheelPosition[4][3]
            wp0x, wp0y, wp0z, wp1x, wp1y, wp1z,
            wp2x, wp2y, wp2z, wp3x, wp3y, wp3z,
            # wheelRotation[4][3] (float16, skipped)
            _wr00, _wr01, _wr02, _wr10, _wr11, _wr12,
            _wr20, _wr21, _wr22, _wr30, _wr31, _wr32,
            # velocity, rpm
            vx, vy, vz,
            rpm,
            # per-wheel float16 channels
            wav0, wav1, wav2, wav3,   # wheelAngularVelocity
            sa0, sa1, sa2, sa3,       # slipAngle
            sr0, sr1, sr2, sr3,       # slipRatio
            _ns0, _ns1, _ns2, _ns3,   # ndSlip
            ld0, ld1, ld2, ld3,       # load
            steer_angle,
            _bodywork_noise,
            _drivetrain_speed,
            # timing (ms)
            current_lap_time, last_lap_time, best_lap_time,
            # uint8 batch 1
            fuel, _fuel_per_lap, gear_raw,
            # tire dirt
            td0, td1, td2, td3,
            # damage
            dmg_front_deform, dmg_rear, dmg_left, dmg_right, dmg_front,
            # pedals
            gas_raw, brake_raw,
            # lap info
            current_lap, _unknown,
            # status
            status, _unknown2,
            # final bytes
            _dirt, engine_health, boost,
        ) = v

        vx_f = float(vx)
        vy_f = float(vy)
        vz_f = float(vz)
        speed_kmh = math.sqrt(vx_f * vx_f + vy_f * vy_f + vz_f * vz_f) * 3.6

        return cls(
            position=Vec3(px, py, pz),
            rotation=Vec3(rot_x, rot_y, rot_z),
            wheel_static_positions=[
                Vec3(wsp0x, wsp0y, wsp0z),
                Vec3(wsp1x, wsp1y, wsp1z),
                Vec3(wsp2x, wsp2y, wsp2z),
                Vec3(wsp3x, wsp3y, wsp3z),
            ],
            wheel_positions=[
                Vec3(wp0x, wp0y, wp0z),
                Vec3(wp1x, wp1y, wp1z),
                Vec3(wp2x, wp2y, wp2z),
                Vec3(wp3x, wp3y, wp3z),
            ],
            velocity=Vec3(vx_f, vy_f, vz_f),
            rpm=float(rpm),
            wheel_angular_velocity=[float(wav0), float(wav1), float(wav2), float(wav3)],
            slip_angle=[float(sa0), float(sa1), float(sa2), float(sa3)],
            slip_ratio=[float(sr0), float(sr1), float(sr2), float(sr3)],
            load=[float(ld0), float(ld1), float(ld2), float(ld3)],
            steer_angle=float(steer_angle),
            current_lap_time=current_lap_time,
            last_lap_time=last_lap_time,
            best_lap_time=best_lap_time,
            fuel=fuel,
            gear=gear_raw - 1,  # convert: 0→-1(R), 1→0(N), 2→1(1st), …
            tire_dirt=[td0, td1, td2, td3],
            damage_front_deform=dmg_front_deform,
            damage_rear=dmg_rear,
            damage_left=dmg_left,
            damage_right=dmg_right,
            damage_front=dmg_front,
            gas=gas_raw / 255.0,
            brake=brake_raw / 255.0,
            current_lap=current_lap,
            status=status,
            engine_health=engine_health,
            boost=boost,
            speed_kmh=speed_kmh,
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

            # Skip global per-frame data: sun angles + track-object state.
            # Each frame takes (_GLOBAL_FRAME_BASE_BYTES +
            # _TRACK_OBJECT_BYTES × num_track_objects) bytes.
            n_tobj = header.num_track_objects
            global_frame_bytes = (
                _GLOBAL_FRAME_BASE_BYTES + _TRACK_OBJECT_BYTES * n_tobj
            ) * header.num_frames
            fp.seek(global_frame_bytes, 1)

            # Read car data in car-major order, collecting per-car frames.
            cars: list[CarHeader] = []
            # car_frame_headers[car_index][frame_index]
            car_frame_headers: list[list[FrameHeader]] = []
            # car_frames[car_index][frame_index]
            car_frames_raw: list[list[CarFrameData]] = []

            for _ in range(header.num_cars):
                car_hdr = CarHeader.read(fp)
                cars.append(car_hdr)

                n = car_hdr.num_frames
                wings_bytes = car_hdr.num_wings * 4

                fh_list: list[FrameHeader] = []
                cf_list: list[CarFrameData] = []

                for frame_i in range(n):
                    # FrameHeader precedes every CarFrameData.
                    # For frame 0 it immediately follows the CarHeader;
                    # for subsequent frames it follows the inter-frame wing data.
                    fh_list.append(FrameHeader.read(fp))
                    cf_list.append(CarFrameData.read(fp))

                    if frame_i < n - 1:
                        fp.seek(wings_bytes, 1)  # skip wing data between frames

                # Skip wing data after the last frame, then trailing CSP bytes.
                fp.seek(wings_bytes, 1)
                (count,) = struct.unpack("<I", _read_exactly(fp, 4))
                if count > 0:
                    fp.seek(count * _CSP_TRAILING_ENTRY_BYTES, 1)

                car_frame_headers.append(fh_list)
                car_frames_raw.append(cf_list)

        # Reorganise from car-major to frame-major order.
        # Use the minimum num_frames across all cars as the common length.
        n_frames = min((len(cf) for cf in car_frames_raw), default=0)

        frame_headers = car_frame_headers[0][:n_frames] if car_frame_headers else []
        frames: list[list[CarFrameData]] = [
            [car_frames_raw[car_i][frame_i] for car_i in range(len(cars))]
            for frame_i in range(n_frames)
        ]

        return cls(
            header=header,
            cars=cars,
            frame_headers=frame_headers,
            frames=frames,
        )
