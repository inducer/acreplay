"""
ac_track_data.py
----------------
Loaders for Assetto Corsa track boundary and AI spline data.

Track data lives inside the AC installation under:
    content/tracks/<track_name>/[<layout>/]

Relevant files
~~~~~~~~~~~~~~
data/side_l.csv     – left boundary  (X, Y, Z per row, comma-separated)
data/side_r.csv     – right boundary (same format)
ai/fast_lane.ai     – AI reference line (binary)

fast_lane.ai format (community-documented)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    [AiSplineHeader]
        version     : u32   always 7 for current AC
        num_points  : u32
        _pad        : 8 bytes

    [AiSplinePoint × num_points]
        position    : 3 × f32   12 bytes   world XYZ
        length      : f32        4 bytes   cumulative arc-length (m)
        id          : u32        4 bytes   sequential index
        _pad        : 12 bytes

    [AiSplineSidePoint × num_points]   left boundary offsets
        position    : 3 × f32   12 bytes

    [AiSplineSidePoint × num_points]   right boundary offsets
        position    : 3 × f32   12 bytes
"""

from __future__ import annotations

import csv
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


# ---------------------------------------------------------------------------
# Shared type
# ---------------------------------------------------------------------------


@dataclass
class Point3:
    """Lightweight 3D point for static geometry."""

    x: float
    y: float
    z: float


# ---------------------------------------------------------------------------
# Boundary CSV loader
# ---------------------------------------------------------------------------


def load_boundary_csv(path: str | Path) -> list[Point3]:
    """
    Read a track boundary CSV (side_l.csv / side_r.csv).

    Expected format: three float columns per row, no header line.
    Extra columns are silently ignored.
    """
    points: list[Point3] = []
    path = Path(path)
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        for row_num, row in enumerate(reader, start=1):
            if not row or row[0].startswith("#"):
                continue
            try:
                x, y, z = float(row[0]), float(row[1]), float(row[2])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"{path.name}: cannot parse row {row_num}: {row!r}"
                ) from exc
            points.append(Point3(x, y, z))
    return points


# ---------------------------------------------------------------------------
# AI spline binary loader
# ---------------------------------------------------------------------------

_AI_HDR_FMT = "<II8x"
_AI_HDR_SIZE = struct.calcsize(_AI_HDR_FMT)

_AI_PT_FMT = "<3ffI12x"
_AI_PT_SIZE = struct.calcsize(_AI_PT_FMT)

_AI_SIDE_FMT = "<3f"
_AI_SIDE_SIZE = struct.calcsize(_AI_SIDE_FMT)


@dataclass
class AiSplineHeader:
    """
    Binary layout (little-endian, 16 bytes):
        version     u32
        num_points  u32
        _pad        8 bytes
    """

    _FMT: ClassVar[str] = _AI_HDR_FMT
    _SIZE: ClassVar[int] = _AI_HDR_SIZE

    version: int
    num_points: int

    @classmethod
    def read(cls, fp) -> AiSplineHeader:
        raw = fp.read(cls._SIZE)
        if len(raw) < cls._SIZE:
            raise EOFError("Truncated AiSplineHeader")
        version, num_points = struct.unpack(cls._FMT, raw)
        return cls(version=version, num_points=num_points)


@dataclass
class AiSplinePoint:
    """
    Binary layout (little-endian, 32 bytes):
        position  3 × f32   12 bytes
        length    f32        4 bytes   cumulative arc-length (m)
        id        u32        4 bytes
        _pad      12 bytes
    """

    _FMT: ClassVar[str] = _AI_PT_FMT
    _SIZE: ClassVar[int] = _AI_PT_SIZE

    position: Point3
    length: float
    point_id: int

    @classmethod
    def read(cls, fp) -> AiSplinePoint:
        raw = fp.read(cls._SIZE)
        if len(raw) < cls._SIZE:
            raise EOFError("Truncated AiSplinePoint")
        px, py, pz, length, pid = struct.unpack(cls._FMT, raw)
        return cls(position=Point3(px, py, pz), length=length, point_id=pid)


@dataclass
class AiSpline:
    """
    Complete parsed AI spline.

    Attributes
    ----------
    header          : AiSplineHeader
    points          : list[AiSplinePoint]   – centre reference line
    left_boundary   : list[Point3] | None
    right_boundary  : list[Point3] | None
    """

    header: AiSplineHeader
    points: list[AiSplinePoint]
    left_boundary: list[Point3] | None
    right_boundary: list[Point3] | None

    @property
    def centre_line(self) -> list[Point3]:
        return [p.position for p in self.points]

    @classmethod
    def from_file(cls, path: str | Path) -> AiSpline:
        path = Path(path)
        with path.open("rb") as fp:
            header = AiSplineHeader.read(fp)
            points = [AiSplinePoint.read(fp) for _ in range(header.num_points)]

            def _read_side(n: int) -> list[Point3] | None:
                pts: list[Point3] = []
                for _ in range(n):
                    raw = fp.read(_AI_SIDE_SIZE)
                    if len(raw) < _AI_SIDE_SIZE:
                        return None
                    x, y, z = struct.unpack(_AI_SIDE_FMT, raw)
                    pts.append(Point3(x, y, z))
                return pts

            left_pts = _read_side(header.num_points)
            right_pts = _read_side(header.num_points) if left_pts else None

        return cls(
            header=header,
            points=points,
            left_boundary=left_pts,
            right_boundary=right_pts,
        )


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def find_track_data_dir(
    ac_install: str | Path,
    track_name: str,
    layout: str | None = None,
) -> Path:
    """Resolve the ``data/`` directory for a track (with optional layout)."""
    base = Path(ac_install) / "content" / "tracks" / track_name
    data_dir = (base / layout / "data") if layout else (base / "data")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Track data directory not found: {data_dir}")
    return data_dir


def find_ai_dir(
    ac_install: str | Path,
    track_name: str,
    layout: str | None = None,
) -> Path:
    """Resolve the ``ai/`` directory for a track (with optional layout)."""
    base = Path(ac_install) / "content" / "tracks" / track_name
    ai_dir = (base / layout / "ai") if layout else (base / "ai")
    if not ai_dir.is_dir():
        raise FileNotFoundError(f"AI directory not found: {ai_dir}")
    return ai_dir
