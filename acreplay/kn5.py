"""
kn5.py
------
Parser for Assetto Corsa KN5 3-D model files, and tools for extracting
track surface boundaries from them.

KN5 format (reverse-engineered from https://github.com/RaduMC/kn5-converter):

    Magic:   b"sc6969"   (6 bytes, no null terminator)
    version: i32

    [If version > 5]
        unknown_int: i32   (observed value: 673425)

    Textures section:
        count: i32
        [count × texture_entry]
            type:  i32
            name:  lstring
            size:  i32
            data:  size bytes   (raw texture file, skipped here)

    Materials section:
        count: i32
        [count × material_entry]
            name:       lstring
            shader:     lstring
            _short:     i16
            [if version > 4]
                _zero:  i32
            prop_count: i32
            [prop_count × shader_property]
                name:   lstring
                value:  f32
                _pad:   36 bytes
            tex_count:  i32
            [tex_count × texture_ref]
                sample_name: lstring
                sample_slot: i32
                tex_name:    lstring

    Nodes: (one root call → depth-first recursion)
        type:           i32   (1 = dummy, 2 = static mesh, 3 = animated mesh)
        name:           lstring
        children_count: i32
        active:         byte

        [type 1 – dummy node]
            transform:  16 × f32   (4 × 4, row-major; row 3 = translation)

        [type 2 – static mesh]
            _flags:      3 bytes
            vertex_count: i32
            [vertex_count × 44-byte vertex]
                position: 3 × f32  (12 bytes)
                normal:   3 × f32  (12 bytes)
                uv:       2 × f32  ( 8 bytes)
                tangent:  3 × f32  (12 bytes, skipped)
            index_count: i32
            indices:     index_count × u16
            material_id: i32
            _pad:        29 bytes

        [type 3 – animated mesh]
            _flags:      3 bytes
            bone_count:  i32
            [bone_count]
                name:    lstring
                matrix:  64 bytes   (4 × 4, skipped)
            vertex_count: i32
            [vertex_count × 76-byte vertex]
                position: 3 × f32  (12 bytes)
                normal:   3 × f32  (12 bytes)
                uv:       2 × f32  ( 8 bytes)
                _pad:     44 bytes  (tangents + bone weights, skipped)
            index_count: i32
            indices:     index_count × u16
            material_id: i32
            _pad:        12 bytes

All integers are little-endian.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KN5_MAGIC = b"sc6969"
_IDENTITY: np.ndarray = np.eye(4, dtype=np.float32)

# Per-vertex byte stride for each mesh type
_STATIC_VERTEX_STRIDE = 44    # pos(12) + normal(12) + uv(8) + tangent(12)
_ANIMATED_VERTEX_STRIDE = 76  # pos(12) + normal(12) + uv(8) + weights+tangents(44)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read_exactly(fp: BinaryIO, n: int) -> bytes:
    data = fp.read(n)
    if len(data) != n:
        raise EOFError(
            f"Expected {n} bytes but got {len(data)} "
            f"(file offset ~{fp.tell()})"
        )
    return data


def _read_lstring(fp: BinaryIO) -> str:
    """4-byte length-prefixed UTF-8 string."""
    (length,) = struct.unpack("<I", _read_exactly(fp, 4))
    return _read_exactly(fp, length).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Material
# ---------------------------------------------------------------------------


@dataclass
class Kn5Material:
    """
    A KN5 material (shader + name).

    Only the name and shader identifier are stored; shader-property values
    and texture references are not needed for geometry classification.
    """

    name: str
    shader: str


# ---------------------------------------------------------------------------
# Mesh
# ---------------------------------------------------------------------------


@dataclass
class Kn5Mesh:
    """
    A single static or animated mesh extracted from a KN5 node tree.

    Vertex positions are stored in world space — the accumulated
    transformation matrix from the node hierarchy has already been applied.

    Attributes
    ----------
    name        : node name as written in the KN5 file
    material_id : index into :attr:`Kn5Model.materials`
    positions   : float32 array of shape ``(N, 3)``, world-space XYZ
    indices     : uint32 array of shape ``(M,)``; every 3 values form one triangle
    """

    name: str
    material_id: int
    positions: np.ndarray   # (N, 3) float32
    indices: np.ndarray     # (M,)   uint32

    @property
    def triangles(self) -> np.ndarray:
        """World-space triangle vertices, shape ``(T, 3, 3)``."""
        n = (len(self.indices) // 3) * 3
        idx = self.indices[:n].reshape(-1, 3)
        return self.positions[idx]

    @property
    def triangles_xz(self) -> np.ndarray:
        """XZ projections of each triangle's vertices, shape ``(T, 3, 2)``."""
        return self.triangles[:, :, [0, 2]]

    @property
    def triangles_y_centroid(self) -> np.ndarray:
        """Mean Y coordinate of each triangle, shape ``(T,)``."""
        return self.triangles[:, :, 1].mean(axis=1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class Kn5Model:
    """
    A fully parsed KN5 model file.

    Attributes
    ----------
    version   : file-format version integer
    materials : ordered list of :class:`Kn5Material`; indices match ``material_id``
    meshes    : list of :class:`Kn5Mesh` with world-space vertex positions
    """

    version: int
    materials: list[Kn5Material]
    meshes: list[Kn5Mesh]

    @classmethod
    def from_file(cls, path: str | Path) -> Kn5Model:
        """Parse a ``.kn5`` file and return a :class:`Kn5Model`."""
        path = Path(path)
        with path.open("rb") as fp:
            return _read_kn5(fp)


# ---------------------------------------------------------------------------
# Path helper
# ---------------------------------------------------------------------------


def find_track_kn5(
    ac_install: str | Path,
    track_name: str,
) -> Path:
    """
    Locate the main KN5 model file for a track.

    Looks for ``<track_name>.kn5`` in
    ``<ac_install>/content/tracks/<track_name>/``.

    Parameters
    ----------
    ac_install : root directory of the Assetto Corsa installation
    track_name : track folder name (e.g. ``"ks_monza"``)

    Raises
    ------
    FileNotFoundError
        If the expected KN5 file is not present.
    """
    kn5 = (
        Path(ac_install) / "content" / "tracks" / track_name / f"{track_name}.kn5"
    )
    if not kn5.is_file():
        raise FileNotFoundError(f"Track KN5 not found: {kn5}")
    return kn5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _in_triangles_xz(
    px: float,
    pz: float,
    triangles_xz: np.ndarray,
) -> np.ndarray:
    """
    Return a boolean mask indicating which triangles contain point ``(px, pz)``.

    Uses the cross-product sign method for a robust inside/on-boundary test.

    Parameters
    ----------
    px, pz        : query point in the XZ plane
    triangles_xz  : array of shape ``(N, 3, 2)``

    Returns
    -------
    Boolean ndarray of shape ``(N,)``.
    """
    v0 = triangles_xz[:, 0, :]   # (N, 2)
    v1 = triangles_xz[:, 1, :]
    v2 = triangles_xz[:, 2, :]

    d1 = (
        (px - v1[:, 0]) * (v0[:, 1] - v1[:, 1])
        - (v0[:, 0] - v1[:, 0]) * (pz - v1[:, 1])
    )
    d2 = (
        (px - v2[:, 0]) * (v1[:, 1] - v2[:, 1])
        - (v1[:, 0] - v2[:, 0]) * (pz - v2[:, 1])
    )
    d3 = (
        (px - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (pz - v0[:, 1])
    )

    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)

    # A point is inside (or on the boundary of) the triangle when the
    # three signed areas are not of mixed sign.
    return ~(has_neg & has_pos)


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------


def find_road_triangles(
    model: Kn5Model,
    start_xz: tuple[float, float],
) -> tuple[int, np.ndarray] | None:
    """
    Identify the road surface material at the given starting position and
    collect all triangles that share it.

    Algorithm
    ---------
    1. For every mesh in the model, find the triangles (projected to the XZ
       plane) that contain *start_xz*.
    2. Among those candidates, select the one with the **highest Y centroid**
       (i.e. the topmost surface in AC's Y-up coordinate system).
    3. Record its ``material_id``.
    4. Return that material ID together with every triangle (from every mesh)
       that uses the same material.

    Parameters
    ----------
    model     : parsed :class:`Kn5Model`
    start_xz  : ``(x, z)`` world-space position of the track's starting point

    Returns
    -------
    ``(material_id, triangles_xz)`` where *triangles_xz* is an
    ``(N, 3, 2)`` float32 array, or ``None`` if *start_xz* is not inside
    any triangle in the model.
    """
    px, pz = start_xz
    best_y = -np.inf
    best_mat_id = -1

    for mesh in model.meshes:
        tris_xz = mesh.triangles_xz     # (T, 3, 2)
        if len(tris_xz) == 0:
            continue

        mask = _in_triangles_xz(px, pz, tris_xz)
        if not mask.any():
            continue

        y_cents = mesh.triangles_y_centroid   # (T,)
        max_y = float(y_cents[mask].max())
        if max_y > best_y:
            best_y = max_y
            best_mat_id = mesh.material_id

    if best_mat_id < 0:
        return None

    # Collect all triangles that belong to the identified material
    pieces: list[np.ndarray] = [
        mesh.triangles_xz
        for mesh in model.meshes
        if mesh.material_id == best_mat_id and len(mesh.indices) > 0
    ]
    if not pieces:
        return None

    return best_mat_id, np.concatenate(pieces, axis=0)


def find_patch_boundary(
    triangles_xz: np.ndarray,
) -> list[list[tuple[float, float]]]:
    """
    Compute the boundary of a triangulated surface patch using *shapely*.

    The triangles are unioned into a single (possibly multi-part) polygon
    and its boundary rings are extracted.

    Parameters
    ----------
    triangles_xz : array of shape ``(N, 3, 2)`` — XZ coordinates of each
                   triangle's three vertices (as produced by
                   :func:`find_road_triangles`)

    Returns
    -------
    A list of **boundary rings**.  Each ring is a list of ``(x, z)`` float
    pairs forming a closed loop (the last point equals the first point).
    The first ring of each polygon component is its exterior outline;
    subsequent rings (if any) are interior holes.
    """
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    if len(triangles_xz) == 0:
        return []

    polys = [Polygon(tri) for tri in triangles_xz]
    merged = unary_union(polys)

    rings: list[list[tuple[float, float]]] = []

    def _extract(geom: Polygon) -> None:
        t = geom.geom_type
        if t == "Polygon":
            rings.append(list(geom.exterior.coords))
            for interior in geom.interiors:
                rings.append(list(interior.coords))
        elif t in ("MultiPolygon", "GeometryCollection"):
            for part in geom.geoms:
                _extract(part)

    _extract(merged)
    return rings


# ---------------------------------------------------------------------------
# Internal parser
# ---------------------------------------------------------------------------


def _read_kn5(fp: BinaryIO) -> Kn5Model:
    magic = _read_exactly(fp, 6)
    if magic != _KN5_MAGIC:
        raise ValueError(f"Not a KN5 file (magic={magic!r})")

    (version,) = struct.unpack("<i", _read_exactly(fp, 4))
    if version > 5:
        fp.read(4)  # unknown i32 (673425)

    # ------------------------------------------------------------------
    # Textures – skip raw data, we only need to advance the file pointer
    # ------------------------------------------------------------------
    (tex_count,) = struct.unpack("<i", _read_exactly(fp, 4))
    for _ in range(tex_count):
        fp.read(4)           # type i32
        _read_lstring(fp)    # texture name
        (tex_size,) = struct.unpack("<i", _read_exactly(fp, 4))
        fp.read(tex_size)    # raw bytes of embedded texture

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    (mat_count,) = struct.unpack("<i", _read_exactly(fp, 4))
    materials: list[Kn5Material] = []
    for _ in range(mat_count):
        name = _read_lstring(fp)
        shader = _read_lstring(fp)
        fp.read(2)           # i16 (alpha blend mode / similar flag)
        if version > 4:
            fp.read(4)       # i32 zero

        (prop_count,) = struct.unpack("<i", _read_exactly(fp, 4))
        for _ in range(prop_count):
            _read_lstring(fp)    # property name
            fp.read(4)           # f32 value
            fp.read(36)          # 9 × f32 padding (vec4 + padding)

        (tex_ref_count,) = struct.unpack("<i", _read_exactly(fp, 4))
        for _ in range(tex_ref_count):
            _read_lstring(fp)    # sampler name (e.g. "txDiffuse")
            fp.read(4)           # i32 slot index
            _read_lstring(fp)    # texture filename

        materials.append(Kn5Material(name=name, shader=shader))

    # ------------------------------------------------------------------
    # Node tree (recursive depth-first)
    # ------------------------------------------------------------------
    meshes: list[Kn5Mesh] = []
    _read_node(fp, version, meshes, _IDENTITY)

    return Kn5Model(version=version, materials=materials, meshes=meshes)


def _read_node(
    fp: BinaryIO,
    version: int,
    meshes: list[Kn5Mesh],
    parent_hmatrix: np.ndarray,
) -> None:
    """
    Read one node from *fp* (depth-first) and append any mesh data found
    to *meshes*.

    Transformation accumulation
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Dummy nodes (type 1) carry a 4 × 4 local-to-parent transform matrix
    *tmatrix*.  The accumulated world-space matrix is::

        hmatrix = tmatrix @ parent_hmatrix

    Mesh nodes (types 2 and 3) have no own matrix; they inherit
    ``parent_hmatrix`` directly.  Vertex positions are transformed to world
    space via ``V_world = V_local_homogeneous @ hmatrix``.
    """
    (node_type,) = struct.unpack("<i", _read_exactly(fp, 4))
    name = _read_lstring(fp)
    (children_count,) = struct.unpack("<i", _read_exactly(fp, 4))
    fp.read(1)   # active byte

    if node_type == 1:
        # Dummy node: read 4×4 row-major transform matrix
        raw = _read_exactly(fp, 64)
        tmatrix = np.frombuffer(raw, dtype=np.float32).reshape(4, 4).copy()
        hmatrix = tmatrix @ parent_hmatrix

    elif node_type in (2, 3):
        # Static or animated mesh: no own matrix
        hmatrix = parent_hmatrix
        fp.read(3)   # three flag bytes

        if node_type == 3:
            # Animated: skip bone data (name + 4×4 inverse-bind matrix each)
            (bone_count,) = struct.unpack("<i", _read_exactly(fp, 4))
            for _ in range(bone_count):
                _read_lstring(fp)
                fp.read(64)   # bone inverse-bind matrix

        (vertex_count,) = struct.unpack("<i", _read_exactly(fp, 4))
        stride = _ANIMATED_VERTEX_STRIDE if node_type == 3 else _STATIC_VERTEX_STRIDE

        # Read all vertex data at once and extract positions (first 3 floats)
        raw_verts = _read_exactly(fp, vertex_count * stride)
        all_floats = np.frombuffer(raw_verts, dtype=np.float32).reshape(
            vertex_count, stride // 4
        )
        local_pos = all_floats[:, :3].copy()   # (N, 3) local-space XYZ

        # Transform to world space: [x, y, z, 1] @ hmatrix → first 3 cols
        ones = np.ones((vertex_count, 1), dtype=np.float32)
        v_hom = np.concatenate([local_pos, ones], axis=1)   # (N, 4)
        world_pos = (v_hom @ hmatrix)[:, :3]                 # (N, 3)

        # Read triangle indices
        (index_count,) = struct.unpack("<i", _read_exactly(fp, 4))
        raw_idx = _read_exactly(fp, index_count * 2)
        indices = np.frombuffer(raw_idx, dtype=np.uint16).astype(np.uint32)

        (material_id,) = struct.unpack("<i", _read_exactly(fp, 4))
        fp.read(29 if node_type == 2 else 12)   # trailing padding

        meshes.append(
            Kn5Mesh(
                name=name,
                material_id=material_id,
                positions=world_pos,
                indices=indices,
            )
        )

    else:
        raise ValueError(
            f"Unknown KN5 node type {node_type!r} for node {name!r}"
        )

    # Recurse into children
    for _ in range(children_count):
        _read_node(fp, version, meshes, hmatrix)
