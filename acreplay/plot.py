"""
ac_plot_racing_line.py
----------------------
Plots a driver's racing line from a parsed .acreplay file, overlaid on
track boundaries sourced from either:
  (a) the AC track's data/side_l.csv + data/side_r.csv, or
  (b) the embedded boundary arrays inside ai/fast_lane.ai.

Usage (command-line)
~~~~~~~~~~~~~~~~~~~~
    python ac_plot_racing_line.py replay.acreplay \
        --track-data /path/to/ac/content/tracks/ks_monza \
        --layout full              # omit for single-layout tracks
        --car 0                    # car index (default 0 = first car)
        --output racing_line.png   # omit to show interactively

Usage (programmatic)
~~~~~~~~~~~~~~~~~~~~
    from ac_plot_racing_line import plot_racing_line
    plot_racing_line(
        replay_path="lap.acreplay",
        track_data_dir="/path/to/track/data",
        ai_dir="/path/to/track/ai",
        car_index=0,
    )
"""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Literal

import matplotlib.collections as mc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from .replay import Replay, Vec3
from .track import (
    AiSpline,
    Point3,
    find_ai_dir,
    find_track_data_dir,
    load_boundary_csv,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _xz(points: list[Vec3] | list[Point3]) -> tuple[np.ndarray, np.ndarray]:
    """Project a list of 3-D points onto the XZ plane (top-down view)."""
    xs = np.array([p.x for p in points], dtype=np.float32)
    zs = np.array([p.z for p in points], dtype=np.float32)
    return xs, zs


def _speed_colour_segments(
    xs: np.ndarray,
    zs: np.ndarray,
    speeds: np.ndarray,
) -> mc.LineCollection:
    """
    Build a LineCollection whose colour encodes speed.

    Slow  →  blue (#2166ac)
    Mid   →  white
    Fast  →  red  (#d6604d)
    """
    points = np.column_stack([xs, zs])
    segments = np.stack([points[:-1], points[1:]], axis=1)

    # Normalise speed to [0, 1]
    vmin, vmax = speeds.min(), speeds.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "speed",
        ["#2166ac", "#f7f7f7", "#d6604d"],  # blue → white → red
    )
    colours = cmap(norm(speeds[:-1]))

    lc = mc.LineCollection(segments, colors=colours, linewidth=2.0, zorder=3)
    return lc, cmap, norm, vmin, vmax


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_racing_line(
    replay: Replay,
    *,
    track_data_dir: str | Path | None = None,
    ai_dir: str | Path | None = None,
    car_index: int = 0,
    colour_by: Literal["speed", "brake", "throttle"] = "speed",
    output_path: str | Path | None = None,
    dpi: int = 150,
    figsize: tuple[float, float] = (12, 10),
) -> None:
    """
    Parse the replay and produce a top-down racing-line plot.

    Parameters
    ----------
    replay_path     : path to the .acreplay file
    track_data_dir  : path to the track's ``data/`` folder
                      (needed for side_l.csv / side_r.csv)
    ai_dir          : path to the track's ``ai/`` folder
                      (needed for fast_lane.ai)
    car_index       : which car to plot (0-based)
    colour_by       : metric to encode as line colour:
                      "speed" | "brake" | "throttle"
    output_path     : save to file when given; show interactively otherwise
    dpi             : output resolution when saving
    figsize         : figure size in inches
    """
    # -- Parse replay -------------------------------------------------------
    n_cars = replay.header.num_cars
    if car_index >= n_cars:
        raise IndexError(
            f"car_index {car_index} out of range; replay has {n_cars} car(s)"
        )

    car_meta = replay.cars[car_index]
    positions = replay.positions_for_car(car_index)
    pos_xs, pos_zs = _xz(positions)

    # Per-frame scalar for colouring
    if colour_by == "speed":
        scalars = np.array(
            [replay.frames[i][car_index].speed_kmh for i in range(len(replay.frames))],
            dtype=np.float32,
        )
        cbar_label = "Speed (km/h)"
    elif colour_by == "brake":
        scalars = np.array(
            [replay.frames[i][car_index].brake for i in range(len(replay.frames))],
            dtype=np.float32,
        )
        cbar_label = "Brake pressure (0–1)"
    elif colour_by == "throttle":
        scalars = np.array(
            [replay.frames[i][car_index].gas for i in range(len(replay.frames))],
            dtype=np.float32,
        )
        cbar_label = "Throttle (0–1)"
    else:
        raise ValueError(f"Unknown colour_by value: {colour_by!r}")  # pyright: ignore[reportUnreachable]

    # -- Load boundary data -------------------------------------------------
    left_pts: list[Point3] | None = None
    right_pts: list[Point3] | None = None
    ai_centre: list[Point3] | None = None

    if track_data_dir is not None:
        track_data_dir = Path(track_data_dir)
        left_csv = track_data_dir / "side_l.csv"
        right_csv = track_data_dir / "side_r.csv"
        if left_csv.exists():
            left_pts = load_boundary_csv(left_csv)
        if right_csv.exists():
            right_pts = load_boundary_csv(right_csv)

    if ai_dir is not None:
        ai_file = Path(ai_dir) / "fast_lane.ai"
        if ai_file.exists():
            spline = AiSpline.from_file(ai_file)
            ai_centre = spline.centre_line
            # Use embedded boundaries if CSV ones aren't available
            if left_pts is None and spline.left_boundary:
                left_pts = spline.left_boundary
            if right_pts is None and spline.right_boundary:
                right_pts = spline.right_boundary

    # -- Build figure -------------------------------------------------------
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Grid
    ax.grid(color="#2e2e4e", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a3a5e")

    # Track boundaries
    boundary_kw = {"color": "#888899", "linewidth": 1.2, "linestyle": "-", "zorder": 1}
    if left_pts:
        lx, lz = _xz(left_pts)
        ax.plot(lx, lz, **boundary_kw, label="Track boundary")
    if right_pts:
        rx, rz = _xz(right_pts)
        ax.plot(rx, rz, **boundary_kw)

    # Fill between boundaries (track surface tint)
    if left_pts and right_pts:
        # Combine boundaries into a closed polygon for filling.
        # The two boundary arrays run in the same direction around the circuit,
        # so reversing one and concatenating gives a closed loop.
        lx, lz = _xz(left_pts)
        rx, rz = _xz(right_pts)
        poly_x = np.concatenate([lx, rx[::-1]])
        poly_z = np.concatenate([lz, rz[::-1]])
        ax.fill(poly_x, poly_z, color="#252540", alpha=0.6, zorder=0)

    # AI reference line
    if ai_centre:
        ax_ref, az_ref = _xz(ai_centre)
        ax.plot(
            ax_ref,
            az_ref,
            color="#aaaacc",
            linewidth=0.8,
            linestyle="--",
            alpha=0.45,
            zorder=2,
            label="AI reference line",
        )

    # Racing line (speed-coloured)
    lc, cmap, norm, _vmin, _vmax = _speed_colour_segments(pos_xs, pos_zs, scalars)
    ax.add_collection(lc)

    # Start/finish marker
    if len(positions) > 0:
        ax.scatter(
            pos_xs[0],
            pos_zs[0],
            s=80,
            color="#00ff88",
            zorder=5,
            marker="o",
            label="Start",
        )

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, color="#ccccdd")
    cbar.ax.yaxis.set_tick_params(color="#ccccdd")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ccccdd")
    cbar.outline.set_edgecolor("#3a3a5e")

    # Axis formatting
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(colors="#888899")
    ax.xaxis.label.set_color("#888899")
    ax.yaxis.label.set_color("#888899")
    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Z (m)")

    # Title
    driver = car_meta.driver_name or f"Car {car_index}"
    car = car_meta.car_model or "unknown"
    track = replay.header.track_name or "unknown track"
    n_frames = replay.header.num_frames
    dt_s = replay.header.recording_interval / 1000.0
    duration = n_frames * dt_s
    ax.set_title(
        f"{driver}  ·  {car}  ·  {track}\n"
        f"{n_frames} frames  ·  Δt = {replay.header.recording_interval:.1f} ms"
        f"  ·  duration ≈ {duration:.1f} s",
        color="#ccccdd",
        fontsize=10,
        pad=10,
    )

    ax.legend(
        facecolor="#252540",
        edgecolor="#3a3a5e",
        labelcolor="#ccccdd",
        fontsize=8,
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(
            output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor()
        )
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot an Assetto Corsa racing line from a .acreplay file."
    )
    p.add_argument("replay", help="Path to the .acreplay file")
    p.add_argument(
        "--track-data",
        metavar="DIR",
        help="Path to the track's data/ directory (for side_l/r.csv)",
    )
    p.add_argument(
        "--ai-dir",
        metavar="DIR",
        help="Path to the track's ai/ directory (for fast_lane.ai)",
    )
    p.add_argument(
        "--ac-install",
        metavar="DIR",
        help=(
            "AC installation root.  Combined with --track-name / --layout "
            "to locate track data automatically."
        ),
    )
    p.add_argument(
        "--track-name", metavar="NAME", help="Track folder name, e.g. ks_monza"
    )
    p.add_argument(
        "--layout",
        metavar="NAME",
        default=None,
        help="Layout sub-folder (omit for single-layout tracks)",
    )
    p.add_argument(
        "--car",
        type=int,
        default=0,
        metavar="N",
        help="Car index to plot (default: 0)",
    )
    p.add_argument(
        "--colour-by",
        default="speed",
        choices=["speed", "brake", "throttle"],
        help="Metric to encode as line colour (default: speed)",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        help="Save plot to this file instead of showing interactively",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    track_data_dir: Path | None = None
    ai_dir: Path | None = None

    replay = Replay.from_file(args.replay)

    if args.track_data:
        track_data_dir = Path(args.track_data)
    elif args.ac_install and args.track_name:
        track_data_dir = find_track_data_dir(
            args.ac_install, args.track_name, args.layout
        )

    if args.ai_dir:
        ai_dir = Path(args.ai_dir)
    elif args.ac_install and args.track_name:
        with contextlib.suppress(FileNotFoundError):
            ai_dir = find_ai_dir(args.ac_install, args.track_name, args.layout)

    plot_racing_line(
        replay=replay,
        track_data_dir=track_data_dir,
        ai_dir=ai_dir,
        car_index=args.car,
        colour_by=args.colour_by,
        output_path=args.output,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
