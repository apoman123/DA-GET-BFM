"""URDF collision geometry as point sets, for the lowest point of a posed robot.

Kinematic motion data (retargeted motion capture) often gives the root height only
relative to the first frame. Putting such a motion on the ground needs the lowest point
of the whole body in every frame, not only of the feet: a clip that starts lying or
crawling rests on the back, hands or knees.

* ``collision_points`` turns every ``<collision>`` element of a URDF into points in its
  link frame: sphere centres with their radius, cylinder rims, box corners, and the
  convex-hull vertices of a mesh read from STL files.
* ``lowest_point_heights`` gives the world height of the lowest of them per frame. The
  lowest point of a convex shape along a direction is one of its hull vertices, so for
  meshes and boxes the heights are exact; a cylinder is sampled at ``CYLINDER_SAMPLES``
  rim points (error below r (1 - cos(pi / CYLINDER_SAMPLES))).
* ``ground_offset`` turns those heights into the height to subtract from the root.

The envelope in ``ground_offset`` (rolling minimum, then moving average) is this
repository's choice; ``docs/data.md#putting-kinematic-motions-on-the-ground`` has the
measurements behind its default window.
"""

from __future__ import annotations

import functools
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

CYLINDER_SAMPLES = 32


def read_stl(path: str | Path) -> np.ndarray:
    """Return the triangle vertices of a binary or ASCII STL file, ``[3 N, 3]``."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        count = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * count:  # binary (its header may still start with "solid")
            records = np.frombuffer(data, dtype=np.dtype([("normal", "<f4", 3), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]), count=count, offset=84)
            return records["vertices"].reshape(-1, 3).astype(np.float64)
    vertices = [line.split()[1:4] for line in data.decode("ascii", errors="replace").splitlines() if line.strip().startswith("vertex")]
    if not vertices:
        raise ValueError(f"{path}: not an STL file")
    return np.asarray(vertices, dtype=np.float64)


def _hull(points: np.ndarray) -> np.ndarray:
    """Return the convex-hull vertices of a point cloud (all unique points if it is flat)."""
    from scipy.spatial import ConvexHull, QhullError

    unique = np.unique(points, axis=0)
    try:
        return unique[ConvexHull(unique).vertices]
    except (QhullError, ValueError):
        return unique


def _origin(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    """Return the rotation and translation of a URDF ``<origin>`` (fixed-axis roll, pitch, yaw)."""
    xyz = np.zeros(3)
    rpy = np.zeros(3)
    if element is not None:
        xyz = np.array([float(v) for v in element.get("xyz", "0 0 0").split()])
        rpy = np.array([float(v) for v in element.get("rpy", "0 0 0").split()])
    (cr, cp, cy), (sr, sp, sy) = np.cos(rpy), np.sin(rpy)
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    return rotation, xyz


def _shape_points(geometry: ET.Element, folder: Path) -> tuple[np.ndarray, float]:
    """Return points of one collision shape in its own frame and the radius to subtract."""
    shape = next(iter(geometry))
    if shape.tag == "sphere":
        return np.zeros((1, 3)), float(shape.get("radius"))
    if shape.tag == "box":
        half = np.array([float(v) for v in shape.get("size").split()]) / 2
        return np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half, 0.0
    if shape.tag == "cylinder":
        radius, length = float(shape.get("radius")), float(shape.get("length"))
        angles = np.arange(CYLINDER_SAMPLES) * 2 * np.pi / CYLINDER_SAMPLES
        rim = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.zeros_like(angles)], axis=1)
        return np.concatenate([rim + [0, 0, -length / 2], rim + [0, 0, length / 2]]), 0.0
    if shape.tag == "mesh":
        name = shape.get("filename", "")
        if "://" in name:
            raise ValueError(f"{name}: only mesh paths relative to the URDF are supported")
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f"{path}: collision mesh missing; run scripts/download_urdfs.py (it fetches the G1 meshes)")
        scale = np.array([float(v) for v in shape.get("scale", "1 1 1").split()])
        return _hull(read_stl(path) * scale), 0.0
    raise ValueError(f"unsupported collision geometry <{shape.tag}>")


@functools.lru_cache(maxsize=8)
def _collision_points(urdf: str, links: tuple[str, ...] | None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    root = ET.parse(urdf).getroot()
    folder = Path(urdf).parent
    output = {}
    for link in root.findall("link"):
        name = link.get("name")
        if links is not None and name not in links:
            continue
        points, radii = [], []
        for collision in link.findall("collision"):
            rotation, translation = _origin(collision.find("origin"))
            shape_points, radius = _shape_points(collision.find("geometry"), folder)
            points.append(shape_points @ rotation.T + translation)
            radii.append(np.full(len(shape_points), radius))
        if points:
            output[name] = (np.concatenate(points), np.concatenate(radii))
    if links is not None and set(links) - set(output):
        raise ValueError(f"links without collision geometry: {sorted(set(links) - set(output))}")
    return output


def collision_points(urdf: str | Path, links: list[str] | None = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return, for every link with collision geometry, its points in the link frame and their radii.

    Args:
        urdf (str | Path): URDF file; mesh paths are relative to its folder.
        links (list[str] | None): Only these links (all by default).

    Returns:
        dict: ``{link: (points [K, 3], radii [K])}``. The lowest point of the link along a
            direction ``d`` is ``min(points @ d - radii)``.
    """
    return _collision_points(str(Path(urdf).resolve()), tuple(links) if links is not None else None)


def lowest_point_heights(
    graph,
    geometry: dict[str, tuple[np.ndarray, np.ndarray]],
    joints: np.ndarray,
    position: np.ndarray,
    orientation: np.ndarray,
) -> np.ndarray:
    """Return the world height of the lowest collision point in each frame, ``[T]``.

    Args:
        graph (EmbodimentGraph): Robot graph whose joint order matches ``joints``.
        geometry (dict): ``collision_points`` output (the links to consider).
        joints (np.ndarray): ``[T, J]`` joint positions.
        position (np.ndarray): ``[T, 3]`` root position in the world (z up).
        orientation (np.ndarray): ``[T, 4]`` root quaternion, x, y, z, w.
    """
    import torch

    from model.conversion.common import rotation_from_xyzw

    links = list(geometry)
    poses = graph.link_poses(torch.as_tensor(joints, dtype=torch.float64), links).numpy()  # [T, L, 4, 4] in the root frame
    up = rotation_from_xyzw(orientation)[:, 2, :]  # world z axis in the root frame, [T, 3]
    lowest = np.full(len(joints), np.inf)
    for index, link in enumerate(links):
        points, radii = geometry[link]
        axis = np.einsum("tj,tjk->tk", up, poses[:, index, :3, :3])  # world z in the link frame
        heights = axis @ points.T + np.einsum("tj,tj->t", up, poses[:, index, :3, 3])[:, None] - radii
        lowest = np.minimum(lowest, heights.min(axis=1))
    return position[:, 2] + lowest


def ground_offset(heights: np.ndarray, rate: float, window: float = 1.0, percentile: float = 5.0) -> np.ndarray:
    """Return the height to subtract from the root so that the motion rests on the ground, ``[T]``.

    With ``window > 0`` the offset follows the lower envelope of the per-frame lowest point:
    a centred rolling minimum over ``window`` seconds, then a centred moving average over
    the same length. Retargeted root heights drift by several centimetres within a clip
    (``docs/data.md``), which a constant cannot remove. Every frame lies in each window
    that is averaged into its offset, so the offset never exceeds the frame's own lowest
    point: no frame goes below the ground. Motion faster than the window, such as steps
    and jumps whose flight phase is shorter than ``window``, is kept. With ``window <= 0``
    the offset is the constant ``percentile`` of the heights.

    Args:
        heights (np.ndarray): ``[T]`` lowest point per frame (``lowest_point_heights``).
        rate (float): Frame rate in Hz.
        window (float): Envelope length in seconds, or 0 for a constant offset.
        percentile (float): Percentile put on the ground by the constant offset.
    """
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    heights = np.asarray(heights, dtype=np.float64)
    if window <= 0:
        return np.full_like(heights, np.percentile(heights, percentile))
    size = max(1, int(round(window * rate)) | 1)  # odd, so the windows are centred
    return uniform_filter1d(minimum_filter1d(heights, size, mode="nearest"), size, mode="nearest")
