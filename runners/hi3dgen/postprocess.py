# SPDX-License-Identifier: MIT
"""Post-processing for the generated mesh. **Follow upstream; add no judgement of our own.**

**Hi3DGen's upstream (Stable3DGen) has no post-processing.** When it was derived
from TRELLIS the dependencies on `kaolin`, `nvdiffrast` and `flexicubes` were
removed, and the equivalent of `postprocess_mesh` went with them (the upstream
README says they were dropped to make commercial use possible). Meshing is
`skimage.measure` marching cubes, returned as-is.

**The multi-view rasterization plus min-cut that TRELLIS performs therefore does
not happen here**, so that nothing absent from upstream is introduced.

## What is added on top of upstream (**exactly two, each fixing a measured defect**)

### 1. Fill small holes (`pymeshfix.fill_small_boundaries`)

Hi3DGen output is **never watertight**. Measuring the 198 boundary edges on the
sample `i2i_00038_.png` (2026-09-01) showed **four loops all lying on the same
plane at z = 39.9 mm**, a flat opening of near-zero thickness. The model is
80.0 mm tall (z from -40.1 to 39.9), so it is **clipped at the top of the
generation volume** rather than randomly torn. Upstream frames the subject at
`size x 1.2`, which a runner cannot change from outside.

The opening is therefore closed with **the same `pymeshfix` call upstream
TRELLIS uses for the same purpose**. `forge`'s `repair_manifold` was tried first
and could not close it (it returned a warning).

### 2. Drop free-floating debris by size and thinness

**Free-floating parts are dropped by size and by thinness**
(`HI3DGEN_DROP_SMALL_PARTS` and `HI3DGEN_DROP_THIN_PARTS` in `.env`).

Measured on the sample `i2i_00038_.png` (2026-09-01): of the 968 components and
33,056 faces outside the body, **941 components and 30,048 faces (90.9 %) were
floating outside it**. That is a real defect for printing.
**Always record how much was dropped.**
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh

from . import config


def fill_small_holes(
    mesh: trimesh.Trimesh,
    max_nbe: int,
    progress: Callable[[str, str], None] | None = None,
    stats: CleanStats | None = None,
) -> trimesh.Trimesh:
    """Close small boundary loops (**the same `pymeshfix` call upstream TRELLIS makes**).

    Args:
        mesh: The mesh to treat.
        max_nbe: Maximum number of edges in a boundary loop to fill. 0 or less
            does nothing.
        progress: Where to report the stage.
        stats: Where to record what happened.

    Returns:
        The mesh with holes filled. **Returns the input unchanged on failure**,
        because a post-processing failure must not fail the generation.
    """
    if max_nbe <= 0:
        return mesh
    before = len(mesh.faces)
    try:
        from pymeshfix import _meshfix

        if progress is not None:
            progress("fill_holes", f"filling small holes (up to {max_nbe} edges)")
        fixer = _meshfix.PyTMesh()
        fixer.load_array(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
        )
        fixer.fill_small_boundaries(nbe=max_nbe, refine=True)
        vertices, faces = fixer.return_arrays()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception as exc:  # noqa: BLE001 - never fail generation over post-processing
        message = f"hole filling failed (mesh returned unchanged): {type(exc).__name__}: {exc}"
        print(f"[postprocess] {message}", file=sys.stderr)
        if stats is not None:
            stats.warnings.append(message)
        return mesh
    if stats is not None:
        stats.filled_faces = len(mesh.faces) - before
    return mesh


@dataclass
class CleanStats:
    """What post-processing changed, and by how much. **Numbers so nothing vanishes silently.**"""

    faces_before: int = 0
    faces_after: int = 0
    filled_faces: int = 0
    parts_before: int = 0
    parts_after: int = 0
    dropped_parts: int = 0
    dropped_faces: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """The shape that goes straight into `metrics`."""
        return {
            "faces_before": self.faces_before,
            "faces_after": self.faces_after,
            "filled_faces": self.filled_faces,
            "parts_before": self.parts_before,
            "parts_after": self.parts_after,
            "dropped_parts": self.dropped_parts,
            "dropped_faces": self.dropped_faces,
            "warnings": self.warnings,
        }


def drop_small_parts(
    mesh: trimesh.Trimesh,
    min_ratio: float,
    min_thick_ratio: float = 0.0,
    progress: Callable[[str, str], None] | None = None,
    stats: CleanStats | None = None,
) -> trimesh.Trimesh:
    """**Drop free-floating debris by size and thinness** (added on top of upstream).

    Components are judged by their axis-aligned bounding box against the model's
    longest side, and only components passing **both** tests are kept:

    1. **Longest extent at least `min_ratio`** (drops small debris). Spatial size
       rather than face count separates finely tessellated crumbs from genuine
       parts that merely have few faces.
    2. **Shortest extent at least `min_thick_ratio`** (drops flakes). Paper-thin
       flakes hovering about 1 % off the surface and lying parallel to it are
       **11-29 % long, so test 1 alone lets them straight through** (they render
       as dark speckles and tabs on the surface).

    Measured on the sample `i2i_00038_.png` (2026-09-02): the remaining flakes
    were **0.1-1.4 % thick** and genuine parts (arms, panels) **11.8 % or more**,
    so 2 % separates them with an order of magnitude to spare. A tilted flake
    would measure thicker in an axis-aligned box, but every flake measured lay
    parallel to the surface and close to axis-aligned, so it did not matter.

    Args:
        mesh: The mesh to treat.
        min_ratio: Minimum size to keep, relative to the model's longest side.
            0 or less skips the test.
        min_thick_ratio: Minimum thickness to keep, on the same scale. 0 or less
            skips the test.
        progress: Where to report the stage.
        stats: Where to record what happened.

    Returns:
        The mesh without the debris. **The largest component is always kept.**
    """
    if min_ratio <= 0 and min_thick_ratio <= 0:
        return mesh
    parts = mesh.split(only_watertight=False)
    if stats is not None:
        stats.parts_before = len(parts)
    if len(parts) <= 1:
        if stats is not None:
            stats.parts_after = len(parts)
        return mesh

    whole = max(float(np.max(mesh.bounding_box.extents)), 1e-12)
    face_counts = np.array([len(p.faces) for p in parts])
    keep = np.ones(len(parts), dtype=bool)
    if min_ratio > 0:
        sizes = np.array([float(np.max(p.bounding_box.extents)) for p in parts])
        keep &= sizes / whole >= min_ratio
    if min_thick_ratio > 0:
        thicks = np.array([float(np.min(p.bounding_box.extents)) for p in parts])
        keep &= thicks / whole >= min_thick_ratio
    keep[int(np.argmax(face_counts))] = True  # always keep the largest component

    if stats is not None:
        stats.parts_after = int(keep.sum())
        stats.dropped_parts = int((~keep).sum())
        stats.dropped_faces = int(face_counts[~keep].sum())
    if progress is not None:
        progress(
            "drop_parts",
            f"dropping free-floating debris "
            f"({int((~keep).sum())} parts / {int(face_counts[~keep].sum())} faces)",
        )
    if not (~keep).any():
        return mesh
    return trimesh.util.concatenate([p for p, k in zip(parts, keep, strict=True) if k])


def clean(
    mesh: trimesh.Trimesh,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[trimesh.Trimesh, CleanStats]:
    """Apply post-processing.

    Returns:
        `(post-processed mesh, record)`.
    """
    stats = CleanStats(faces_before=len(mesh.faces))
    mesh = fill_small_holes(mesh, config.FILL_HOLES_MAX_NBE, progress, stats)
    mesh = drop_small_parts(mesh, config.DROP_SMALL_PARTS, config.DROP_THIN_PARTS, progress, stats)
    stats.faces_after = len(mesh.faces)
    return mesh, stats
