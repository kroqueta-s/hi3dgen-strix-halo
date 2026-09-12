# SPDX-License-Identifier: MIT
"""Verify `drop_small_parts` (size and thinness).

A synthetic mesh is built from a body, a genuine detached part, a small crumb
and a surface flake, to confirm that **what should survive survives and what
should go is gone**.

Run it with this repository's virtual environment (trimesh is required)::

    .venv\\Scripts\\python.exe .\\tests\\test_drop_parts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = " OK " if ok else "FAIL"
    print(f"  {mark}  {name}" + (f"  ({detail})" if detail and not ok else ""))
    if not ok:
        FAILURES.append(name)


def build_specimen() -> tuple[trimesh.Trimesh, int]:
    """Body, part, crumb and flake. Returns (mesh, face count that should survive)."""
    body = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # A genuine part: 20 % long, 20 % thick (an arm, say).
    limb = trimesh.creation.box(extents=(0.2, 0.2, 0.2))
    limb.apply_translation((0.8, 0.0, 0.0))
    # A small crumb: 5 % long (the size test alone already removes it).
    crumb = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    crumb.apply_translation((0.0, 0.8, 0.0))
    # A surface flake: 30 % long, 0.5 % thick (this is what used to slip through).
    flake = trimesh.creation.box(extents=(0.3, 0.3, 0.005))
    flake.apply_translation((0.0, 0.0, 0.51))
    mesh = trimesh.util.concatenate([body, limb, crumb, flake])
    return mesh, len(body.faces) + len(limb.faces)


def run(label: str, drop_small_parts) -> None:  # noqa: ANN001
    mesh, want_faces = build_specimen()
    out = drop_small_parts(mesh, min_ratio=0.10, min_thick_ratio=0.02)
    parts = out.split(only_watertight=False)
    check(f"{label}: body and part survive as two components", len(parts) == 2, f"got {len(parts)}")
    check(f"{label}: face count matches body plus part", len(out.faces) == want_faces)

    # With the thinness test off, the flake stays (0 must mean disabled).
    out2 = drop_small_parts(mesh, min_ratio=0.10, min_thick_ratio=0.0)
    check(
        f"{label}: min_thick_ratio=0 keeps the flake",
        len(out2.split(only_watertight=False)) == 3,
    )

    # Both at 0 does nothing at all.
    out3 = drop_small_parts(mesh, min_ratio=0.0, min_thick_ratio=0.0)
    check(f"{label}: both at 0 changes nothing", len(out3.faces) == len(mesh.faces))

    # The largest component survives even when it is itself thin.
    thin_body = trimesh.creation.box(extents=(1.0, 1.0, 0.01))
    out4 = drop_small_parts(thin_body.copy(), min_ratio=0.10, min_thick_ratio=0.02)
    check(
        f"{label}: the largest component always survives",
        len(out4.faces) == len(thin_body.faces),
    )


def _reference_drop(
    mesh: trimesh.Trimesh, min_ratio: float, min_thick: float
) -> tuple[int, int, int]:
    """The decision the `split()`-based version made: (parts, kept, dropped).

    **This is the implementation that was replaced**, kept here as the
    reference. It builds a `Trimesh` per component - and `split()` runs
    `fill_holes()` on each, a repair nobody asked for - so it is the slow one;
    what it must not be is a *different* one.
    """
    parts = mesh.split(only_watertight=False)
    whole = max(float(np.max(mesh.bounding_box.extents)), 1e-12)
    face_counts = np.array([len(p.faces) for p in parts])
    keep = np.ones(len(parts), dtype=bool)
    if min_ratio > 0:
        keep &= (
            np.array([float(np.max(p.bounding_box.extents)) for p in parts]) / whole >= min_ratio
        )
    if min_thick > 0:
        keep &= (
            np.array([float(np.min(p.bounding_box.extents)) for p in parts]) / whole >= min_thick
        )
    keep[int(np.argmax(face_counts))] = True
    return len(parts), int(keep.sum()), int((~keep).sum())


def _soup(seed: int = 0) -> trimesh.Trimesh:
    """Many components, some sharing an edge three ways, with random flips.

    **A handful of boxes would not settle anything**: the labelling and the
    `split()` it replaces only disagree where the mesh is awkward - many
    components, inconsistent winding, an edge more than two faces use.
    """
    rng = np.random.default_rng(seed)
    pieces = []
    for _i in range(60):
        piece = trimesh.creation.icosphere(subdivisions=1, radius=rng.uniform(0.05, 0.5))
        piece.apply_translation(rng.uniform(-3, 3, size=3))
        faces = piece.faces.copy()
        flip = rng.random(len(faces)) < 0.5
        faces[flip] = faces[flip][:, [0, 2, 1]]
        pieces.append(trimesh.Trimesh(vertices=piece.vertices, faces=faces, process=False))
    mesh = trimesh.util.concatenate(pieces)
    base = len(mesh.vertices)
    extra_v = np.array([[5, 0, 0], [6, 0, 0], [5, 1, 0], [5, 0, 1], [5, -1, 0]], dtype=float)
    extra_f = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4]]) + base
    return trimesh.Trimesh(
        vertices=np.vstack([mesh.vertices, extra_v]),
        faces=np.vstack([mesh.faces, extra_f]),
        process=False,
    )


def run_equivalence(drop_small_parts, stats_class) -> None:  # noqa: ANN001
    """**The same parts, and not one face more.**

    `split()` filled the holes in every component it built, so the old version
    could return *more* faces than it was given. A face count that has grown is
    the sign that the replaced behaviour is still there.
    """
    mesh = _soup()
    stats = stats_class(faces_before=len(mesh.faces))
    out = drop_small_parts(mesh, 0.10, 0.02, None, stats)
    before, kept, dropped = _reference_drop(mesh, 0.10, 0.02)
    check(
        "equivalence: same number of parts found",
        stats.parts_before == before,
        f"{stats.parts_before} against {before}",
    )
    check(
        "equivalence: same parts kept",
        stats.parts_after == kept,
        f"{stats.parts_after} against {kept}",
    )
    check("equivalence: same parts dropped", stats.dropped_parts == dropped)
    check(
        "equivalence: no face was invented",
        len(out.faces) == len(mesh.faces) - stats.dropped_faces,
        f"{len(out.faces)} against {len(mesh.faces) - stats.dropped_faces}",
    )
    kept_positions = {tuple(map(tuple, np.round(out.vertices[f], 9))) for f in out.faces}
    input_positions = {tuple(map(tuple, np.round(mesh.vertices[f], 9))) for f in mesh.faces}
    check(
        "equivalence: every surviving face is one of the input's",
        kept_positions <= input_positions,
    )


def main() -> int:
    from runners.hi3dgen import postprocess as hi3

    run("hi3dgen", hi3.drop_small_parts)
    run_equivalence(hi3.drop_small_parts, hi3.CleanStats)

    total = 5
    print(f"\n{total - len(FAILURES)}/{total} passed")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
