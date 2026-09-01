# SPDX-License-Identifier: MIT
"""生成したメッシュの後処理。**上流の手法に倣い、独自の判断を足さない。**

**Hi3DGen の上流（Stable3DGen）には後処理が無い。** TRELLIS から派生する際に
`kaolin` / `nvdiffrast` / `flexicubes` への依存を外しており、`postprocess_mesh` に
あたるものごと落としている（README が「商用利用できるように外した」と書いている）。
メッシュ化は `skimage.measure` の marching cubes で、そのまま返している。

したがって **TRELLIS 側で行う「多視点ラスタライズ＋min-cut」はここでは行わない。**
上流に無い処理を持ち込まないためである。

## 上流に無い処理（**2 つだけ。どちらも実測した欠陥への手当て**）

### 1. 小さな穴を塞ぐ（`pymeshfix.fill_small_boundaries`）

Hi3DGen の出力は **watertight にならない**。実測（2026-09-01・検体 `i2i_00038_.png`）で
境界辺 198 本の正体を調べると、**4 本のループがすべて z = 39.9mm の同一平面**にあり、
厚みがほぼ 0 の平らな開口だった。高さは 80.0mm（z は -40.1〜39.9）なので、
**生成ボリュームの天面で切り取られている**（ランダムな割れではない）。
被写体の切り出しは上流が `size×1.2` で決めており、ランナー側からは変えられない。

そこで **TRELLIS 上流が使っているのと同じ `pymeshfix` の同じ呼び出し**で塞ぐ。
`forge` の `repair_manifold` は既に試して閉じられていない（警告を返した）。

### 2. 外側に浮いた破片を大きさと薄さで落とす

**外側に浮いている破片を、大きさと薄さで落とす。**
（`.env` の `HI3DGEN_DROP_SMALL_PARTS` / `HI3DGEN_DROP_THIN_PARTS`）

実測（2026-09-01・検体 `i2i_00038_.png`）では、本体以外 968 個・33,056 面のうち
**941 個・30,048 面（90.9%）が本体の外側**に浮いていた。印刷用途では実害になる。
**落とした量は必ず記録に残す。**
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
    """小さな境界ループを塞ぐ（**上流 TRELLIS と同じ `pymeshfix` の呼び出し**）。

    Args:
        mesh: 対象。
        max_nbe: 塞ぐ境界ループの辺数の上限。0 以下なら何もしない。
        progress: 段の通知先。
        stats: 記録の置き場。

    Returns:
        穴を塞いだメッシュ。**失敗したら入力をそのまま返す**（後処理の失敗で生成を落とさない）。
    """
    if max_nbe <= 0:
        return mesh
    before = len(mesh.faces)
    try:
        from pymeshfix import _meshfix

        if progress is not None:
            progress("fill_holes", f"小さな穴を塞ぐ（辺 {max_nbe} 本まで）")
        fixer = _meshfix.PyTMesh()
        fixer.load_array(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
        )
        fixer.fill_small_boundaries(nbe=max_nbe, refine=True)
        vertices, faces = fixer.return_arrays()
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    except Exception as exc:  # noqa: BLE001 - 後処理の失敗で生成を落とさない
        message = f"穴埋めに失敗した（メッシュはそのまま返す）: {type(exc).__name__}: {exc}"
        print(f"[postprocess] {message}", file=sys.stderr)
        if stats is not None:
            stats.warnings.append(message)
        return mesh
    if stats is not None:
        stats.filled_faces = len(mesh.faces) - before
    return mesh


@dataclass
class CleanStats:
    """後処理で何がどれだけ変わったかの記録。**黙って消さないための数字。**"""

    faces_before: int = 0
    faces_after: int = 0
    filled_faces: int = 0
    parts_before: int = 0
    parts_after: int = 0
    dropped_parts: int = 0
    dropped_faces: int = 0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """`metrics` へそのまま載せられる形。"""
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
    """**外側に浮いた破片を「小ささ」と「薄さ」で落とす**（上流に無い、こちらの追加）。

    判定は成分の外接箱（軸平行）と全体の最長辺の比で、2 つの条件の**両方**を
    満たす成分だけ残す：

    1. **最長辺が `min_ratio` 以上**（小さな破片を落とす）。面数ではなく空間の大きさで
       見るのは、細かく分割された小片と、面数が少ないだけの正当な部品を分けるため
    2. **最短辺が `min_thick_ratio` 以上**（薄片を落とす）。表面から約 1% 浮いて
       平行に張り付く紙のような薄片は、**長さが 11〜29% あるので 1 だけでは
       素通りする**（描画では表面の黒い斑点や板状の突起に見える）

    実測（検体 `i2i_00038_.png`・2026-09-02）：残っていた薄片は**厚み 0.1〜1.4%**、
    正当な部品（腕・パネル）は**厚み 11.8% 以上**で、2% にすると桁の余裕で分離できた。
    斜めの薄片は軸平行の外接箱では厚めに出るが、実測の薄片はすべて表面に平行＝
    ほぼ軸平行で問題にならなかった。

    Args:
        mesh: 対象。
        min_ratio: 残す最小の大きさ（全体の最長辺に対する比）。0 以下なら判定しない。
        min_thick_ratio: 残す最小の厚み（同）。0 以下なら判定しない。
        progress: 段の通知先。
        stats: 記録の置き場。

    Returns:
        破片を除いたメッシュ。**最大の成分だけは必ず残す。**
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
    keep[int(np.argmax(face_counts))] = True  # 最大の成分は必ず残す

    if stats is not None:
        stats.parts_after = int(keep.sum())
        stats.dropped_parts = int((~keep).sum())
        stats.dropped_faces = int(face_counts[~keep].sum())
    if progress is not None:
        progress(
            "drop_parts",
            f"浮いた破片を落とす（{int((~keep).sum())} 個 / {int(face_counts[~keep].sum())} 面）",
        )
    if not (~keep).any():
        return mesh
    return trimesh.util.concatenate([p for p, k in zip(parts, keep, strict=True) if k])


def clean(
    mesh: trimesh.Trimesh,
    progress: Callable[[str, str], None] | None = None,
) -> tuple[trimesh.Trimesh, CleanStats]:
    """後処理を掛ける。

    Returns:
        `(後処理後のメッシュ, 記録)`。
    """
    stats = CleanStats(faces_before=len(mesh.faces))
    mesh = fill_small_holes(mesh, config.FILL_HOLES_MAX_NBE, progress, stats)
    mesh = drop_small_parts(mesh, config.DROP_SMALL_PARTS, config.DROP_THIN_PARTS, progress, stats)
    stats.faces_after = len(mesh.faces)
    return mesh, stats
