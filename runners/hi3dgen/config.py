# SPDX-License-Identifier: MIT
"""Configuration for the Hi3DGen runner, read from `.env`.

**This runner is self-contained.** It never reads hearth's configuration, so it
works unchanged as the standalone `hi3dgen-strix-halo` repository.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# runners/hi3dgen/config.py -> the repository root.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent
load_dotenv(REPO_ROOT / ".env")


def _str(key: str, default: str = "") -> str:
    raw = os.getenv(key)
    return raw.strip() if raw is not None and raw.strip() != "" else default


def _int(key: str, default: int) -> int:
    raw = os.getenv(key)
    return int(raw) if raw is not None and raw.strip() != "" else default


def _float(key: str, default: float) -> float:
    raw = os.getenv(key)
    return float(raw) if raw is not None and raw.strip() != "" else default


def _bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# The upstream clone (**never a fork**).
HI3DGEN_REPO: Path = Path(_str("HI3DGEN_REPO"))
# Weights for the 3D pipeline (the directory holding `pipeline.json`).
HI3DGEN_WEIGHTS_DIR: Path = Path(_str("HI3DGEN_WEIGHTS_DIR"))
# Background removal (BiRefNet). **Upstream opens `weights/BiRefNet` by relative
# path, so the runner loads it and assigns it to `birefnet_model` instead.**
HI3DGEN_BIREFNET_DIR: Path = Path(_str("HI3DGEN_BIREFNET_DIR"))
# Normal estimation (StableNormal): the torch.hub repository, the weight
# version, and where to cache it.
HI3DGEN_NORMAL_HUB: str = _str("HI3DGEN_NORMAL_HUB", "hugoycj/StableNormal")
HI3DGEN_NORMAL_VERSION: str = _str("HI3DGEN_NORMAL_VERSION", "yoso-normal-v1-8-1")
HI3DGEN_NORMAL_CACHE_DIR: Path = Path(_str("HI3DGEN_NORMAL_CACHE_DIR"))
HI3DGEN_NORMAL_RESOLUTION: int = _int("HI3DGEN_NORMAL_RESOLUTION", 768)

# Defaults from the upstream demo (`app.py`). **Only 6 steps on the sparse
# stage** is what distinguishes Hi3DGen.
SS_STEPS: int = _int("HI3DGEN_SS_STEPS", 50)
SLAT_STEPS: int = _int("HI3DGEN_SLAT_STEPS", 6)
SS_GUIDANCE: float = _float("HI3DGEN_SS_GUIDANCE", 3.0)
SLAT_GUIDANCE: float = _float("HI3DGEN_SLAT_GUIDANCE", 3.0)

# Attention head chunk. Measured best on Hunyuan3D at 4. **Do not change it
# without evidence.**
ATTN_HEAD_CHUNK: int = _int("HI3DGEN_ATTN_HEAD_CHUNK", 4)

# Whether to set TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL **before torch is
# imported**. Measured 2026-09-01 on gfx1151: it makes the flash and
# memory-efficient kernels available, taking seq=4096 from 0.135s to 0.012s and
# seq=9216 from 1.167s to 0.059s, with identical output. **Setting it afterwards
# has no effect**, so it goes at the top of `__main__.py`.
FAST_ATTENTION: bool = _bool("HI3DGEN_FAST_ATTENTION", True)

# Whether to set TORCH_BLAS_PREFER_HIPBLASLT and ROCBLAS_USE_HIPBLASLT **before
# torch is imported**. Measured 2026-09-02 on gfx1151 (ROCm 7.2.1): hipBLASLt
# runs the sparse-conv shim's skinny GEMM (M=voxels, N=128, K=2048) at
# 14 TFLOPS where rocBLAS delivers 1.5, taking the slat stage from 17.1 s to
# 10.7 s; the attention-bound structure stage does not move. Which backend was
# in effect is recorded in `metrics.blas_backend`.
PREFER_HIPBLASLT: bool = _bool("HI3DGEN_PREFER_HIPBLASLT", True)

# Whether to run the render-loop keepalive during generation (`gfxlight.py`).
# Measured 2026-09-02: no effect in any state (display on: nothing to fix;
# display off: the GPU pins at 600 MHz with the loop alive - see gfx1151-gemm
# docs/displayoff.md; keeping the display awake is what works). Kept as an
# experiment switch. Generation works as before if it fails to start;
# `metrics.gfx_keepalive` records whether it was alive.
GFX_KEEPALIVE: bool = _bool("HI3DGEN_GFX_KEEPALIVE", True)

# Whether to hold the console display awake during generation
# (`displaykeep.py`, SetThreadExecutionState). When the display turns off -
# lid, or the display-off timeout, locked or not - the driver pins the GPU
# near 600 MHz and generation runs ~4x slower until it comes back (measured
# 2026-09-02; gfx1151-gemm docs/displayoff.md). **Off by default**: it keeps
# the panel lit, and a machine whose display never sleeps needs nothing here.
# `metrics.display_keepalive` records whether the hold took effect.
DISPLAY_KEEPALIVE: bool = _bool("HI3DGEN_DISPLAY_KEEPALIVE", False)


# **Cap on dedicated VRAM (GB).** gfx1151 has 32 GB of dedicated VRAM, but the
# total from `torch.cuda.mem_get_info` is 43.87 GB because it counts shared
# memory. Overflow therefore raises nothing and **silently becomes several times
# slower** (measured on 2026-09-01). Passing the cap to torch as well turns that
# into an **immediate OOM**.
VRAM_LIMIT_GB: float = _float("HI3DGEN_VRAM_LIMIT_GB", 30.0)

# Heartbeat interval in seconds. It exists so that **nothing runs silently for a
# long time**.
HEARTBEAT_SEC: float = _float("HI3DGEN_HEARTBEAT_SEC", 10.0)


# --- Post-processing ---------------------------------------------------------
# **Upstream (Stable3DGen) has no post-processing**, so no visibility-based face
# removal happens here. The only addition is dropping free-floating debris by
# size; 0 disables it. Measured: at 10 % the arms and hands (15 %) survive and
# the visible debris (6.5 % and below) is gone.
DROP_SMALL_PARTS: float = _float("HI3DGEN_DROP_SMALL_PARTS", 0.10)

# **Also not in upstream.** Drop detached components whose bounding-box minimum
# extent is below this fraction of the model's longest side (paper-thin flakes).
# Flakes hovering about 1 % off the surface are 11-29 % long, so DROP_SMALL_PARTS
# alone lets them through. Measured 2026-09-02: flakes are 1.4 % thick or less,
# genuine parts 11.8 % or more. 0 disables it.
DROP_THIN_PARTS: float = _float("HI3DGEN_DROP_THIN_PARTS", 0.02)

# Maximum number of edges in a boundary loop that gets filled (0 disables it).
# Hi3DGen output is clipped at the top of the generation volume and is therefore
# never watertight. The largest loop measured had 128 vertices, so 250 leaves
# room to spare.
FILL_HOLES_MAX_NBE: int = _int("HI3DGEN_FILL_HOLES_MAX_NBE", 250)
