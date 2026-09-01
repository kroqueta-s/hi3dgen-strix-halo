# SPDX-License-Identifier: MIT
"""Hi3DGen ランナーの設定（`.env` から読み込む）。

**このランナーは自分の中で閉じている。** hearth の設定を参照しないので、
`hi3dgen-strix-halo` として独立リポジトリへ出しても、そのまま動く。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# runners/hi3dgen/config.py -> リポジトリのルート。
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


# 上流の clone（**フォークしない**）。
HI3DGEN_REPO: Path = Path(_str("HI3DGEN_REPO"))
# 3D パイプラインの重み（`pipeline.json` があるディレクトリ）。
HI3DGEN_WEIGHTS_DIR: Path = Path(_str("HI3DGEN_WEIGHTS_DIR"))
# 背景除去（BiRefNet）。**ベンダーコードは `weights/BiRefNet` を相対パスで開くので、
# ランナー側で読んで `birefnet_model` に差す。**
HI3DGEN_BIREFNET_DIR: Path = Path(_str("HI3DGEN_BIREFNET_DIR"))
# 法線推定（StableNormal）。torch.hub の repo と重みの版、および取得先。
HI3DGEN_NORMAL_HUB: str = _str("HI3DGEN_NORMAL_HUB", "hugoycj/StableNormal")
HI3DGEN_NORMAL_VERSION: str = _str("HI3DGEN_NORMAL_VERSION", "yoso-normal-v1-8-1")
HI3DGEN_NORMAL_CACHE_DIR: Path = Path(_str("HI3DGEN_NORMAL_CACHE_DIR"))
HI3DGEN_NORMAL_RESOLUTION: int = _int("HI3DGEN_NORMAL_RESOLUTION", 768)

# 上流のデモの既定値（`app.py`）。**疎な段は 6 歩しか回さない**のが Hi3DGen の特徴。
SS_STEPS: int = _int("HI3DGEN_SS_STEPS", 50)
SLAT_STEPS: int = _int("HI3DGEN_SLAT_STEPS", 6)
SS_GUIDANCE: float = _float("HI3DGEN_SS_GUIDANCE", 3.0)
SLAT_GUIDANCE: float = _float("HI3DGEN_SLAT_GUIDANCE", 3.0)

# アテンションのヘッド分割。Hunyuan3D 実測で 4 が最良。**根拠なく変えない。**
ATTN_HEAD_CHUNK: int = _int("HI3DGEN_ATTN_HEAD_CHUNK", 4)

# **torch を import する前に** TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL を立てるか。
# 実測（2026-09-01・gfx1151）：立てると flash / mem-efficient が使えるようになり、
# seq=4096 で 0.135s -> 0.012s、seq=9216 で 1.167s -> 0.059s。出力は一致する。
# **後から os.environ へ入れても効かない**ので、`__main__.py` の先頭で置く。
FAST_ATTENTION: bool = _bool("HI3DGEN_FAST_ATTENTION", True)

# 生成中だけ「3D の常夜灯」を点けるか（`gfxlight.py`）。Windows の AMD ドライバは
# compute だけの負荷ではクロックを上げない（実測：GEMM 単独 600 MHz / 3D 併用 2.35 GHz・
# 4.3 倍）。点かなくても生成は従来どおり動く。効いたかは metrics.gfx_keepalive に載る。
GFX_KEEPALIVE: bool = _bool("HI3DGEN_GFX_KEEPALIVE", True)


# **専用 VRAM の上限（GB）。** gfx1151 の専用 VRAM は 32GB だが、
# `torch.cuda.mem_get_info` の total は共有メモリ込みの 43.87GB を返す。
# そのため溢れても例外にならず、**黙って数倍遅くなる**（2026-09-01 に実測で踏んだ）。
# ここを torch にも伝えて、超えたら OOM で**すぐ落ちる**ようにする。
VRAM_LIMIT_GB: float = _float("HI3DGEN_VRAM_LIMIT_GB", 30.0)

# 生存確認を流す間隔（秒）。**黙って長時間走らせない**ためのもの。
HEARTBEAT_SEC: float = _float("HI3DGEN_HEARTBEAT_SEC", 10.0)


# --- 後処理 -----------------------------------------------------------------
# **上流（Stable3DGen）には後処理が無い**ので、可視率による面の除去は行わない。
# 足すのは「外側に浮いた破片を大きさで落とす」だけ。0 で無効。
# 実測では 10% で腕と手（15%）は残り、目に見える破片（6.5% 以下）が消えた。
DROP_SMALL_PARTS: float = _float("HI3DGEN_DROP_SMALL_PARTS", 0.10)

# **これも上流に無い追加。** 外接箱の最短辺が全体の最長辺のこの比未満の分離成分
# （＝紙のような薄片）を落とす。表面から約 1% 浮いた薄片は長さが 11〜29% あるので
# DROP_SMALL_PARTS だけでは素通りする。実測：薄片は厚み 1.4% 以下・正当な部品は
# 11.8% 以上（2026-09-02）。0 で無効。
DROP_THIN_PARTS: float = _float("HI3DGEN_DROP_THIN_PARTS", 0.02)

# 小さな穴を塞ぐときの境界ループの辺数の上限（0 で無効）。
# Hi3DGen の出力は生成ボリュームの天面で切れていて watertight にならない。
# 実測では最大のループが 128 頂点だったので、余裕をみて 250 にする。
FILL_HOLES_MAX_NBE: int = _int("HI3DGEN_FILL_HOLES_MAX_NBE", 250)
