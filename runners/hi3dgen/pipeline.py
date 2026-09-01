# SPDX-License-Identifier: MIT
"""Hi3DGen（画像 → 法線 → メッシュ）の実体。**GPU を握るのはここだけ。**

Hi3DGen は TRELLIS の派生で、**画像そのものではなく法線マップを 3D パイプラインへ渡す**
（normal bridging）。そのため前段が 2 つある：

1. **BiRefNet**（背景除去）。**ベンダーコードが `weights/BiRefNet` を相対パスで開く**ので、
   ここで読み込んで `pipeline.birefnet_model` に差しておく（ベンダーコードは書き換えない）
2. **StableNormal**（法線推定）。`torch.hub` から取る

gfx1151 / Windows / ROCm での必須の細工は `shims.install()` に閉じ込めてある。
**根拠は `shims.py` の docstring と `docs/02_port_report.md`。**
"""

from __future__ import annotations

import gc
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import trimesh
from PIL import Image

from . import config, postprocess, shims

NAME = "hi3dgen"
VERSION = "trellis-normal-v0-1"

_PIPELINE: Any = None
_NORMAL: Any = None
_LOAD_SEC: float = 0.0
# 速いアテンション（AOTriton）が効いているか。**metrics に載せて記録に残す。**
_FAST_ATTENTION: bool = False


class _DeviceWatch:
    """デバイスの実使用量を追い、**一定間隔で生存を知らせる**監視スレッド。

    以前は使用量のピークを取るだけだった。それだけだと、長い段の途中で
    「進んでいるのか、止まっているのか」が外から一切分からない。実際に
    2026-09-01、生成が黙って 12 分以上走るのを何度も待ってしまった。

    見ているのは 3 つ：

    1. **生存**（`heartbeat`）。既定 10 秒ごとに経過秒と VRAM を流す。
       呼び出し側はこれが止まったことで「進んでいない」を判定できる
    2. **専用 VRAM の超過**（`vram_over`）。**専用 VRAM は 32GB しかない。**
       `torch.cuda.mem_get_info` の total（43.87GB）は共有メモリ込みの嘘なので、
       溢れても例外にならず、**黙って数倍遅くなる**。ここを跨いだ瞬間に知らせる
    3. ピーク（従来どおり `metrics` へ載せる）

    **このスレッドから `progress` を呼ぶので、呼び出し側の emit は鍵で守ること。**
    """

    def __init__(
        self,
        progress: Callable[[str, str], None] | None = None,
        stage: str = "",
        interval: float = 2.0,
        heartbeat_sec: float = 10.0,
        limit_gb: float = 0.0,
    ) -> None:
        self.interval = interval
        self.heartbeat_sec = heartbeat_sec
        self.limit_gb = limit_gb
        self.stage = stage
        self.peak_used_gb = 0.0
        self.exceeded = False
        self._progress = progress
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _say(self, stage: str, message: str) -> None:
        if self._progress is not None:
            self._progress(stage, message)

    def _run(self) -> None:
        started = time.perf_counter()
        last_beat = started
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1024**3
            if used > self.peak_used_gb:
                self.peak_used_gb = used
            now = time.perf_counter()
            if self.limit_gb > 0 and used > self.limit_gb and not self.exceeded:
                self.exceeded = True
                self._say(
                    "vram_over",
                    f"**専用 VRAM を超えた**（{used:.2f}GB > {self.limit_gb:.2f}GB）。"
                    "共有メモリへ溢れているので、このまま待っても遅いだけ",
                )
            if now - last_beat >= self.heartbeat_sec:
                last_beat = now
                self._say(
                    "heartbeat",
                    f"{self.stage or '実行中'} 経過 {now - started:.0f}s / "
                    f"VRAM {used:.2f}GB（ピーク {self.peak_used_gb:.2f}GB）",
                )
            self._stop.wait(self.interval)

    def __enter__(self) -> _DeviceWatch:
        self._t.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join(timeout=5)


def apply_vram_limit() -> float:
    """**専用 VRAM を超えたら黙って遅くなるのではなく、その場で落ちる**ようにする。

    `torch.cuda.mem_get_info` の総容量は共有メモリ込み（gfx1151 で 43.87GB）で、
    専用 VRAM の 32GB を超えても例外にならない。超えた分はホスト側のメモリへ落ちるので、
    **例外も警告も出ないまま数倍遅くなる**（2026-09-01、疎畳み込みで実測。最終的には
    42.02GB まで確保して `torch.OutOfMemoryError` に至った）。

    そこで割り当ての上限を総容量に対する割合で torch へ伝える。上限を超える確保は
    `torch.OutOfMemoryError` になるので、**待たされずに気付ける。**

    Returns:
        実際に設定した上限（GB）。設定できなければ 0.0。
    """
    limit = float(config.VRAM_LIMIT_GB)
    if limit <= 0 or not torch.cuda.is_available():
        return 0.0
    _, total = torch.cuda.mem_get_info()
    total_gb = total / 1024**3
    fraction = min(max(limit / total_gb, 0.05), 1.0)
    torch.cuda.set_per_process_memory_fraction(fraction)
    return limit


def device_memory_gb() -> tuple[float, float]:
    """デバイスの (使用中, 合計) を GB で返す。

    **`total` は共有メモリ込みの値で、専用 VRAM の 32GB とは一致しない。**
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 1024**3, total / 1024**3


def _prepare_environment() -> None:
    """`hi3dgen` を import する**前に**、環境変数とシムを整える。

    上流は import 時に環境変数を読んで分岐するので、**後から設定しても効かない。**
    """
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_BACKEND", "spconv")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "flash_attn")
    os.environ.setdefault("SPCONV_ALGO", "native")

    repo = str(config.HI3DGEN_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    global _FAST_ATTENTION
    _FAST_ATTENTION = shims.install(head_chunk=config.ATTN_HEAD_CHUNK)


def load_pipeline(progress: Callable[[str, str], None] | None = None) -> tuple[Any, Any]:
    """3D パイプライン・BiRefNet・StableNormal をまとめて読み込む。

    **段ごとに時間を知らせる。** 切替が遅いときに「どこで待っているか」が
    分からないと手が打てない（2026-09-01 に実際に困った）。
    """
    global _PIPELINE, _NORMAL, _LOAD_SEC
    if _PIPELINE is not None and _NORMAL is not None:
        return _PIPELINE, _NORMAL

    if not config.HI3DGEN_REPO.is_dir():
        raise FileNotFoundError(f"Stable3DGen の clone が無い: {config.HI3DGEN_REPO}")
    if not (config.HI3DGEN_WEIGHTS_DIR / "pipeline.json").is_file():
        raise FileNotFoundError(f"重みが無い: {config.HI3DGEN_WEIGHTS_DIR}/pipeline.json")
    if not config.HI3DGEN_BIREFNET_DIR.is_dir():
        raise FileNotFoundError(f"BiRefNet の重みが無い: {config.HI3DGEN_BIREFNET_DIR}")

    def _say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    _say("import", "hi3dgen を import する（シムを差し込んでから）")
    _prepare_environment()
    limit = apply_vram_limit()
    _say("vram_limit", f"専用 VRAM の上限を {limit:.1f}GB に設定した（超えたら即 OOM で落ちる）")
    from hi3dgen.pipelines import Hi3DGenPipeline
    from transformers import AutoModelForImageSegmentation

    started = time.perf_counter()
    _say("weights", "3D パイプラインの重みを読み込む")
    pipeline = Hi3DGenPipeline.from_pretrained(str(config.HI3DGEN_WEIGHTS_DIR))
    _say("to_gpu", "GPU へ載せる")
    pipeline.cuda()

    # **ベンダーコードの `_lazy_load_birefnet` は `weights/BiRefNet` を相対パスで開く。**
    # 先に差しておけば呼ばれないので、ベンダーコードを書き換えずに済む。
    pipeline.birefnet_model = (
        AutoModelForImageSegmentation.from_pretrained(
            str(config.HI3DGEN_BIREFNET_DIR), trust_remote_code=True
        )
        .to(pipeline.device)
        .eval()
    )

    _say("birefnet", "背景除去の重みを読み込んだ")
    normal = torch.hub.load(
        config.HI3DGEN_NORMAL_HUB,
        "StableNormal_turbo",
        trust_repo=True,
        yoso_version=config.HI3DGEN_NORMAL_VERSION,
        local_cache_dir=str(config.HI3DGEN_NORMAL_CACHE_DIR),
    )

    _LOAD_SEC = time.perf_counter() - started
    _say("loaded", f"読み込み終わり（{_LOAD_SEC:.1f}s）")
    _PIPELINE, _NORMAL = pipeline, normal
    return _PIPELINE, _NORMAL


def unload_pipeline() -> bool:
    """重みを解放して VRAM を返す。

    Returns:
        実際に解放したか（読み込んでいなければ False）。
    """
    global _PIPELINE, _NORMAL, _LOAD_SEC
    if _PIPELINE is None and _NORMAL is None:
        return False
    _PIPELINE = None
    _NORMAL = None
    _LOAD_SEC = 0.0
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return True


@dataclass
class MeshResult:
    """生成の結果と実測。"""

    mesh: trimesh.Trimesh
    foreground: Image.Image
    normal: Image.Image
    load_sec: float
    gen_sec: float
    preprocess_sec: float
    normal_sec: float
    cond_sec: float
    structure_sec: float
    slat_sec: float
    decode_sec: float
    n_voxels: int
    vram_peak_gb: float
    fast_attention: bool
    clean: dict[str, Any]
    ss_steps: int
    slat_steps: int
    ss_guidance: float
    slat_guidance: float
    seed: int


def generate_mesh(
    image: Image.Image,
    ss_steps: int | None = None,
    slat_steps: int | None = None,
    ss_guidance: float | None = None,
    slat_guidance: float | None = None,
    seed: int = 0,
    progress: Callable[[str, str], None] | None = None,
) -> MeshResult:
    """画像 1 枚からメッシュを生成する。

    **前処理（背景除去）と法線推定はこのランナーの責任**（契約 §4）。
    **座標系は上流のまま（Z-up）で返す。** 実寸化はしない（下流の `forge` の仕事）。

    上流の `pipeline.run()` と**同じ手順を自分で踏む**。理由は 2 つ：

    1. **どの段で時間を使っているかを測るため。** 2026-09-01 に「hearth 経由だと
       生成だけが 4 倍遅い（59.4s -> 229.6s）」という差が出て、`run()` を丸ごと
       呼んでいると内訳が取れず切り分けられなかった
    2. **活性ボクセル数を記録するため。** 疎な段の費用はこれでほぼ決まる

    **ベンダーコードは書き換えていない**（公開メソッドを順に呼んでいるだけ）。
    """
    pipeline, normal_predictor = load_pipeline(progress)

    def _say(stage: str, message: str) -> None:
        if progress is not None:
            progress(stage, message)

    ss = config.SS_STEPS if ss_steps is None else int(ss_steps)
    slat = config.SLAT_STEPS if slat_steps is None else int(slat_steps)
    ss_cfg = config.SS_GUIDANCE if ss_guidance is None else float(ss_guidance)
    slat_cfg = config.SLAT_GUIDANCE if slat_guidance is None else float(slat_guidance)

    with _DeviceWatch(
        progress=progress,
        stage="生成",
        heartbeat_sec=config.HEARTBEAT_SEC,
        limit_gb=config.VRAM_LIMIT_GB,
    ) as sampler:
        _say("rembg", "背景を除去する（BiRefNet）")
        started = time.perf_counter()
        foreground = pipeline.preprocess_image(image, resolution=1024)
        preprocess_sec = time.perf_counter() - started

        _say("normal", "法線を推定する（StableNormal）")
        started = time.perf_counter()
        normal_image = normal_predictor(
            foreground,
            resolution=config.HI3DGEN_NORMAL_RESOLUTION,
            match_input_resolution=True,
            data_type="object",
        )
        normal_sec = time.perf_counter() - started

        # **`torch.no_grad()` を外さない。** 上流の `run()` にはデコレータで付いているが、
        # `sample_sparse_structure` などの個別のメソッドには付いていない。段に分けて測ると
        # ここが抜け、**自動微分のグラフを溜め込んで VRAM を食い尽くす**
        # （2026-09-01、復号の段で 29.66GB まで確保して OOM。
        #   上限を切っていたので 100 秒で気付けた）。
        with torch.no_grad():
            started = time.perf_counter()
            _say("cond", "画像を条件ベクトルにする")
            cond = pipeline.get_cond([normal_image])
            cond_sec = time.perf_counter() - started

            torch.manual_seed(int(seed))

            _say("structure", f"疎構造をサンプルする（steps={ss}）")
            step_started = time.perf_counter()
            coords = pipeline.sample_sparse_structure(
                cond, 1, {"steps": ss, "cfg_strength": ss_cfg}
            )
            structure_sec = time.perf_counter() - step_started
            n_voxels = int(coords.shape[0])

            _say("slat", f"潜在をサンプルする（steps={slat} / 活性ボクセル {n_voxels}）")
            step_started = time.perf_counter()
            slat_latent = pipeline.sample_slat(
                cond, coords, {"steps": slat, "cfg_strength": slat_cfg}
            )
            slat_sec = time.perf_counter() - step_started

            _say("decode", "メッシュへ復号する")
            step_started = time.perf_counter()
            outputs = pipeline.decode_slat(slat_latent, ["mesh"])
            decode_sec = time.perf_counter() - step_started
            gen_sec = time.perf_counter() - started

    extracted = outputs["mesh"][0]
    if not bool(getattr(extracted, "success", True)):
        raise RuntimeError("メッシュが空だった（前景が取れていない可能性がある）")

    mesh = trimesh.Trimesh(
        vertices=extracted.vertices.detach().float().cpu().numpy(),
        faces=extracted.faces.detach().cpu().numpy(),
        process=False,
    )
    # **後処理は上流の手法に倣う。** 詳細と、上流に無い 1 点は postprocess.py の docstring。
    mesh, clean_stats = postprocess.clean(mesh, progress)
    return MeshResult(
        mesh=mesh,
        foreground=foreground,
        normal=normal_image,
        load_sec=_LOAD_SEC,
        gen_sec=gen_sec,
        preprocess_sec=preprocess_sec,
        normal_sec=normal_sec,
        cond_sec=cond_sec,
        structure_sec=structure_sec,
        slat_sec=slat_sec,
        decode_sec=decode_sec,
        n_voxels=n_voxels,
        vram_peak_gb=sampler.peak_used_gb,
        fast_attention=_FAST_ATTENTION,
        clean=clean_stats.as_dict(),
        ss_steps=ss,
        slat_steps=slat,
        ss_guidance=ss_cfg,
        slat_guidance=slat_cfg,
        seed=int(seed),
    )
