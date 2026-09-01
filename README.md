# hi3dgen-strix-halo

**[Hi3DGen / Stable3DGen](https://github.com/Stable-X/Stable3DGen) image-to-mesh
on AMD Strix Halo (gfx1151), Windows, ROCm — with no CUDA-only package
installed.**

Hi3DGen goes image → normal map → mesh. Upstream needs `spconv` and `xformers`,
neither of which exists for Windows + ROCm. This repository supplies pure-torch
replacements that are injected at launch time, so **upstream code is cloned and
run unmodified**.

This is a runner for [hearth](https://github.com/kroqueta-s/hearth): it speaks
one JSON object per line over stdin/stdout. It also runs standalone.

## Install

```powershell
git clone https://github.com/kroqueta-s/hi3dgen-strix-halo
cd hi3dgen-strix-halo
.\install.ps1
```

That creates a virtual environment, installs ROCm PyTorch, clones upstream at a
pinned commit, downloads the weights (5.4 GB across three repositories), writes
`.env`, and **verifies the replacements against exact references** before you
trust any mesh.

Requirements: Windows, an AMD GPU with ROCm 7.2.1 drivers, Python 3.12, ~15 GB
of disk, and about 17 GB of VRAM at peak.

## Use

```powershell
.venv\Scripts\python.exe -m runners.hi3dgen
```

```json
{"id": 1, "method": "capabilities"}
{"id": 2, "method": "image_to_mesh", "params": {"image_path": "C:/in.png", "out_dir": "C:/out"}}
```

`image_to_mesh` writes `raw.ply` plus the intermediate `foreground.png` and
`normal.png`. Parameters: `ss_steps`, `slat_steps`, `ss_guidance`,
`slat_guidance`, `seed`.

## What is replaced, and why it is safe

| Upstream dependency | Replacement | Verified by |
|---|---|---|
| `spconv` (sparse conv) | `runners/hi3dgen/shims.py` — submanifold convolution in torch | Exact agreement with a dense `F.conv3d` reference |
| `xformers` / `flash_attn` | Same file — `F.scaled_dot_product_attention` | Agreement with a naive attention reference |

```powershell
.venv\Scripts\python.exe tests\test_shims.py
```

Submanifold convolution is exactly a dense convolution restricted to occupied
voxels, so it can be checked against a reference without the original library.

## Two pins you must not float

- **`diffusers==0.31.0`** — StableNormal imports `diffusers.models.controlnet`,
  moved in 0.32 and absent in 0.40.
- **`transformers==4.46.3`** — diffusers 0.31 imports `FLAX_WEIGHTS_NAME`,
  removed in transformers 5.

## Measurements (gfx1151, Radeon 8060S, 32 GB dedicated VRAM)

One image, `ss_steps = 12`, `slat_steps = 6`, 2026-09-01:

| Stage | Time |
|---|--:|
| Load weights (3D + BiRefNet + StableNormal) | 29 s |
| Background removal (BiRefNet) | 8 s |
| Normal estimation (StableNormal) | 5 s |
| Sparse structure | 46 s |
| Structured latent | 43 s |
| Decode to mesh | 10 s |
| Post-processing | 9 s |
| **Total** | **161 s** |

Peak VRAM 16.2 GB. Output 791,914 faces, watertight, no boundary edges and no
non-manifold edges after downstream repair.

**The first run is much slower**: MIOpen tunes convolution kernels once, which
took 793 s for normal estimation and 165 s for background removal. The second
run took 1.7 s and 3.4 s for the same stages. Do not judge speed from the first
run.

Attention is 10–20× faster when `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is
set **before** torch is imported. The runner sets it for you.

## Limits

- **No texture.** Upstream produces geometry only.
- **The mesh is clipped at the top of the generation volume.** Measured on one
  sample, all four boundary loops lay on a single plane at the top. Upstream
  frames the subject at 1.2× its bounding box and that is not adjustable from
  outside. The runner closes those openings with `pymeshfix`, the same call
  upstream TRELLIS uses for the same purpose.
- **Free-floating parts.** Measured on one sample, 941 of 968 stray parts were
  outside the body. Upstream has no post-processing at all, so the runner drops
  parts below 10 % of the model's longest side and records how much it dropped.
  Set `HI3DGEN_DROP_SMALL_PARTS=0` to disable.
- Generation time varies on this hardware for identical settings. Do not use it
  as a pass/fail signal.

## License

MIT (see [LICENSE](LICENSE)). Upstream Stable3DGen is MIT. Weights:
`Stable-X/trellis-normal-v0-1` MIT, `ZhengPeng7/BiRefNet` MIT,
`Stable-X/yoso-normal-v1-8-1` Apache-2.0. This repository contains no upstream
code and no weights.
