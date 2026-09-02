# Where the GPU time goes in one generation

Measured with [`tools/profile_gemm.py`](../tools/profile_gemm.py) on gfx1151
(Radeon 8060S), Windows 11, ROCm 7.2.1, torch 2.9.1+rocm7.2.1, clock keepalive
on, fast attention on, 2026-09-02. Reference fp16 GEMM taken alongside:
21–24 TFLOPS at 2048³, 29–31 TFLOPS at 4096³ (rocBLAS). Sample:
`assets/sample.png`, upstream defaults (`ss_steps=50`, `slat_steps=6`),
28 605 active voxels.

Shares are of profiled device time; walls are from an unprofiled run. This
torch build has no Kineto, so shares are decision-grade rather than exact.

| Stage | Wall | GEMM | Attention | Other |
|---|--:|--:|--:|--:|
| structure (dominant) | 44.1 s | 11.6 s (25 %) | 24.8 s (54 %) | 9.6 s (21 %) |
| slat | 17.1 s | 4.8 s (28 %) | 3.4 s (20 %) | 9.1 s (53 %) |
| decode | 3.5 s | 0.5 s (10 %) | 1.1 s | 3.6 s |

**The dominant stage is attention-bound, not GEMM-bound**: 4096 tokens through
flash attention, 100 forward passes (50 steps × CFG). The largest GEMMs, all
fp16:

| Role | M | N | K | Calls | rocBLAS TFLOPS |
|---|--:|--:|--:|--:|--:|
| MLP down (structure) | 4096 | 1024 | 4096 | 2112 | 18.7 |
| MLP up (structure) | 4096 | 4096 | 1024 | 2112 | 25.3 |
| QKV (structure) | 4096 | 3072 | 1024 | 2112 | 31.1 |
| Out proj (structure) | 4096 | 1024 | 1024 | 6336 | 29.6 |
| **Sparse-conv shim (slat)** | **28605** | **128** | **2048** | **297** | **1.5** |
| Decode | 65536 | 96–192 | 96–768 | ~1600 | 11–20 |

The one pathological entry is the sparse-convolution projection: skinny
(N = 128), and rocBLAS runs it at 1.5 TFLOPS — 2.9 s of the slat stage on its
own.

## With hipBLASLt (now the default)

Same measurement with `TORCH_BLAS_PREFER_HIPBLASLT=1` and
`ROCBLAS_USE_HIPBLASLT=1` (what `HI3DGEN_PREFER_HIPBLASLT=on` sets):

| Stage | rocBLAS | hipBLASLt | Speedup |
|---|--:|--:|--:|
| structure | 44.1 s | 44.0 s | 1.00× (attention-bound) |
| slat | 17.1 s | **10.7 s** | **1.60×** |
| whole generation | 69.7 s | 62.9 s | 1.11× |

The gain is almost entirely the skinny sparse-conv GEMM, which hipBLASLt runs
at 14 TFLOPS instead of 1.5. `metrics.blas_backend` records which backend a
run used.

Hi3DGen shares its architecture (and therefore these shapes, modulo the voxel
count) with TRELLIS; Hunyuan3D looks completely different. The three-pipeline
comparison, the shape-overlap analysis, and everything about this GPU that
does not depend on the model live in
[gfx1151-gemm](https://github.com/kroqueta-s/gfx1151-gemm).
