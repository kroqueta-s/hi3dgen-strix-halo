# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
    Hi3DGen (Stable3DGen) on Strix Halo (gfx1151 / Windows / ROCm) - one-command install.

.DESCRIPTION
    Creates a dedicated virtual environment, installs ROCm PyTorch, clones the
    upstream Stable3DGen repository at a pinned commit, downloads the weights and
    writes a .env file.

    **No CUDA-only package is installed.** spconv and xformers are replaced at launch time by pure-torch shims that live in
    runners/hi3dgen/. Upstream code is never patched.

.PARAMETER Root
    Where the virtual environment, the upstream clone and the weights go.
    Defaults to the parent of this repository.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Root D:\models\trellis
#>
[CmdletBinding()]
param(
    [string]$Root = (Join-Path (Split-Path -Parent $PSScriptRoot) "hi3dgen-strix-halo-data"),
    [string]$Python = "py -3.12"
)

$ErrorActionPreference = "Stop"
$repo = $PSScriptRoot

# Pinned versions. Do not float these: the ROCm wheels and the upstream commit
# are the two things that decide whether this works at all.
$TorchIndex = "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/"
$TorchVersion = "2.9.1+rocm7.2.1"
$TorchvisionVersion = "0.24.1+rocm7.2.1"
$UpstreamUrl = "https://github.com/Stable-X/Stable3DGen.git"
$UpstreamCommit = "c29f668ecec44b197275e9bf77f823c0c8a21076"
$WeightsRepos = @("Stable-X/trellis-normal-v0-1", "ZhengPeng7/BiRefNet", "Stable-X/yoso-normal-v1-8-1")

$venv = Join-Path $Root ".venv"
$upstream = Join-Path $Root "Stable3DGen"
$weights = Join-Path $Root "weights"
$py = Join-Path $venv "Scripts\python.exe"

Write-Host "==> Root: $Root"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

# 1. Virtual environment ------------------------------------------------------
if (-not (Test-Path $py)) {
    Write-Host "==> Creating virtual environment"
    & cmd /c "$Python -m venv `"$venv`""
}
& $py -m pip install --upgrade pip

# 2. ROCm PyTorch -------------------------------------------------------------
# torch requires the `rocm` meta-package, which lives on the same index.
# Passing the wheel URL directly fails with "No matching distribution for rocm".
Write-Host "==> Installing ROCm PyTorch"
& $py -m pip install --no-cache-dir --find-links $TorchIndex `
    "torch==$TorchVersion" "torchvision==$TorchvisionVersion"

# 3. Upstream repository (never forked, never patched) ------------------------
if (-not (Test-Path $upstream)) {
    Write-Host "==> Cloning upstream Stable3DGen"
    git clone $UpstreamUrl $upstream
}
Push-Location $upstream
git fetch --depth 1 origin $UpstreamCommit
git checkout $UpstreamCommit
git submodule update --init --recursive
Pop-Location

# 4. Pure-python dependencies -------------------------------------------------
Write-Host "==> Installing dependencies"
& $py -m pip install --no-cache-dir -r (Join-Path $repo "requirements.txt")

# 5. Weights ------------------------------------------------------------------
Write-Host "==> Downloading weights (about 5.4 GB)"
foreach ($r in $WeightsRepos) {
    $name = $r.Split("/")[-1]
    & $py -c "from huggingface_hub import snapshot_download; snapshot_download('$r', local_dir=r'$weights\$name')"
}

# 6. .env ---------------------------------------------------------------------
$envPath = Join-Path $repo ".env"
if (-not (Test-Path $envPath)) {
    Write-Host "==> Writing .env"
    (Get-Content (Join-Path $repo ".env.example") -Raw).
        Replace("__REPO__", $upstream).
        Replace("__WEIGHTS__", $weights) | Set-Content -Path $envPath -Encoding utf8
}

# 7. Verify the shims before trusting any mesh --------------------------------
Write-Host "==> Verifying the shims (exact agreement with dense reference)"
& $py (Join-Path $repo "tests\test_shims.py")
& $py (Join-Path $repo "tests\test_raster.py")

Write-Host ""
Write-Host "Done. Point hearth at this checkout:"
Write-Host "  HEARTH_RUNNER_HI3DGEN_PYTHON=$py"
Write-Host "  HEARTH_RUNNER_HI3DGEN_MODULE=runners.hi3dgen"
Write-Host "  HEARTH_RUNNER_HI3DGEN_CWD=$repo"
