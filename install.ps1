# SPDX-License-Identifier: MIT
<#
.SYNOPSIS
    Hi3DGen (Stable3DGen) on Strix Halo (gfx1151 / Windows / ROCm) - one-command install.

.DESCRIPTION
    Creates a dedicated virtual environment, installs ROCm PyTorch, clones the
    upstream Stable3DGen repository at a pinned commit, downloads the weights and
    writes a .env file.

    **No CUDA-only package is installed.** spconv and xformers are replaced at
    launch time by pure-torch shims that live in runners/hi3dgen/. Upstream code
    is never patched.

.PARAMETER Root
    Where the virtual environment, the upstream clone and the weights go.
    Defaults to the parent of this repository.

.EXAMPLE
    .\install.ps1
    .\install.ps1 -Root D:\models\hi3dgen
#>
[CmdletBinding()]
param(
    # Where the virtual environment, the upstream clone and the weights go.
    # Empty means: next to this repository, in hi3dgen-strix-halo-data.
    [string]$Root = "",
    [string]$Python = "py -3.12"
)

# Native tools (git, pip) report progress on stderr. Under output redirection,
# Windows PowerShell 5.1 turns those lines into error records, and a "Stop"
# preference would kill the script on the first one. So the preference stays
# "Continue" and every native step is checked through its exit code instead.
$ErrorActionPreference = "Continue"
function Assert-Ok([string]$step) {
    if ($LASTEXITCODE) { throw "$step failed with exit code $LASTEXITCODE" }
}

# $PSScriptRoot can be empty while param defaults are evaluated under
# Windows PowerShell 5.1, so the paths are resolved here instead.
$repo = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $Root) { $Root = Join-Path (Split-Path -Parent $repo) "hi3dgen-strix-halo-data" }

# Pinned versions. Do not float these: the ROCm wheels and the upstream commit
# are the two things that decide whether this works at all.
# The device packages matter: torch ships its GPU kernels as external kernel
# packs, and **both** the family (gfx115x: flash-attention images) and the
# exact-arch (gfx1151: torch kernel pack + ROCm library kernels) wheels are
# required. Without the exact-arch one, the first kernel launch fails with
# `hipErrorInvalidImage`. On another GPU, replace both suffixes with yours.
$TorchIndex = "https://stable.repo.amd.com/rocm/whl-next/"
$TorchVersion = "2.13.0+rocm10.0.0"
$TorchvisionVersion = "0.28.0+rocm10.0.0"
$TorchDeviceWheels = @(
    "amd-torch-device-gfx115x==$TorchVersion",
    "amd-torch-device-gfx1151==$TorchVersion",
    "amd-torchvision-device-gfx1151==$TorchvisionVersion"
)
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
    Assert-Ok "virtual environment creation"
}
& $py -m pip install --upgrade pip
Assert-Ok "pip upgrade"

# 2. ROCm PyTorch -------------------------------------------------------------
# torch pulls the `rocm` runtime packages from the same index; PyPI stays as a
# fallback for the pure-python dependencies only (the exact +rocm pins can
# never match anything on PyPI).
Write-Host "==> Installing ROCm PyTorch"
& $py -m pip install --no-cache-dir --index-url $TorchIndex `
    --extra-index-url https://pypi.org/simple `
    "torch==$TorchVersion" "torchvision==$TorchvisionVersion" @TorchDeviceWheels
Assert-Ok "PyTorch installation"

# 3. Upstream repository (never forked, never patched) ------------------------
if (-not (Test-Path $upstream)) {
    Write-Host "==> Cloning upstream Stable3DGen"
    # A shallow clone: a full history can stall for minutes in server-side pack
    # preparation. The pinned commit is fetched right below, also shallow.
    git clone --depth 1 $UpstreamUrl $upstream 2>&1 | Out-Host
    Assert-Ok "git clone"
}
Push-Location $upstream
git fetch --depth 1 origin $UpstreamCommit 2>&1 | Out-Host
if ($LASTEXITCODE) { Pop-Location; throw "git fetch failed ($LASTEXITCODE)" }
git checkout $UpstreamCommit 2>&1 | Out-Host
if ($LASTEXITCODE) { Pop-Location; throw "git checkout failed ($LASTEXITCODE)" }
git submodule update --init --recursive 2>&1 | Out-Host
if ($LASTEXITCODE) { Pop-Location; throw "git submodule update failed ($LASTEXITCODE)" }
Pop-Location

# 4. Pure-python dependencies -------------------------------------------------
Write-Host "==> Installing dependencies"
& $py -m pip install --no-cache-dir -r (Join-Path $repo "requirements.txt")
Assert-Ok "dependency installation"

# 5. Weights ------------------------------------------------------------------
Write-Host "==> Downloading weights (about 5.4 GB)"
foreach ($r in $WeightsRepos) {
    $name = $r.Split("/")[-1]
    & $py -c "from huggingface_hub import snapshot_download; snapshot_download('$r', local_dir=r'$weights\$name')"
    Assert-Ok "weights download ($r)"
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
Assert-Ok "shim verification"
& $py (Join-Path $repo "tests\test_drop_parts.py")
Assert-Ok "debris-filter verification"

Write-Host ""
Write-Host "Done. Generate a first mesh with:"
Write-Host "  $py $repo\tools\run_single.py --image $repo\assets\sample.png --out $Root\out"
Write-Host ""
Write-Host "Or point hearth at this checkout:"
Write-Host "  HEARTH_RUNNER_HI3DGEN_PYTHON=$py"
Write-Host "  HEARTH_RUNNER_HI3DGEN_MODULE=runners.hi3dgen"
Write-Host "  HEARTH_RUNNER_HI3DGEN_CWD=$repo"
