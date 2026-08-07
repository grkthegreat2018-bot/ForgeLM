#Requires -Version 5.1
<#
.SYNOPSIS
    Create a Python 3.11+ virtual environment and install ForgeAI research dependencies.
.DESCRIPTION
    Picks the best available Python interpreter (py launcher preferred), installs
    PyTorch with CUDA 12.8 support for Blackwell/RTX 5070, then installs the rest
    of the requirements and applies the Triton Windows path-length patch.
#>
param(
    [string]$VenvDir = "$PSScriptRoot\venv",
    [string]$PythonCmd = "py -3.13"
)

$ErrorActionPreference = "Stop"

function Test-Command {
    param([string]$Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Prefer the system Python 3.13 if `py` is available; otherwise fall back to `python`.
if (-not (Test-Command "py")) {
    $PythonCmd = "python"
}

Write-Host "Using Python command: $PythonCmd" -ForegroundColor Cyan
& $PythonCmd --version

# Create virtual environment.
if (Test-Path $VenvDir) {
    Write-Host "Virtual environment already exists at $VenvDir. Removing old env..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $VenvDir
}
Write-Host "Creating virtual environment at $VenvDir ..." -ForegroundColor Cyan
& $PythonCmd -m venv $VenvDir

$pip = Join-Path $VenvDir "Scripts\pip.exe"

# Upgrade pip/setuptools/wheel first.
& $pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.8 and the rest of the project dependencies.
& $pip install -r "$PSScriptRoot\requirements.txt"

$python = Join-Path $VenvDir "Scripts\python.exe"

# Validate core imports.
$validationScript = @'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
try:
    import bitsandbytes as bnb
    print("bitsandbytes version:", bnb.__version__)
except Exception as e:
    print("bitsandbytes not available (training will fall back to AdamW):", e)
'@
$validationScript | & $python -

# Patch Triton's cache file so torch.compile works on Windows.
$patchScript = @'
import inspect
import os
import sys
try:
    import triton.runtime.cache as tc
    import uuid
    src = inspect.getsourcefile(tc.FileCacheManager)
    if src and os.path.exists(src):
        with open(src, "r") as f:
            text = f.read()
        new_text = text
        new_text = new_text.replace("rnd_id = str(uuid.uuid4())", "rnd_id = str(uuid.uuid4())[:8]")
        new_text = new_text.replace('pid = os.getpid()\n        # use temp dir', '# use temp dir')
        new_text = new_text.replace('f"tmp.pid_{pid}_{rnd_id}"', 'f"tmp.{rnd_id}"')
        if new_text != text:
            with open(src, "w") as f:
                f.write(new_text)
            print("Patched triton cache.py for Windows MAX_PATH")
except Exception as e:
    print("Could not patch triton cache:", e)
'@
$patchScript | & $python -

Write-Host "`nSetup complete. Activate with: .\venv\Scripts\Activate.ps1" -ForegroundColor Green
