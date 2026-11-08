param(
    [string]$Python = "py",
    [string]$Cache = (Join-Path $PSScriptRoot "models"),
    [switch]$WithTripoSR,
    [switch]$WithSAM2,
    [switch]$SkipWeights
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Run-Native {
    param([string]$Exe, [string[]]$Arguments)
    $PreviousPreference = $ErrorActionPreference
    try {
        # Windows PowerShell 5 trata stderr redirecionado como ErrorRecord,
        # inclusive avisos legítimos de pip. O código de saída decide a falha.
        $ErrorActionPreference = "Continue"
        & $Exe @Arguments 2>&1 | ForEach-Object { Write-Host "$_" }
        $NativeExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousPreference
    }
    if ($NativeExitCode -ne 0) { throw "Falha ao executar $Exe (código $NativeExitCode). Consulte a saída acima." }
}

try {
    if (-not (Test-Path .venv/Scripts/python.exe)) {
        if ($Python -eq "py") {
            Run-Native $Python @("-3.11", "-m", "venv", ".venv")
        } else {
            Run-Native $Python @("-m", "venv", ".venv")
        }
    }
    $VenvPython = Join-Path $PSScriptRoot ".venv/Scripts/python.exe"
    Run-Native $VenvPython @("-m", "ensurepip", "--upgrade")
    Run-Native $VenvPython @("-c", "import sys; assert sys.version_info[:2] in [(3,11),(3,12)], 'Use Python 3.11 ou 3.12'")
    Run-Native $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    # Família de wheels testável e consistente, com kernels Blackwell.
    Run-Native $VenvPython @("-m", "pip", "install", "torch==2.7.1", "torchvision==0.22.1", "--index-url", "https://download.pytorch.org/whl/cu128")
    Run-Native $VenvPython @("-m", "pip", "install", "-e", ".[ai,test]")
    $Cache = [IO.Path]::GetFullPath($Cache)
    $CodeRoot = Join-Path $Cache "code"
    New-Item -ItemType Directory -Force $CodeRoot | Out-Null
    $HunyuanCode = Join-Path $CodeRoot "Hunyuan3D-2"
    if (-not (Test-Path (Join-Path $HunyuanCode "setup.py"))) {
        Run-Native "git" @("clone", "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git", $HunyuanCode)
        Run-Native "git" @("-C", $HunyuanCode, "checkout", "f8db63096c8282cb27354314d896feba5ba6ff8a")
    }
    # Somente geometria: não compila rasterizadores/extensões de textura.
    Run-Native $VenvPython @("-m", "pip", "install", "--no-deps", "-e", $HunyuanCode)
    if ($WithTripoSR) {
        $TripoCode = Join-Path $CodeRoot "TripoSR"
        if (-not (Test-Path (Join-Path $TripoCode "tsr/system.py"))) {
            Run-Native "git" @("clone", "https://github.com/VAST-AI-Research/TripoSR.git", $TripoCode)
            Run-Native "git" @("-C", $TripoCode, "checkout", "107cefdc244c39106fa830359024f6a2f1c78871")
        }
        # Adapta um único import oficial para marching cubes CPU scikit-image.
        Run-Native $VenvPython @("-m", "img23d.vendor", "--cache", $Cache)
    }
    if ($WithSAM2) {
        Run-Native $VenvPython @("-m", "pip", "install", "-e", ".[sam2]")
        $SAMCode = Join-Path $CodeRoot "sam2"
        if (-not (Test-Path (Join-Path $SAMCode "setup.py"))) {
            Run-Native "git" @("clone", "https://github.com/facebookresearch/sam2.git", $SAMCode)
            Run-Native "git" @("-C", $SAMCode, "checkout", "2b90b9f5ceec907a1c18123530e92e794ad901a4")
        }
        $PreviousSAMBuild = $env:SAM2_BUILD_CUDA
        try {
            $env:SAM2_BUILD_CUDA = "0"
            Run-Native $VenvPython @("-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", $SAMCode)
        } finally {
            $env:SAM2_BUILD_CUDA = $PreviousSAMBuild
        }
        Run-Native $VenvPython @("-c", "from sam2.build_sam import build_sam2; from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator; print('Import SAM 2 OK')")
    }
    Run-Native $VenvPython @("-m", "img23d", "--check-gpu")
    Run-Native $VenvPython @("-c", "from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline; print('Import Hunyuan OK')")
    if ($WithTripoSR) {
        $env:IMG23D_CACHE = $Cache
        Run-Native $VenvPython @("-c", "import os,sys; from pathlib import Path; sys.path.insert(0,str(Path(os.environ['IMG23D_CACHE'])/'code'/'TripoSR')); from tsr.system import TSR; print('Import TripoSR OK')")
    }
    if (-not $SkipWeights) {
        $DownloadArgs = @("-m", "img23d.download", "--cache", $Cache)
        if ($WithTripoSR) { $DownloadArgs += "--triposr" }
        if ($WithSAM2) { $DownloadArgs += "--sam2" }
        Run-Native $VenvPython $DownloadArgs
    }
    # Pasta única evita ACLs de temporários criados por outra sessão/usuário.
    New-Item -ItemType Directory -Force (Join-Path $Cache "test-runs") | Out-Null
    $TestTemp = Join-Path $Cache ("test-runs/" + [guid]::NewGuid().ToString("N"))
    Run-Native $VenvPython @("-m", "pytest", "-q", "--basetemp", $TestTemp, "-p", "no:cacheprovider")
    Run-Native $VenvPython @("-m", "pip", "check")
    & $VenvPython -m pip freeze | Set-Content -Encoding UTF8 (Join-Path $Cache "installed-requirements.txt")
    Write-Host "Setup concluído. Cache: $Cache"
    Write-Host ".\.venv\Scripts\python.exe -m img23d --ui --cache `"$Cache`""
} catch {
    Write-Error "Setup interrompido: $_. Confira Python 3.11/3.12, Git, driver NVIDIA e a saída acima."
    exit 1
}
