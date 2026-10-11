# PhiStream device validation: local-only, no model downloads or auto-installs.
# Run from the PhiStream repository root using PowerShell.
# Requirements: Python >=3.10, numpy, PhiStream installed editable, official CUDA-enabled PyTorch.
param(
    [string]$Python = "python",
    [ValidateSet("cuda", "cpu")][string]$Device = "cuda",
    [string]$Fixture = "artifacts\gpu-fixture",
    [string]$OutputFolder = "benchmarks\local"
)
$ErrorActionPreference = "Stop"

function Invoke-PhiPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python validation failed (exit $LASTEXITCODE): $($Arguments -join ' ')"
    }
}

Write-Host "ΦStream device validation on this machine" -ForegroundColor Cyan
Write-Host "No downloads, installs, network uploads or repository pushes."
Invoke-PhiPython -Arguments @("-m", "phistream.cli", "doctor", "--device", $Device)
if (-not (Test-Path -LiteralPath $Fixture)) {
    Invoke-PhiPython -Arguments @("-m", "phistream.cli", "init-fixture", $Fixture)
}
Invoke-PhiPython -Arguments @("-m", "phistream.cli", "verify", $Fixture)
New-Item -ItemType Directory -Force -Path $OutputFolder | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$target = Join-Path $OutputFolder "phistream-$Device-$stamp.json"
if (Test-Path -LiteralPath $target) {
    throw "Refusing to overwrite $target"
}
Invoke-PhiPython -Arguments @("-m", "phistream.cli", "validate-device",
    $Fixture, "--device", $Device, "--tokens", "8", "--pin-mib", "0.125",
    "--output", $target)
Write-Host "Receipt saved locally: $target" -ForegroundColor Green
Write-Host "Import it from disk via the PhiStream web dashboard. Nothing is uploaded."
