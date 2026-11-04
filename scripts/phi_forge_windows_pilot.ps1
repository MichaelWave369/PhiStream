# Φ Model Forge R0.3. Windows local stages; check (read-only) by default.
[CmdletBinding()]
param(
    [ValidateSet('check','cuda','curate','audit-tokens','train','eval-base','eval-adapter','blind-review')]
    [string]$Stage = 'check',
    [string]$Python = 'python',
    [string]$Dataset = '',
    [string]$OutputRoot = '',
    [string]$Model = 'Qwen/Qwen3-4B-Instruct-2507',
    [string]$Revision = '',
    [ValidateRange(1,300)][int]$Steps = 8,
    [ValidateRange(128,1024)][int]$MaxLength = 256,
    [switch]$ApproveTraining,
    [switch]$AllowModelDownload
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $OutputRoot) { $OutputRoot = Join-Path $HOME 'phi-model-forge' }
if (-not $Dataset) { $Dataset = Join-Path $OutputRoot 'reviewed-r03.jsonl' }
$Seed = Join-Path $Repo 'model_forge\seed_examples.jsonl'
$Adapter = Join-Path $OutputRoot 'adapter-r03'
$BaseReceipt = Join-Path $OutputRoot 'base-holdout-r03.json'
$AdapterReceipt = Join-Path $OutputRoot 'adapter-holdout-r03.json'
$BlindPack = Join-Path $OutputRoot 'blind-review-r03.json'
$BlindKey = Join-Path $OutputRoot 'blind-review-key-r03.json'

if ($Stage -eq 'train' -and -not $ApproveTraining) {
    throw 'Training requires -ApproveTraining, not merely a reviewed dataset.'
}
if ($AllowModelDownload -and $Stage -notin @('audit-tokens','train','eval-base','eval-adapter')) {
    throw 'Model download permission applies only to an explicit model-run stage.'
}
if ($Stage -notin @('check','cuda','curate') -and -not (Test-Path -LiteralPath $Dataset -PathType Leaf)) {
    throw "Curated dataset missing: $Dataset. Run -Stage curate first."
}
$ArgsList = @()
switch ($Stage) {
    'check' {
        $Candidate = if (Test-Path -LiteralPath $Dataset -PathType Leaf) { $Dataset } else { $Seed }
        $ArgsList = @('-m','phistream.forge_preflight','--dataset',$Candidate)
    }
    'cuda' {
        $Candidate = if (Test-Path -LiteralPath $Dataset -PathType Leaf) { $Dataset } else { $Seed }
        $ArgsList = @('-m','phistream.forge_preflight','--dataset',$Candidate,'--probe-cuda')
    }
    'curate' {
        $ArgsList = @('-m','phistream.forge_curate','--dataset',$Seed,'--output',$Dataset)
    }
    'audit-tokens' {
        $ArgsList = @('-m','phistream.forge_token_audit','--dataset',$Dataset,
            '--model',$Model,'--max-length',"$MaxLength")
    }
    'train' {
        $ArgsList = @('-m','scripts.phi_model_train','--dataset',$Dataset,
            '--model',$Model,'--output',$Adapter,'--steps',"$Steps",
            '--max-length',"$MaxLength",'--execute')
        if ($Model -ne 'Qwen/Qwen3-4B-Instruct-2507') { $ArgsList += '--allow-model-override' }
    }
    'eval-base' {
        $ArgsList = @('-m','scripts.phi_model_eval','--mode','base',
            '--model',$Model,'--dataset',$Dataset,'--output',$BaseReceipt)
    }
    'eval-adapter' {
        if (-not (Test-Path -LiteralPath $Adapter -PathType Container)) {
            throw "LoRA adapter directory is missing: $Adapter"
        }
        $ArgsList = @('-m','scripts.phi_model_eval','--mode','adapter',
            '--model',$Model,'--adapter',$Adapter,'--dataset',$Dataset,'--output',$AdapterReceipt)
    }
    'blind-review' {
        foreach ($Needed in @($BaseReceipt,$AdapterReceipt)) {
            if (-not (Test-Path -LiteralPath $Needed -PathType Leaf)) {
                throw "Missing local evaluation: $Needed"
            }
        }
        $ArgsList = @('-m','phistream.forge_review_pack','--dataset',$Dataset,
            '--base',$BaseReceipt,'--adapter',$AdapterReceipt,'--pack',$BlindPack,'--key',$BlindKey)
    }
}
if ($Revision -and $Stage -in @('audit-tokens','train','eval-base','eval-adapter')) { $ArgsList += @('--revision',$Revision) }
if ($AllowModelDownload) { $ArgsList += '--allow-download' }
Write-Host "Model Forge local stage: $Stage"
Write-Host 'Stages are independent. No automatic model training or publishing.'
Push-Location $Repo
try {
    & $Python @ArgsList
    if ($LASTEXITCODE -ne 0) { throw "Model Forge stage $Stage returned $LASTEXITCODE" }
} finally {
    Pop-Location
}
