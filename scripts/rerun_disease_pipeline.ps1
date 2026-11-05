$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = (Get-Command python -ErrorAction Stop).Source
}

function Invoke-Python {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

function Invoke-AnnualRun {
    param([string]$Tag, [string[]]$Options = @())
    $pythonArgs = @("models/stgnn/annual.py", "--tag", $Tag) + $Options
    $logPath = Join-Path $repoRoot "logs\$Tag.log"
    Write-Host "=== ST-GNN $Tag ==="
    & $python @pythonArgs *> $logPath
    if ($LASTEXITCODE -ne 0) {
        throw "ST-GNN run '$Tag' failed. See $logPath"
    }
    Get-Content $logPath -Tail 4
}

New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot "logs") | Out-Null

$pipeline = @(
    @("src/ingestion/disease/build_pan_india_state_tensor.py"),
    @("src/ingestion/disease/build_macro_checksum.py"),
    @("src/ingestion/weather/build_weather_tensor.py"),
    @("src/ingestion/weather/build_lag_features.py"),
    @("src/ingestion/disease/audit_zeros.py"),
    @("src/ingestion/disease/build_clean_tensor.py"),
    @("src/ingestion/disease/coverage_by_year.py"),
    @("src/models/baselines/train_baselines.py"),
    @("src/models/baselines/visualize_baselines.py")
)

foreach ($arguments in $pipeline) {
    Write-Host "=== $($arguments[0]) ==="
    Invoke-Python -Arguments $arguments
}

$experiments = @(
    @{ Tag = "geo_w"; Options = @() },
    @{ Tag = "geo_now"; Options = @("--weather", "0") },
    @{ Tag = "identity"; Options = @("--graph", "identity") },
    @{ Tag = "knn"; Options = @("--graph", "knn") },
    @{ Tag = "dist"; Options = @("--graph", "dist") },
    @{ Tag = "hybrid"; Options = @("--graph", "hybrid") },
    @{ Tag = "gat"; Options = @("--kind", "gat") },
    @{ Tag = "single"; Options = @("--single", "1") },
    @{ Tag = "K2"; Options = @("--K", "2") },
    @{ Tag = "K5"; Options = @("--K", "5") },
    @{ Tag = "horizons"; Options = @("--horizons", "1,2,3") }
)

foreach ($experiment in $experiments) {
    Invoke-AnnualRun -Tag $experiment.Tag -Options $experiment.Options
}

Invoke-AnnualRun -Tag "clean" -Options @("--data", "data/processed/pan_india_state_enriched_tensor_clean.csv")

Invoke-Python -Arguments @(
    "models/stgnn/compare_runs.py",
    "geo_w", "geo_now", "identity", "knn", "dist", "hybrid", "gat", "single", "K2", "K5",
    "horizons@2", "horizons@3"
)

Invoke-AnnualRun -Tag "final" -Options @("--forecast", "1", "--horizons", "1,2,3")
Invoke-Python -Arguments @("models/stgnn/predict.py")
Invoke-Python -Arguments @("scripts/make_registry.py")
Invoke-Python -Arguments @("-m", "src.risk.calibrate")
Invoke-Python -Arguments @("-m", "src.api.ingest")
Invoke-Python -Arguments @("-m", "src.api.export_examples")
Invoke-Python -Arguments @("-m", "scripts.make_mocks")
Invoke-Python -Arguments @("src/monitoring/drift.py", "--out", "frontend/mocks/drift_report.json")
Invoke-Python -Arguments @("src/monitoring/drift.py", "--out", "data/processed/drift_report.json")

Write-Host "Disease pipeline rerun completed through the latest year in the source CSVs."
