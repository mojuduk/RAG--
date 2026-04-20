param(
    [string]$CondaEnv = "nougat310",
    [string]$QaFile = "data/qa_gold_v23_merged.json",
    [string]$Model = "qwen3.5-plus",
    [string]$BaseUrl = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    [string]$ApiKeyEnv = "OPENAI_API_KEY",
    [int]$TopK = 5,
    [int]$MaxNewTokens = 120,
    [switch]$RebuildIndex
)

$ErrorActionPreference = "Stop"

function Require-File([string]$PathValue) {
    if (-not (Test-Path $PathValue)) {
        throw "File not found: $PathValue"
    }
}

function Require-Env([string]$Name) {
    $value = [Environment]::GetEnvironmentVariable($Name, "Process")
    if (-not $value) { $value = [Environment]::GetEnvironmentVariable($Name, "User") }
    if (-not $value) { $value = [Environment]::GetEnvironmentVariable($Name, "Machine") }
    if (-not $value) {
        throw "Environment variable `$${Name} is not set."
    }
}

Require-File $QaFile
Require-Env $ApiKeyEnv

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$summary = "data/eval_summary_${Model}_cloud_${ts}.csv"
$details = "data/eval_details_${Model}_cloud_${ts}.csv"

Write-Host "[INFO] Conda env: $CondaEnv"
Write-Host "[INFO] QA file:    $QaFile"
Write-Host "[INFO] Model:      $Model"
Write-Host "[INFO] Base URL:   $BaseUrl"
Write-Host "[INFO] Summary:    $summary"
Write-Host "[INFO] Details:    $details"

if ($RebuildIndex) {
    Write-Host "[INFO] Rebuilding TF-IDF and vector indexes..."
    conda run -n $CondaEnv python main.py index
    conda run -n $CondaEnv python main.py vindex
}

Write-Host "[INFO] Running cloud eval..."
conda run -n $CondaEnv python main.py eval `
    --qa-file $QaFile `
    --top-k $TopK `
    --llm-provider openai `
    --openai-model $Model `
    --openai-base-url $BaseUrl `
    --openai-api-key-env $ApiKeyEnv `
    --max-new-tokens $MaxNewTokens `
    --out-summary $summary `
    --out-details $details

Write-Host "[DONE] Evaluation finished."
Write-Host "[DONE] Summary CSV: $summary"
Write-Host "[DONE] Details CSV: $details"
