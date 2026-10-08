# run_resilient.ps1
# Re-runs the experiment until it finishes. Safe because every trial is
# checkpointed; each restart skips completed trials.
#
# Usage:
#   .\run_resilient.ps1 -Models gpt-4.1 -MaxUsd 2.0
#   .\run_resilient.ps1 -Models gpt-5-mini,gpt-5.6-luna -MaxUsd 1.0 -Effort low
#   .\run_resilient.ps1 -Script .\impact3_ablation.py -Models gpt-4.1-mini -MaxUsd 0.40 -Extra "--trials","5"
#
# Exit codes from the Python script:
#   0 = finished   2 = out of credits / budget cap (do NOT auto-retry)
#   3 = bad API parameters (do NOT auto-retry)   other = crash/network (retry)

param(
    [string]$Script = ".\impact2_cross_model_transfer.py",
    [Parameter(Mandatory = $true)][string[]]$Models,
    [string[]]$Extra = @(),
    [double]$MaxUsd = 0,
    [string]$Effort = "minimal",
    [int]$MaxRestarts = 30,
    [switch]$Pilot
)

$argsList = @($Script) + $Models + @("--effort", $Effort) + $Extra
if ($MaxUsd -gt 0) { $argsList += @("--max-usd", $MaxUsd) }
if ($Pilot) { $argsList += "--pilot" }

for ($i = 1; $i -le $MaxRestarts; $i++) {
    Write-Host "=== Attempt $i of $MaxRestarts ($(Get-Date -Format s)) ==="
    python @argsList
    $code = $LASTEXITCODE

    if ($code -eq 0) { Write-Host "DONE."; exit 0 }
    if ($code -eq 2) { Write-Host "Stopped: credits or budget cap. Progress saved."; exit 2 }
    if ($code -eq 3) { Write-Host "Stopped: API rejected parameters. Fix and rerun."; exit 3 }

    Write-Host "Crashed (exit $code). Progress is saved. Retrying in 60s..."
    Start-Sleep -Seconds 60
}
Write-Host "Gave up after $MaxRestarts attempts. Progress is saved; rerun later."
exit 1
