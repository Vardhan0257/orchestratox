# run_v2.ps1 -- pre-registered runs on frozen bank v2 (27 scenarios).
# One call = one model x one payload style. Env vars are set only for this call.
#
# Examples (run each as its own command):
#   .\run_v2.ps1 -Model gpt-5.6-luna -Style explicit -Effort none -Depths 0,3 -MaxUsd 0.20
#   .\run_v2.ps1 -Model gpt-5.6-luna -Style policy   -Effort none -Depths 3   -MaxUsd 0.20
# D0 (clean) does not depend on the payload style, so run it ONCE per model (with the first style).
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [Parameter(Mandatory = $true)][ValidateSet("explicit", "policy")][string]$Style,
    [string]$Depths = "0,3",
    [int]$Trials = 3,
    [double]$MaxUsd = 0,
    [string]$Effort = "minimal",
    [switch]$Pilot
)
try {
    $env:SCENARIO_BANK  = "results/scenario_bank_v2_frozen.json"
    $env:FULL_SCENARIOS = "27"
    $env:FULL_TRIALS    = "$Trials"
    $env:DEPTHS_RUN     = $Depths
    $env:PAYLOAD_STYLE  = $Style
    $env:RUN_TAG        = "_v2_$Style"
    $p = @{ Models = @($Model); Effort = $Effort }
    if ($MaxUsd -gt 0) { $p.MaxUsd = $MaxUsd }
    if ($Pilot) { $p.Pilot = $true }
    & .\run_resilient.ps1 @p
}
finally {
    Remove-Item Env:SCENARIO_BANK, Env:FULL_SCENARIOS, Env:FULL_TRIALS, Env:DEPTHS_RUN, Env:PAYLOAD_STYLE, Env:RUN_TAG -ErrorAction SilentlyContinue
}
