<#
  Full final-phase experiment grid. Two streams run concurrently (8-core CPU, 4 threads each).
  Every script is resumable, so re-running this after an interruption only does the missing pieces.

      powershell -ExecutionPolicy Bypass -File experiments\run_all.ps1
      powershell -ExecutionPolicy Bypass -File experiments\run_all.ps1 -Seeds "0 1 2" -Rounds 20

  Approximate CPU cost (this machine): ~6-7 h sequential, ~4 h with the two streams.
  Stream A: baselines -> ablation (A/B/C/D + references, all seeds) -> conformal -> report
  Stream B: attacks -> epsilon sweep -> alpha sweep -> robustness -> flower demo
#>
param([string]$Seeds = "0 1 2", [int]$Rounds = 20, [int]$RobustRounds = 10)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONWARNINGS = "ignore"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
New-Item -ItemType Directory -Force "results\logs" | Out-Null
$seedArgs = $Seeds.Split(" ")

$streamA = {
    param($Root, $seedArgs, $Rounds)
    Set-Location $Root; $env:PYTHONWARNINGS = "ignore"; $env:OMP_NUM_THREADS = "4"
    python -u experiments/run_baselines.py --seeds @seedArgs --epochs 15 *>> results\logs\streamA.log
    python -u experiments/run_ablation.py --group ablation --seeds @seedArgs --rounds $Rounds *>> results\logs\streamA.log
    python -u experiments/run_federated.py --config experiments/configs/full_system_clipped.yaml --seeds @seedArgs --rounds $Rounds *>> results\logs\streamA.log
    python -u experiments/run_conformal.py *>> results\logs\streamA.log
    "STREAM A DONE $(Get-Date)" >> results\logs\streamA.log
}
$streamB = {
    param($Root, $seedArgs, $Rounds, $RobustRounds)
    Set-Location $Root; $env:PYTHONWARNINGS = "ignore"; $env:OMP_NUM_THREADS = "4"
    python -u experiments/run_attacks.py --shadows 8 *>> results\logs\streamB.log
    python -u experiments/run_ablation.py --group epsilon --seeds $seedArgs[0] --rounds $Rounds *>> results\logs\streamB.log
    python -u experiments/run_ablation.py --group alpha --seeds $seedArgs[0] --rounds $Rounds *>> results\logs\streamB.log
    python -u experiments/run_robustness.py --rounds $RobustRounds --seeds $seedArgs[0] *>> results\logs\streamB.log
    python -u experiments/run_flower.py --rounds 5 *>> results\logs\streamB.log
    "STREAM B DONE $(Get-Date)" >> results\logs\streamB.log
}

$jobA = Start-Job -ScriptBlock $streamA -ArgumentList $Root, $seedArgs, $Rounds
$jobB = Start-Job -ScriptBlock $streamB -ArgumentList $Root, $seedArgs, $Rounds, $RobustRounds
Write-Host "Started stream A (job $($jobA.Id)) and stream B (job $($jobB.Id)). Logs: results\logs\stream{A,B}.log"
Wait-Job $jobA, $jobB | Out-Null
Receive-Job $jobA, $jobB | Out-Null
python -u experiments/make_results_report.py
Write-Host "ALL DONE $(Get-Date)"
