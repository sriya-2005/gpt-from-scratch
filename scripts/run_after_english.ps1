# Runs every stage after the laptop English stage, in order, as a separate background process:
#   2. story + fact statements, starting from the English model     (~40 min)
#   3. story-knowledge test                                         (~2 min)
#   4. question-answer fine-tuning                                  (~30 min)
#   5. diagnostics: training / reworded probe / validation questions, sentence completion
# Everything is written to logs\after_english.log. The test split is NOT used.
#
# Usage (from the medhas folder, after models\checkpoints\english_cpu\best.pt exists):
#   powershell -ExecutionPolicy Bypass -File scripts\run_after_english.ps1
# Watch:  Get-Content logs\after_english.log -Wait
param([switch]$Worker)  # internal: set when the script re-launches itself in the background

$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "logs\after_english.log"
$steps = @(
    "scripts\train.py --preset verdict_facts --init-from models\checkpoints\english_cpu\best.pt --run-name english_facts --epochs 40 --lr 1e-4",
    "scripts\eval_story.py --ckpt models\checkpoints\english_facts\best.pt --trained-frac 1.0",
    "scripts\finetune_qa.py --init-from models\checkpoints\english_facts\best.pt --rehearsal-data verdict_facts --run-name english_qa",
    "scripts\eval_qa.py --ckpt models\checkpoints\english_qa\best.pt --split train --quiet",
    "scripts\eval_qa.py --ckpt models\checkpoints\english_qa\best.pt --split probe --quiet",
    "scripts\eval_qa.py --ckpt models\checkpoints\english_qa\best.pt --split val --quiet",
    "scripts\eval_cloze.py --ckpt models\checkpoints\english_facts\best.pt models\checkpoints\english_qa\best.pt"
)

if ($Worker) {
    # Run the steps one after another; cmd's ">>" appends Python's UTF-8 output unchanged.
    Set-Location $root
    foreach ($step in $steps) {
        Add-Content -Path $log -Encoding UTF8 -Value "==== $(Get-Date -Format 'HH:mm')  python $step"
        cmd /c "`"$python`" $step >> `"$log`" 2>&1"
        if ($LASTEXITCODE -ne 0) {
            Add-Content -Path $log -Encoding UTF8 -Value "==== FAILED (exit code $LASTEXITCODE), stopping"
            exit 1
        }
    }
    Add-Content -Path $log -Encoding UTF8 -Value "==== ALL DONE $(Get-Date)"
    exit 0
}

if (-not (Test-Path (Join-Path $root "models\checkpoints\english_cpu\best.pt"))) {
    Write-Output "models\checkpoints\english_cpu\best.pt not found: finish the English stage first."
    exit 1
}
Set-Content -Path $log -Encoding UTF8 -Value "Started $(Get-Date)"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$proc = Start-Process powershell -WindowStyle Hidden -PassThru `
    -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"", "-Worker"
$proc.Id | Set-Content (Join-Path $root "logs\after_english.pid")
Write-Output "Started stages 2-5 (process $($proc.Id)). Log: $log"
