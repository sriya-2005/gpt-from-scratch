# Starts the laptop English stage (train.py --preset english_cpu) as a separate background
# process, so it keeps running after you close VS Code or the terminal. If a checkpoint from an
# earlier chunk exists it resumes from it; each chunk stops by itself after 5 hours.
#
# Usage (from the medhas folder):   powershell -ExecutionPolicy Bypass -File scripts\run_english_cpu.ps1
# Watch progress:                   Get-Content logs\english_cpu_train.log -Wait
# Stop it early:                    Stop-Process -Id (Get-Content logs\english_cpu_train.pid)

$root = Split-Path $PSScriptRoot -Parent
$python = Join-Path $root ".venv\Scripts\python.exe"
$log = Join-Path $root "logs\english_cpu_train.log"

$trainArgs = @("scripts\train.py", "--preset", "english_cpu")
if (Test-Path (Join-Path $root "models\checkpoints\english_cpu\latest.pt")) {
    $trainArgs += "--resume"
}

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$proc = Start-Process -FilePath $python -ArgumentList $trainArgs -WorkingDirectory $root -WindowStyle Hidden `
    -RedirectStandardOutput $log -RedirectStandardError "$log.progress" -PassThru
$proc.Id | Set-Content (Join-Path $root "logs\english_cpu_train.pid")
Write-Output "Started training (process $($proc.Id)) with: $($trainArgs -join ' ')"
Write-Output "Log: $log"
