# run.ps1 — ответы с источниками, цитатами и режимом «не знаю» (день 24).
#
#   .\run.ps1                                   # калибровка порогов + проверка на 14 вопросах
#   .\run.ps1 -NoJudge                          # то же без LLM-судьи (дешевле)
#   .\run.ps1 -Mode ask -Query "Какие слои памяти в дне 11?"
#   .\run.ps1 -Mode chat                        # диалог, /trace — как отработал поиск
#
# Ключ AITunnel спросит со скрытым вводом. Модель по умолчанию — gpt-5.6-luna.
# Если Windows блокирует запуск:  powershell -ExecutionPolicy Bypass -File .\run.ps1

param(
    [ValidateSet("eval", "calibrate", "ask", "chat")]
    [string]$Mode = "eval",
    [string]$Model = "gpt-5.6-luna",
    [ValidateSet("aitunnel", "local")]
    [string]$Backend = "aitunnel",
    [string]$Query = "",
    [switch]$NoJudge
)

Set-Location $PSScriptRoot

function Find-Python {
    foreach ($candidate in @("py", "python",
        "$env:LOCALAPPDATA\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\python.exe")) {
        try {
            $output = & $candidate --version 2>&1
            if ($output -match "Python \d") { return $candidate }
        }
        catch { }
    }
    return $null
}

$py = Find-Python
if (-not $py) {
    Write-Host "Python не найден. Установите Python 3 с python.org или из Microsoft Store." -ForegroundColor Red
    exit 1
}
Write-Host "Использую Python: $py" -ForegroundColor Green

Write-Host "Устанавливаю зависимости..." -ForegroundColor Cyan
& $py -m pip install -r requirements.txt --quiet --disable-pip-version-check

if (-not ($Backend -eq "local" -and $Mode -eq "calibrate") -and -not $env:AITUNNEL_API_KEY) {
    $secureKey = Read-Host "Введите ваш AITUNNEL_API_KEY" -AsSecureString
    $env:AITUNNEL_API_KEY = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
        [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    )
}

switch ($Mode) {
    "calibrate" { & $py .\calibrate.py --backend $Backend }
    "eval" {
        Write-Host "`n=== Шаг 1: подбор порогов (фильтр и «не знаю») ===" -ForegroundColor Cyan
        & $py .\calibrate.py --backend $Backend; if ($LASTEXITCODE -ne 0) { exit 1 }
        Write-Host "`n=== Шаг 2: проверка ответов на 14 вопросах ===" -ForegroundColor Cyan
        if ($NoJudge) { & $py .\evaluate.py --backend $Backend --model $Model --no-judge }
        else          { & $py .\evaluate.py --backend $Backend --model $Model }
    }
    "ask" {
        if (-not $Query) { Write-Host "Укажите -Query ""вопрос""" -ForegroundColor Yellow; exit 1 }
        & $py .\calibrate.py --backend $Backend | Out-Null
        & $py .\agent.py --backend $Backend --model $Model --ask $Query
    }
    "chat" {
        & $py .\calibrate.py --backend $Backend | Out-Null
        & $py .\agent.py --backend $Backend --model $Model
    }
}
