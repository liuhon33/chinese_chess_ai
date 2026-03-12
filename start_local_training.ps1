param(
    [switch]$NewRun
)

$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = 'C:\Users\Hongrui_Liu\.conda\envs\pytorch_learn\python.exe'
$ConfigType = 'local_torch'
$DataDir = Join-Path $RepoRoot 'mydata'
$Gpu = '0'
$PowerShellExe = 'powershell.exe'

# Edit the values above if you want a different Python executable, data directory, config, or GPU.
# First run:   .\start_local_training.ps1 -NewRun
# Resume run:  .\start_local_training.ps1

New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

function New-WindowCommand {
    param(
        [string]$Title,
        [string[]]$Arguments
    )

    $quotedArgs = $Arguments | ForEach-Object {
        '"{0}"' -f ($_ -replace '"', '`"')
    }
    $pythonCommand = "& '$Python' $($quotedArgs -join ' ')"
    return "Set-Location '$RepoRoot'; `$Host.UI.RawUI.WindowTitle = '$Title'; $pythonCommand"
}

$selfArgs = @('cchess_alphazero/run.py', 'self', '--type', $ConfigType, '--gpu', $Gpu, '--data-dir', $DataDir)
if ($NewRun) {
    $selfArgs += '--new'
}

$workers = @(
    @{ Title = 'cchess self'; Args = $selfArgs },
    @{ Title = 'cchess opt'; Args = @('cchess_alphazero/run.py', 'opt', '--type', $ConfigType, '--gpu', $Gpu, '--data-dir', $DataDir) },
    @{ Title = 'cchess eval'; Args = @('cchess_alphazero/run.py', 'eval', '--type', $ConfigType, '--gpu', $Gpu, '--data-dir', $DataDir) }
)

foreach ($worker in $workers) {
    $command = New-WindowCommand -Title $worker.Title -Arguments $worker.Args
    Start-Process -FilePath $PowerShellExe -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $command) | Out-Null
}