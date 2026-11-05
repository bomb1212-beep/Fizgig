param(
    [ValidateSet('doctor','prepare','caption','plan','run','status')]
    [string]$Command = 'doctor',
    [string]$Job = ''
)
$ErrorActionPreference = 'Stop'
$automationRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
$runnerArgs = @((Join-Path $PSScriptRoot 'runner.py'), $Command)
if ($Job) { $runnerArgs += @('--job', $Job) }
& (Join-Path $automationRoot 'venv\Scripts\python.exe') @runnerArgs
exit $LASTEXITCODE
