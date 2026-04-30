param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,

    [Parameter(Mandatory = $true)]
    [string]$WorkspaceRoot,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [Parameter(Mandatory = $true)]
    [string]$SummaryPath,

    [Parameter(Mandatory = $false)]
    [string]$LogPath = ""
)

$ErrorActionPreference = 'Stop'

if ($LogPath) {
    $logDir = Split-Path -Parent $LogPath
    if ($logDir) {
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    }
    "[$(Get-Date -Format s)] Waiting for PID $ProcessId" | Out-File -FilePath $LogPath -Encoding utf8 -Append
}

Wait-Process -Id $ProcessId

Set-Location $WorkspaceRoot

$pythonExe = Join-Path $WorkspaceRoot '.venv\Scripts\python.exe'
$scriptPath = Join-Path $WorkspaceRoot 'scripts\summarize_winner_only_benchmark_suite.py'

& $pythonExe $scriptPath --output-dir $OutputDir --summary-path $SummaryPath

if ($LASTEXITCODE -ne 0) {
    throw "Summary generation failed with exit code $LASTEXITCODE"
}

if ($LogPath) {
    "[$(Get-Date -Format s)] Summary saved to $SummaryPath" | Out-File -FilePath $LogPath -Encoding utf8 -Append
}