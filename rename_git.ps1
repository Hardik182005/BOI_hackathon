# This script will wait for the stale .git folder to release and rename .git_new to .git.
Write-Host "Waiting for stale .git lock to release..." -ForegroundColor Cyan
$stalePath = Join-Path $PSScriptRoot ".git"
$newPath = Join-Path $PSScriptRoot ".git_new"

while ($true) {
    $exists = Test-Path -Path $stalePath -ErrorAction SilentlyContinue
    if (-not $exists) {
        # Check if we get permission denied, which means lock is still held
        try {
            [System.IO.Directory]::Exists($stalePath) | Out-Null
            break
        }
        catch {
            # Stale path is still locked in delete-pending state
        }
    }
    Start-Sleep -Seconds 1
}

if (Test-Path -Path $newPath) {
    Rename-Item -Path $newPath -NewName ".git" -Force
    Write-Host "Successfully renamed .git_new to .git!" -ForegroundColor Green
    # Self delete
    Remove-Item -Path $MyInvocation.MyCommand.Path -Force
} else {
    Write-Host ".git_new not found or already renamed." -ForegroundColor Yellow
}
