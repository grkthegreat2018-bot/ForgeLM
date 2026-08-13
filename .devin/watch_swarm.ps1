param([int]$Tail = 30)
# Live tail of the research team swarm output with color coding
$file = "C:\Users\tmk68\AppData\Local\Temp\devin.exe-overflows\shell-0644e6-ebef276c11f8432f\content.txt"

if (-not (Test-Path $file)) {
    # Find the latest overflow file
    $latest = Get-ChildItem "C:\Users\tmk68\AppData\Local\Temp\devin.exe-overflows\shell-*\content.txt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($latest) { $file = $latest.FullName }
    else { Write-Host "No swarm output file found. Is the swarm running?"; exit }
}

Write-Host "Watching: $file" -ForegroundColor DarkGray
Write-Host "Press Ctrl+C to stop.`n" -ForegroundColor DarkGray

# Strip ANSI codes for clean display, color by message type
Get-Content $file -Wait -Tail $Tail | ForEach-Object {
    $line = $_ -replace '\x1b\[[0-9;]*m', ''  # strip ANSI
    if ($line -match '\[search\]') {
        Write-Host $line -ForegroundColor Cyan
    } elseif ($line -match '\[result\]') {
        Write-Host $line -ForegroundColor Green
    } elseif ($line -match '\[critique\]') {
        Write-Host $line -ForegroundColor Yellow
    } elseif ($line -match '\[draft\]') {
        Write-Host $line -ForegroundColor Magenta
    } elseif ($line -match '\[tool\]') {
        Write-Host $line -ForegroundColor Blue
    } elseif ($line -match '\[doc\]') {
        Write-Host $line -ForegroundColor DarkGreen
    } elseif ($line -match '\[chat\]') {
        Write-Host $line -ForegroundColor Gray
    } elseif ($line -match 'ROUND|Phase|====') {
        Write-Host $line -ForegroundColor White
    } else {
        Write-Host $line
    }
}
