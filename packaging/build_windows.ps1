# Build MicrosCount for Windows: PyInstaller folder -> Inno Setup installer (.exe) and a portable .zip.
# Usage (from the repository root, inside a Python environment with MicrosCount's dependencies):
#   powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$Root = (Get-Location).Path
$Version = (python -c "import sys; sys.path.insert(0, 'src'); from microscount import __version__; print(__version__)").Trim()
$Out = Join-Path $Root "dist\release"
Write-Host "== MicrosCount $Version (Windows)"

if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist\MicrosCount) { Remove-Item -Recurse -Force dist\MicrosCount }
python -m PyInstaller --noconfirm --clean packaging\microscount.spec --distpath dist --workpath build
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
$App = Join-Path $Root "dist\MicrosCount"

Write-Host "== smoke tests of the frozen application"
& "$App\microscount-cli.exe" selftest --report "$Root\dist\selftest-windows.txt"
if ($LASTEXITCODE -ne 0) { Get-Content "$Root\dist\selftest-windows.txt"; throw "self-test failed" }
$env:MICROSCOUNT_QUIT_AFTER_MS = "3000"
$env:QT_QPA_PLATFORM = "offscreen"
$p = Start-Process -FilePath "$App\MicrosCount.exe" -PassThru -Wait
Remove-Item Env:\MICROSCOUNT_QUIT_AFTER_MS
Remove-Item Env:\QT_QPA_PLATFORM
if ($p.ExitCode -ne 0) { throw "GUI smoke test failed (exit $($p.ExitCode))" }

Copy-Item README.md, LICENSE, THIRD_PARTY_NOTICES.md, CITATION.cff $App
New-Item -ItemType Directory -Force -Path $Out | Out-Null

Write-Host "== portable zip"
Compress-Archive -Path $App -DestinationPath "$Out\MicrosCount-$Version-Windows-x64-portable.zip" -Force

Write-Host "== Inno Setup installer"
$iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
  foreach ($c in @("${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe")) {
    if (Test-Path $c) { $iscc = $c; break }
  }
}
if (-not $iscc) { throw "Inno Setup 6 (ISCC.exe) not found: install it from https://jrsoftware.org/isinfo.php or 'choco install innosetup'" }
& $iscc "/DMyAppVersion=$Version" "packaging\windows\microscount.iss"
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Get-ChildItem $Out
