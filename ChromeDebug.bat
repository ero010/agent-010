@echo off
REM 010's own Chrome (separate profile copy - your main Chrome stays untouched).
REM If logins ever expire here, delete the .chrome-debug folder and ask 010 to re-sync it.
cd /d "%~dp0"
if not exist ".chrome-debug" (
  echo Copying your Chrome profile once for 010...
  robocopy "%LOCALAPPDATA%\Google\Chrome\User Data" ".chrome-debug" /E /XD Cache "Code Cache" GPUCache ShaderCache "Service Worker" Crashpad "Crash Reports" /XF "Singleton*" /R:1 /W:1 /NFL /NDL /NJH /NJS
)
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%~dp0.chrome-debug" --no-first-run
echo 010's Chrome is open. Log into your sites there ONCE, then leave it open.
pause
