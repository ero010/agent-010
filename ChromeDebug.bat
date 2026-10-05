@echo off
REM 010 uses YOUR Chrome profile (your tabs, your logins).
REM Close Chrome first if it is open, then run this.
taskkill /F /T /IM chrome.exe 2>nul
timeout /t 3 /nobreak >nul
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222 --user-data-dir="%LOCALAPPDATA%\Google\Chrome\User Data" --profile-directory="Profile 7" --no-first-run
echo 010 is now inside YOUR Chrome. Leave this window open.
pause
