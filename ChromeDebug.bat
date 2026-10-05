@echo off
echo === 010 Chrome bridge ===
echo Close Google Chrome COMPLETELY first (all windows),
echo then press any key to reopen it for 010.
pause >nul
start "" "C:\Program Files\Google\Chrome\Application\chrome.exe" --remote-debugging-port=9222
echo.
echo 010 can now use YOUR Chrome (your tabs, your logins). Leave it open.
pause
