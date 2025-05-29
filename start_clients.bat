:: filepath: c:\Users\ladia\Desktop\Federated Learning\Flower Hands-On\fl-ida-tutorial-main\start_clients.bat
@echo off
echo Starting 10 FL clients with different IDs...

:: Define window dimensions and position variables
set WINDOW_WIDTH=80
set WINDOW_HEIGHT=25
set COL_COUNT=5
set ROW_COUNT=2

FOR /L %%i IN (0,1,9) DO (
    :: Calculate row and column for this window
    set /a col=%%i %% %COL_COUNT%
    set /a row=%%i / %COL_COUNT%
    
    :: Calculate position (in character units, not pixels)
    set /a pos_x=%col% * %WINDOW_WIDTH%
    set /a pos_y=%row% * %WINDOW_HEIGHT%
    
    :: Start client with position using simpler approach
    start "FL Client %%i" /d "%~dp0" cmd /C "mode con: cols=%WINDOW_WIDTH% lines=%WINDOW_HEIGHT% & title FL Client %%i & python client.py --cid=%%i & echo Client execution completed. Window will close in 3 seconds... & timeout /t 3 > nul"
    
    :: Use simpler window movement with built-in Windows tools (no PowerShell)
    for /f "tokens=4-5" %%a in ('mode con ^| findstr "Columns Lines"') do (
        set cols=%%a
        set lines=%%b
    )
    
    echo Started client with ID %%i
    timeout /t 1 /nobreak >nul
)

echo All clients started successfully!