@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 goto build_with_launcher

set "PYTHON="
for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do if exist "%%~fD\python.exe" set "PYTHON=%%~fD\python.exe"
if not defined PYTHON (
    echo Python was not found. Install Python and PyInstaller build requirements first.
    exit /b 1
)

"%PYTHON%" -m PyInstaller --noconfirm --clean --onefile --windowed --name SecurityQuiz_Adjusted --add-data "BackgroundImage_Clean.png;." --add-data "sound;sound" --collect-all cv2 --collect-all pygrabber --collect-all comtypes --hidden-import pygrabber.dshow_graph --hidden-import comtypes.client app.py
goto finish

:build_with_launcher
py -3 -m PyInstaller --noconfirm --clean --onefile --windowed --name SecurityQuiz_Adjusted --add-data "BackgroundImage_Clean.png;." --add-data "sound;sound" --collect-all cv2 --collect-all pygrabber --collect-all comtypes --hidden-import pygrabber.dshow_graph --hidden-import comtypes.client app.py

:finish
if errorlevel 1 exit /b 1
if not exist "dist\261113セキュリティクイズ問題.json" (
    copy /b "261113セキュリティクイズ問題.json" "dist\261113セキュリティクイズ問題.json" >nul
    if errorlevel 1 exit /b 1
)
echo.
echo Build complete: dist\SecurityQuiz_Adjusted.exe
echo Place the JSON question file alongside the executable.
endlocal
