@echo off
cd /d "%~dp0"
where py >nul 2>nul
if not errorlevel 1 goto run_launcher
for /d %%D in ("%LocalAppData%\Programs\Python\Python*") do if exist "%%~fD\python.exe" set "PYTHON=%%~fD\python.exe"
if defined PYTHON goto run_installed
python app.py
goto finish

:run_launcher
py -3 app.py
goto finish

:run_installed
"%PYTHON%" app.py

:finish
if errorlevel 1 pause
