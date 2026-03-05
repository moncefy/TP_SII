@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PHP_EXE="

REM Try WAMP latest PHP first
for /f %%I in ('dir /b /ad /o-n "C:\wamp64\bin\php\php*" 2^>nul') do (
    if not defined PHP_EXE set "PHP_EXE=C:\wamp64\bin\php\%%I\php.exe"
)

REM Fallbacks
if not defined PHP_EXE if exist "C:\xampp\php\php.exe" set "PHP_EXE=C:\xampp\php\php.exe"
if not defined PHP_EXE if exist "C:\php\php.exe" set "PHP_EXE=C:\php\php.exe"

if not defined PHP_EXE (
    echo Could not find php.exe automatically.
    echo Please edit this file and set PHP_EXE manually.
    pause
    exit /b 1
)

echo Using: %PHP_EXE%
echo Initializing database...
"%PHP_EXE%" "%PROJECT_DIR%init_db.php"
if errorlevel 1 (
    echo Failed to initialize database.
    pause
    exit /b 1
)

echo.
echo Opening demo in browser...
start "" "http://localhost:6969/index.php"

echo Starting PHP server on http://localhost:6969
echo Press Ctrl + C to stop.
echo.
"%PHP_EXE%" -S localhost:6969 -t "%PROJECT_DIR%"

endlocal
