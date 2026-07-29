@echo off
cd /d "%~dp0"

if not exist "data\Online Retail.xlsx" (
    echo.
    echo DATASET NOT FOUND
    echo Copy "Online Retail.xlsx" into the data folder.
    echo Expected location:
    echo %CD%\data\Online Retail.xlsx
    echo.
    pause
    exit /b 1
)

python main.py
echo.
pause
