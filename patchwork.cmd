@echo off
setlocal

:: Get the directory of this script
set DIR=%~dp0

:: Activate the virtual environment
call "%DIR%.venv\Scripts\activate.bat"

:: Run the Python script with the provided directory path
python "%DIR%main.py" %*

endlocal