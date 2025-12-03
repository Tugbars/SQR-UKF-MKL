@echo off
REM deploy_python.bat - Copy DLL to python folder for Python bindings
REM Run from project root: deploy_python.bat

setlocal

REM Get script directory (project root)
set "ROOT=%~dp0"
set "DLL_SRC=%ROOT%build\Release\student_t_srukf.dll"
set "PY_DIR=%ROOT%python"

REM Check if DLL exists
if not exist "%DLL_SRC%" (
    echo ERROR: DLL not found at %DLL_SRC%
    echo.
    echo Build it first with:
    echo   cmake --build build --target student_t_srukf_shared --config Release
    exit /b 1
)

REM Check if python folder exists
if not exist "%PY_DIR%" (
    echo Creating python folder...
    mkdir "%PY_DIR%"
)

REM Copy DLL
echo Copying DLL to python folder...
copy /Y "%DLL_SRC%" "%PY_DIR%\" >nul

if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS: student_t_srukf.dll copied to python/
    echo.
    echo Test with:
    echo   cd python
    echo   python -c "from srukf import StudentTSRUKF; print('OK')"
    echo   python compare_ukf.py
) else (
    echo ERROR: Failed to copy DLL
    exit /b 1
)

endlocal
