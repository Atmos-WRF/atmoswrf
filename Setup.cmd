@echo off
rem Atmos-WRF, first run on a new PC.  Checks the folder and the graphics card, finds Python 3.13
rem (or puts its own copy in runtime\python: NuGet's official package, nothing installed into Windows) and
rem builds the runtime's venv at the versions in constraints.txt (one runtime for every version on the PC,
rem %LOCALAPPDATA%\ARHT-MODEL\runtime: a new version only installs what it changed):
rem   - with an NVIDIA card: ArWen's GPU core, CuPy and CUDA 12 (requirements.txt, about 3 GB), then a GPU test;
rem   - without one (AMD, Intel, no graphics card), or "Setup.cmd cpu": ArWen on the processor
rem     (requirements-cpu.txt, about 0.4 GB), then a CPU test.
rem Then the Thompson microphysics lookup tables (~650 MB, GPU only: the CPU backend runs without them);
rem "Setup.cmd tempo" also gets TEMPO's (~930 MB).  Running it again only fetches what is missing.
rem "Atmos-WRF.cmd" runs this by itself on the first start.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set "PYVER=3.13.5"
set "SB_DIR=%~dp0"

rem --- unpacked? (a .cmd opened inside the zip runs alone from a temp folder)
if not exist "arhtmodelviewer\__init__.py" goto :notunpacked

rem --- an NVIDIA graphics card with its driver (CUDA), else the CPU backend
set "BACKEND=cuda"
set "ASKED="
if /i "%~1"=="cpu" set "ASKED=1"
if /i "%~2"=="cpu" set "ASKED=1"
set "TEMPO="
if /i "%~1"=="tempo" set "TEMPO=--tempo"
if /i "%~2"=="tempo" set "TEMPO=--tempo"
if defined ASKED set "BACKEND=cpu"

rem --- the minimum PC (syscheck.ps1): Windows, 8 GB of memory and an NVIDIA GTX 10 series or newer with 4 GB;
rem     without one, the CPU backend on 8+ cores with AVX2 and 16 GB (exit 2); else it stops here (exit 1)
set "CHK="
if defined ASKED set "CHK=-Cpu"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SB_DIR%syscheck.ps1" %CHK%
if errorlevel 2 set "BACKEND=cpu"
if errorlevel 2 goto :checkpath
if errorlevel 1 goto :rejected
if "%BACKEND%"=="cpu" goto :checkpath
where nvidia-smi >nul 2>nul || set "BACKEND=cpu"
if "%BACKEND%"=="cpu" goto :checkpath
set "DRV="
for /f "tokens=1 delims=." %%a in ('nvidia-smi --query-gpu^=driver_version --format^=csv^,noheader 2^>nul') do if not defined DRV set "DRV=%%a"
if not defined DRV set "BACKEND=cpu"
if "%BACKEND%"=="cpu" goto :checkpath
if %DRV% LSS 528 echo Note: your NVIDIA driver ^(%DRV%^) may be older than 527.41, the minimum for CUDA 12; if the GPU test fails, update it.

:checkpath
rem --- a short folder: the CUDA packages and ArWen's own data nest ~140-150 characters deep and Windows
rem     stops at 260
powershell -NoProfile -Command "exit $env:SB_DIR.Length"
if errorlevel 101 goto :toolong

rem --- one runtime for every version on this PC (runtime-link.ps1): %LOCALAPPDATA%\ARHT-MODEL\runtime, this
rem     folder's "runtime" a junction to it; an older version's install is adopted, not downloaded again
set "RT="
for /f "usebackq delims=" %%r in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%SB_DIR%runtime-link.ps1" -Project "%SB_DIR%."`) do set "RT=%%r"
if not defined RT goto :rtfail
if not exist "%RT%\" goto :rtfail
powershell -NoProfile -Command "exit $env:RT.Length"
if errorlevel 93 goto :toolongrt

echo.
if exist "%RT%\venv\Scripts\python.exe" (
  echo Checking the installed packages: only what this version changed is downloaded.
  goto :pyfind
)
if "%BACKEND%"=="cuda" (
  echo First-time setup: this downloads about 3.7 GB ^(ArWen, CuPy, CUDA 12, microphysics tables^) and takes 5-25 minutes.
) else (
  if defined ASKED echo Setting up the CPU backend: the viewer will run ArWen on the processor. That works,
  if not defined ASKED echo No suitable NVIDIA graphics card: the viewer will run ArWen on the processor ^(CPU^). That works,
  echo but it is much slower than an NVIDIA card: use 2-3 km grids. First-time setup downloads about
  echo 0.4 GB ^(ArWen and a C++ compiler^) and takes 3-10 minutes.
)
echo.

:pyfind
rem --- Python 3.13: the PC's own, else the viewer's own copy in runtime\python
set "PY="
if exist "%RT%\python\tools\python.exe" set "PY="%RT%\python\tools\python.exe""
if not defined PY where py >nul 2>nul && py -3.13 -c "pass" >nul 2>nul && set "PY=py -3.13"
if not defined PY python -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 13))" >nul 2>nul && set "PY=python"
if defined PY goto :havepy
echo Downloading Python %PYVER% ^(14 MB, NuGet's python package^) into runtime\python ...
if not exist "%RT%\cache" mkdir "%RT%\cache"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol='Tls12'; $z=(Join-Path $env:RT 'cache\python.zip'); Invoke-WebRequest -UseBasicParsing ('https://api.nuget.org/v3-flatcontainer/python/' + $env:PYVER + '/python.' + $env:PYVER + '.nupkg') -OutFile $z; Expand-Archive -Force $z (Join-Path $env:RT 'python'); Remove-Item $z" || goto :fail
if not exist "%RT%\python\tools\python.exe" goto :fail
set "PY="%RT%\python\tools\python.exe""
:havepy

rem the venv is made at the runtime's real path, so it never depends on one version's folder
if exist "%RT%\venv\Scripts\python.exe" goto :haveenv
echo Creating runtime\venv ...
%PY% -m venv "%RT%\venv" || goto :fail
:haveenv

rem pip's download cache stays in the runtime (shared by every version), not in the user profile
set "PIP_CACHE_DIR=%RT%\cache\pip"
"%RT%\venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
if "%BACKEND%"=="cpu" goto :installcpu
echo Installing ArWen ^(gpuwm^), CuPy and CUDA 12 ...
"%RT%\venv\Scripts\python.exe" -m pip install -r requirements.txt -c constraints.txt || goto :fail

echo.
echo Testing the GPU ...
"%RT%\venv\Scripts\python.exe" -c "import warnings; warnings.filterwarnings('ignore', message='CUDA path'); import cupy as cp; x = cp.arange(1000000, dtype=cp.float32); assert float((x * 2).sum()) > 0; print('GPU OK:', cp.cuda.runtime.getDeviceProperties(0)['name'].decode())" || goto :gpufail

rem --- microphysics lookup tables into runtime\tables (gpuwm checks each file's SHA-256); without them
rem     runs still work, on the built-in core with Milbrandt-Yau
echo.
echo Downloading the microphysics lookup tables ^(Thompson, about 650 MB^) ...
"%RT%\venv\Scripts\python.exe" -m arhtmodelviewer.tables %TEMPO% || set "TABLESFAILED=1"
if defined TABLESFAILED echo The tables did not all download ^(see above^). The viewer still works without them;
if defined TABLESFAILED echo run Setup.cmd again later to fetch the rest.
goto :done

:installcpu
echo Installing ArWen ^(gpuwm^) and the C++ compiler for the CPU backend ...
"%RT%\venv\Scripts\python.exe" -m pip install -r requirements-cpu.txt -c constraints.txt || goto :fail
echo.
echo Testing the CPU backend ^(compiles a first kernel^) ...
set "ARHTMODELVIEWER_BACKEND=cpu"
"%RT%\venv\Scripts\python.exe" -m arhtmodelviewer.cpu.selftest || goto :cpufail

:done
rem the packages this version needs are in: the launcher runs setup again only when a version changes them
powershell -NoProfile -ExecutionPolicy Bypass -File "%SB_DIR%runtime-link.ps1" -Project "%SB_DIR%." -Stamp
rem files from a downloaded zip carry Windows' "downloaded" mark; the launchers keep warning without this
powershell -NoProfile -Command "Get-ChildItem -LiteralPath $env:SB_DIR -File | Unblock-File" >nul 2>nul

echo.
echo Setup finished.
if "%BACKEND%"=="cpu" echo The first run compiles the model for this processor ^(a few minutes, once^).
if /i "%~1"=="/auto" exit /b 0
echo Start the viewer with "Atmos-WRF.cmd".
pause
exit /b 0

:notunpacked
echo Atmos-WRF has to be unpacked first: right-click the zip, "Extract All...", then run
echo "Atmos-WRF.cmd" from the extracted folder.
goto :fail

:rejected
echo Nothing was downloaded or installed.
goto :fail

:toolong
echo This folder's path is too long for the packages ^(Windows stops at 260 characters^):
echo   %SB_DIR%
echo Move the folder somewhere shorter, for example C:\Atmos-WRF, and run it again.
goto :fail

:rtfail
echo runtime-link.ps1 could not set up the shared runtime ^(see its message above^), so nothing was
echo downloaded.  Run Setup.cmd again; if it keeps failing, set ARHTMODELVIEWER_RUNTIME to a short folder,
echo for example C:\Atmos-WRF-runtime.
goto :fail

:toolongrt
echo The shared runtime's path is too long for the packages ^(Windows stops at 260 characters^):
echo   %RT%
echo Set ARHTMODELVIEWER_RUNTIME to a short folder, for example C:\Atmos-WRF-runtime, and run it again.
goto :fail

:gpufail
echo The GPU test failed ^(see above^). Update the NVIDIA driver from https://www.nvidia.com/drivers,
echo restart the PC and run Setup.cmd again.  Or run the viewer on the processor instead: delete the
echo runtime\venv folder and run "Setup.cmd cpu".
goto :fail

:cpufail
echo The CPU backend test failed ^(see above^).
goto :fail

:fail
echo.
echo Setup did not finish ^(see the messages above^).
pause
exit /b 1
