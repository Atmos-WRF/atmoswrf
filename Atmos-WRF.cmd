@echo off
rem Atmos-WRF: starts the model (GPU or CPU) with Level II export and GR2Analyst polling, and opens
rem the viewer.  The polling feed is also shared with other computers on the network (LAN, or a VPN such
rem as Tailscale): in GR2Analyst there, File - Configure Polling - Add http://<this PC's address>:8830/poll/
rem (the Radars tab and this window show it).  Windows asks once whether Python may use the network:
rem allow it on private networks.  The first run on a PC runs Setup.cmd by itself.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist "arhtmodelviewer\__init__.py" goto :notunpacked
rem one runtime for every version on this PC (runtime-link.ps1): a new version links to it instead of
rem downloading it again, and runs setup only for packages it changed
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0runtime-link.ps1" -Project "%~dp0." >nul
if not exist "runtime\venv\Scripts\python.exe" call Setup.cmd /auto || exit /b 1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0runtime-link.ps1" -Project "%~dp0." -Check
if errorlevel 1 call Setup.cmd /auto || exit /b 1
"runtime\venv\Scripts\python.exe" -m arhtmodelviewer serve 8830 --lan %*
pause
exit /b 0

:notunpacked
echo Atmos-WRF has to be unpacked first: right-click the zip, "Extract All...", then run
echo "Atmos-WRF.cmd" from the extracted folder.
pause
exit /b 1
