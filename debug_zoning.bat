@echo off

echo.
echo Zoning Startup Script
echo.
SET mypath=%~dp0
set mypath=%mypath:~0,-1%

call "%mypath%\qgis_env.bat"
IF %errorlevel% NEQ 0 (
  echo could not locate qgis_env.bat file, please run the installation script again
  pause
  exit
)
if %OSGEO4W_ROOT% == "" (
  echo.
  echo could not load the QGIS v3 environment.
  pause
  exit
)

if not exist %OSGEO4W_ROOT% echo %OSGEO4W_ROOT% does not exist
if not exist %QGISAPP% echo %QGISAPP% does not exist
if not exist %PYAPP% echo %PYAPP% does not exist
if not exist %QTAPP% echo %QTAPP% does not exist
if not exist %OSGEO4W_ROOT%\bin\o4w_env.bat echo o4w_env.bat does not exist
if not exist %OSGEO4W_ROOT%\bin\qt5_env.bat echo qt5_env.bat does not exist
if not exist %OSGEO4W_ROOT%\bin\py3_env.bat echo py3_env.bat does not exist

if exist %OSGEO4W_ROOT%\bin\o4w_env.bat call %OSGEO4W_ROOT%\bin\o4w_env.bat

if not exist %OSGEO4W_ROOT%\bin\qt5_env.bat goto setqtenv
call %OSGEO4W_ROOT%\bin\qt5_env.bat
goto setpyenv

:setqtenv
if exist %QGISAPP%\qtplugins set QT_PLUGIN_PATH=%QGISAPP%\qtplugins;%QTAPP%\plugins
if not exist %QGISAPP%\qtplugins echo QT env not set

:setpyenv
if not exist %OSGEO4W_ROOT%\bin\py3_env.bat goto setpyenvexp
call %OSGEO4W_ROOT%\bin\py3_env.bat
goto startapp

:setpyenvexp
if exist %OSGEO4W_ROOT%\apps\%QGISAP%\bin PATH=%OSGEO4W_ROOT%\apps\%QGISAP%\bin;%OSGEO4W_ROOT%\bin;%PATH%
if exist %OSGEO4W_ROOT%\apps\%QGISAP%\bin path %OSGEO4W_ROOT%\apps\%QGISAP%\bin;%PATH%
if exist %QGISAPP%\bin set QGIS_PREFIX_PATH=%OSGEO4W_ROOT:\=/%/apps/%QGISAP%
set GDAL_FILENAME_IS_UTF8=YES
set VSI_CACHE=TRUE
set VSI_CACHE_SIZE=1000000
if exist %OSGEO4W_ROOT%\apps\%QGISAP%\python set PYTHONPATH=%OSGEO4W_ROOT%\apps\%QGISAP%\python;%PYTHONPATH%

:startapp
echo.
echo PATH: %PATH%
echo.
echo PYTHONPATH: %PYTHONPATH%
echo.
START /B python zoning.py
pause