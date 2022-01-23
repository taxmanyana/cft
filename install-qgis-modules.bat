@echo off
echo.
echo CFT Installation Script
echo.
SET mypath=%~dp0
set mypath=%mypath:~0,-1%

set OSGEO4W_ROOT=""
rem Detect QGIS Version
FOR /F "tokens=* USEBACKQ" %%g in (`dir /b "C:\Program Files (x86)\QGIS 3*" 2^>NUL`) do (
  set OSGEO4W_ROOT="C:\Program Files (x86)\%%g"
)
FOR /F "tokens=* USEBACKQ" %%g in (`dir /b "C:\Program Files\QGIS 3*" 2^>NUL`) do (
  set OSGEO4W_ROOT="C:\Program Files\%%g"
)

if %OSGEO4W_ROOT% == "" (
  echo.
  echo could not find any QGIS v3 installation.
  pause
  exit
)

echo set OSGEO4W_ROOT=%OSGEO4W_ROOT%> "%mypath%\qgis_env.bat"
set QGISAPP=""
set QGISAP=""
FOR /F "tokens=* USEBACKQ" %%g in (`dir /b %OSGEO4W_ROOT%\apps\qgis* 2^>NUL`) do (
  set QGISAPP=%OSGEO4W_ROOT%\apps\%%g
  set QGISAP=%%g
)
set QTAPP=""
FOR /F "tokens=* USEBACKQ" %%g in (`dir /b %OSGEO4W_ROOT%\apps\qt* 2^>NUL`) do (
  set QTAPP=%OSGEO4W_ROOT%\apps\%%g
)
set PYAPP=""
FOR /F "tokens=* USEBACKQ" %%g in (`dir /b %OSGEO4W_ROOT%\apps\Python* 2^>NUL`) do (
  set PYAPP=%OSGEO4W_ROOT%\apps\%%g
)

if exist %QGISAPP% echo set QGISAPP=%QGISAPP%>> "%mypath%\qgis_env.bat"
if exist %QGISAPP% echo set QGISAP=%QGISAP%>> "%mypath%\qgis_env.bat"
if exist %QTAPP% echo set QTAPP=%QTAPP%>> "%mypath%\qgis_env.bat"
if exist %PYAPP% echo set PYAPP=%PYAPP%>> "%mypath%\qgis_env.bat"

if exist %OSGEO4W_ROOT%\bin\o4w_env.bat call %OSGEO4W_ROOT%\bin\o4w_env.bat
if exist "%mypath%\qgis_env.bat" call "%mypath%\qgis_env.bat"


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
echo QGIS installation to be used: %OSGEO4W_ROOT%
echo.
echo installing the python modules...
echo.
echo upgrading pip...
python -m pip install --upgrade pip
if %errorlevel% equ 0 (
 echo pip upgraded successfully
)
echo.
echo upgrading untangle...
python -m pip install --upgrade untangle
if %errorlevel% equ 0 (
 echo untangle upgraded successfully
)
echo.
echo upgrading numpy...
python -m pip install --upgrade numpy
if %errorlevel% equ 0 (
 echo numpy upgraded successfully
)
echo.
echo upgrading netCDF4...
python -m pip install --upgrade netCDF4
if %errorlevel% equ 0 (
 echo netCDF4 upgraded successfully
)
echo.
echo upgrading pandas...
python -m pip install --upgrade pandas
if %errorlevel% equ 0 (
 echo pandas upgraded successfully
)
echo.
echo upgrading setuptools...
python -m pip install --upgrade setuptools
if %errorlevel% equ 0 (
 echo setuptools upgraded successfully
)
echo.
echo upgrading scikit-learn...
python -m pip install --upgrade scikit-learn
if %errorlevel% equ 0 (
 echo sscikit-learn upgraded successfully
)
echo.
echo upgrading statsmodels...
python -m pip install --upgrade statsmodels
if %errorlevel% equ 0 (
 echo statsmodels upgraded successfully
)
echo.
echo upgrading scipy...
python -m pip install --upgrade scipy
if %errorlevel% equ 0 (
 echo scipy upgraded successfully
)
echo.
echo upgrading geojson...
python -m pip install --upgrade geojson
if %errorlevel% equ 0 (
 echo geojson upgraded successfully
)
echo.
echo upgrading shapely...
python -m pip install --upgrade shapely
if %errorlevel% equ 0 (
 echo shapely upgraded successfully
)
echo.
echo upgrading descartes...
python -m pip install --upgrade descartes
if %errorlevel% equ 0 (
 echo descartes upgraded successfully
)
echo.
pause
