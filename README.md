# cft
Climate Forecasting Toolbox


INTRODUCTION
------------
The Climate Forecasting Toolbox is a Python based tool for statistical climate forecasting applications. 

CREDITS
=======
Developer: Thembani Moitlhobogi
Methodologies: Mduduzi Sunshine Gamedze


SOURCE CODE
------------
The CFT source code is maintained at:  https://github.com/taxmanyana/cft.git

INSTALLATION ON WINDOWS
--------------------
1. Ensure you have installed the latest version of QGIS 3 in your computer
2. Unpack (unzip) the cft-x.x.x.zip ZIP file to a directory of your choosing (e.g. My Documents)
3. Navigate into the extracted folder "cft-x.x.x" 
4. Right-click the "install-cft-modules.bat" and select "Run as Administrator" to install the required python modules (internet connection required)
5. Once installed, CFT is easily be run by double-clicking on "start_cft.bat"

INSTALLATION ON LINUX
--------------------
The installation script will download dependency sources, compile and deploy the CFT
1. Unpack (unzip) the cft-x.x.x.zip ZIP file to a directory of your choosing
2. On the terminal, navigate into the extracted folder:  cd cft-x.x.x 
3. Run the installation script using the following command:   ./install-cft-linux.sh
   NB: install-cft-linux.sh will try to download some dependencies using hardcoded URLs, if a URL fails you can edit the script with correct/updated URL and re-run 
4. Once installed, you can run CFT from the terminal using the following commands:
   source python3/bin/activate
   python3 cft.py settings.json
5. CFT also has a Desktop launcher "CFT.desktop" which will be installed to the Desktop. On the Desktop double-click on "CFT.desktop" to run the tool (if it is the first time then it will bring a pop-up for you to trust and accept)
6. The MPI version can be run on the terminal by executing the following commands:
   source python3/bin/activate
   mpirun -n 40 python3 cft_mpi.py settings.json

ALTERNATIVE INSTALLATION ON LINUX (FOR UBUNTU ONLY)
--------------------
1. On the terminal, install the required dependencies using the command:   sudo apt install python3-pip python3-venv libffi-dev
2. Unpack (unzip) the cft-x.x.x.zip ZIP file to a directory of your choosing
3. On the terminal, navigate into the extracted folder:  cd cft-x.x.x
4. Run the installation script using the following command:   ./install-cft-ubuntu.sh
5. Once installed, you can run CFT from the terminal using command:    ./cft_ubuntu.sh
6. CFT also has a Desktop launcher "CFT.desktop" which will be copied to the Desktop. On the Desktop double-click on "CFT.desktop" to run the tool (if it is the first time then it will bring a pop-up for you to trust and accept)



FEATURES
--------
- Can easily select a year to forecast
- Option to forecast a 3-month season, or a single month
- Adjustable model training period
- Supports multiple predictors (in NetCDF format). Tool will loop and generate forecast for each predictor
- Support for gridded ground data (NetCDF) as the predictant
- Supports point (station) as the predictant
- Supports season cumulation (rainfall) or season average (temperature) functions
- Supports Linear Regression and Artificial Intelligence (Multilayer Perceptron regression)
- For station (point) data, a forecast can be summarized per zone, where a weighted average forecast is produced from the stations within the zone
- Computes skill scores for each forecast

Future developments
- Validates forecasts, if ground data is available for forecast period

