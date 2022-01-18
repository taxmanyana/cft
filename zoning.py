#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:59:45 2022

@author: thembani
"""


from functions import *
import matplotlib.pyplot as plt
import numpy as np
import threading
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import pandas as pd
import os, time, sys, re
from datetime import datetime
import geojson
from pathlib import Path
import math

try:
    from osgeo import gdal, ogr
except:
    pass

from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import QThread, QObject, QDate, QTime, QDateTime, Qt
qtCreatorFile = "zoning.ui"
settingsfile = 'zoning.json'
csvheader = 'Year,Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec'

def lat(stn, df):
    return float(df.loc[df.Station==stn]['Lat'])

def lon(stn, df):
    return float(df.loc[df.Station==stn]['Lon'])

def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)
    # In IDW, weights are 1 / distance
    weights = 1.0 / dist
    # Make weights sum to one
    weights /= weights.sum(axis=0)
    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi

def linear_rbf(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)
    # Mutual pariwise distances between observations
    internal_dist = distance_matrix(x,y, x,y)
    # Now solve for the weights such that mistfit at the observations is minimized
    weights = np.linalg.solve(internal_dist, z)
    # Multiply the weights for each interpolated point by the distances
    zi =  np.dot(dist.T, weights)
    return zi


def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T
    # Make a distance matrix between pairwise observations
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])
    return np.hypot(d0, d1)


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d hour(s) %d minute(s) %d second(s)" % (hour, minutes, seconds)

def shortest_distance(x1, y1, a, b, c): 
    return abs((a * x1 + b * y1 + c)) / (math.sqrt(a * a + b * b))


def fixname(text):
    newname = re.sub("[^0-9a-zA-Z]+", "_", str(text).strip())
    return newname.strip('_')

def concat_csvs(csvs, missing):
    dfs_files = []
    for file in csvs:
        dfs_files.append(pd.read_csv(file, encoding = 'ISO-8859-9'))
    dfs_files = pd.concat((dfs_files), axis=0)
    dfs_files = dfs_files.replace(missing, np.nan)
    dfs_files = dfs_files.dropna(how='all')
    dfs_files['ID'] = dfs_files['ID'].apply(fixname)
    return dfs_files


def stationdata(idata, station, season):
    smonths = season_months[season][:]
    smonths.insert(0, 'Year')
    sdata = idata.loc[idata.ID==station][smonths]
    sdata.drop_duplicates('Year', inplace=True)
    sdata = sdata.apply(pd.to_numeric, errors='coerce')
    sdata.set_index('Year', inplace=True)
    return sdata

def bigger(bounds1, bounds2):
    # check if bounds1 is bigger than bounds2
    minx1, miny1, maxx1, maxy1 = bounds1
    minx2, miny2, maxx2, maxy2 = bounds2
    if (maxy1 - miny1) * (maxx1 - minx1) > (maxy2 - miny2) * (maxx2 - minx2):
        return 1
    else:
        return 0


def smooth(shp,out,name,smooth_size):
    driver = ogr.GetDriverByName('GeoJSON')
    dataSource = driver.Open(shp, 0)  # 0 It's read-only ,1 Can write 
    layer = dataSource.GetLayer(0)
    t = int(layer.GetFeatureCount())
    drv = ogr.GetDriverByName('GeoJSON')
    Polygon = drv.CreateDataSource(out)
    oLayer = Polygon.CreateLayer(name)
    oFieldID = ogr.FieldDefn("Zone", ogr.OFTInteger)
    oLayer.CreateField(oFieldID, 1)
    feature = ogr.Feature(oLayer.GetLayerDefn())
    ID=0
    for i in range(0, t):
        feat = layer.GetFeature(i)
        geom = feat.GetGeometryRef()
        ID = ID+1
        buffer = geom.Buffer(smooth_size).Buffer(-smooth_size)
        feature.SetGeometry(buffer)
        feature.SetField(0, ID)
        oLayer.CreateFeature(feature)


#
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)

class MyApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MyApp()
    window.show()
    
    try:
        with open(settingsfile, "r") as read_file:
            config = json.load(read_file)
    except:
        config = {}
        config['outdir'] = ''
        config['base_vector'] = {"file": '', "ID": 0, "attr": []}
        config['interpolation'] = 'idw'
        config['ExplainedVariance'] = 70
        config['gridsize'] = 0.05
        config['rotation'] = {"types": ["varimax", "quartimax"], "indx": 0}
        config['inputFormat'] = "CSV"
        config['predictandMissingValue'] = -9999
        config['composition'] = "Sum"
        config['period'] = {"season": ['JFM', 'FMA', 'MAM', 'AMJ', 
                                       'MJJ', 'JJA', 'JAS', 'ASO', 
                                       'SON', 'OND', 'NDJ', 'DJF'], "indx": 9}
        config['startyr'] = 1981
        config['endyr'] = 2010
        config['zones'] = ''
        config['smoothing'] = 0
        config['predictandList'] = []
        config['predictandattr'] = {"params": ['pre'], "indx": 0}
        window.statusbar.showMessage("Default settings loaded.")

    def closeapp():
        sys.exit(app.exec_())

    def addBaseVector():
        global config
        window.inputlayerlabel.setText('')
        config['base_vector'] = {"file": '', "ID": 0, "attr": []}
        vectorfieldsx = []
        # window.zoneIDcomboBox.setDuplicatesEnabled(False)
        fileName = QtWidgets.QFileDialog.getOpenFileName(window,
                  'Add File', '..' + os.sep, filter="GeoJson File (*.geojson)")
        config['base_vector']['file'] = fileName[0]
        window.inputlayerlabel.setText(os.path.basename(config.get('base_vector',{}).get('file')))
    
    def getOutDir():
        global config
        config['outdir'] = QtWidgets.QFileDialog.getExistingDirectory(directory='..' + os.sep)
        window.outdirlabel.setText(config.get('outdir'))

    def addPredictands():
        global config
        global csvheader
        config['predictandList'] = []
        window.predictandlistWidget.clear()
        window.predictandparamcombobox.clear()
        window.statusbar.showMessage("")
        if window.CSVRadio.isChecked() == True:
            config['inputFormat'] = "CSV"
            fileNames = QtWidgets.QFileDialog.getOpenFileNames(window,
                    'Add File(s)', '..' + os.sep, filter="CSV File (*.csv)")
            for filename in fileNames[0]:
                with open(filename) as f:
                    fline = f.readline().rstrip()
                if fline.count(',') < 4:
                    window.statusbar.showMessage(
                        "Format error in "+os.path.basename(filename)+", check if comma delimited")
                    continue
                if csvheader not in fline:
                    window.statusbar.showMessage(
                        "Format error, one or more column headers incorrect in " + os.path.basename(filename))
                    continue
                config['predictandList'].append(filename)
                window.predictandlistWidget.addItem(os.path.basename(filename))
        elif window.NetCDFRadio.isChecked() == True:
            config['inputFormat'] = "NetCDF"
            try:
                fileName = QtWidgets.QFileDialog.getOpenFileNames(window,
                    'Add File', '..' + os.sep, filter="NetCDF File (*.nc*)")[0]
                predictand = Dataset(fileName[0])
                for key in predictand.variables.keys():
                    if key not in ['Y', 'X', 'Z', 'T', 'zlev', 'time', 'lon', 'lat']:
                        window.predictandparamcombobox.addItem(key)
                config['predictandList'].append(fileName[0])
                window.predictandlistWidget.addItem(os.path.basename(fileName[0]))
            except:
                window.statusbar.showMessage(
                    "Could not read predictand file, check if it is a valid NetCDF")
                return

    def clearPredictands():
        global config
        config['predictandList'] = []
        window.predictandlistWidget.clear()
        window.predictandparamcombobox.clear()

    def change_format_type():
        global config
        window.predictandlistWidget.clear()
        window.predictandparamcombobox.clear()
        if window.CSVRadio.isChecked() == True:
            config['inputFormat'] = "CSV"
        else:
            config['inputFormat'] = "NetCDF"

    def change_interpolation():
        global config
        if window.radioButtonIDW.isChecked() == True:
            config['interpolation'] = "idw"
        else: 
            config['interpolation'] = "linear"
            
    def change_composition():
        global config
        if window.cumRadio.isChecked() == True:
            config['composition'] = "Sum"
        else: 
            config['composition'] = "Avg"
        
    def write_config():
        global settingsfile
        global config
        config['ExplainedVariance'] = float(window.PEV.text())
        config['gridsize'] = float(window.gridsizelineEdit.text())
        config['predictandMissingValue'] = str(window.missingvalueslineEdit.text())  
        config['startyr'] = int(window.startyearLineEdit.text())
        config['endyr'] = int(window.endyearLineEdit.text())
        config['zones'] = window.ZonesLineEdit.text()
        config['rotation']['indx'] = config.get('rotation').get('types').index(window.rotationDcombobox.currentText())
        config['period']['indx'] = config.get('period').get('season').index(window.periodComboBox.currentText())
        # Write configuration to settings file
        import json
        with open(settingsfile, 'w') as fp:
            json.dump(config, fp, indent=4)

    def launch_zoning_Thread():
        t = threading.Thread(target=exec_zoning)
        t.start()
    
    def exec_zoning():
        global config
        write_config()
        scriptpath = os.path.dirname(os.path.realpath(__file__))
        scriptpath = Path(scriptpath)
        outdir = Path(config.get('outdir'))
        csvs = config.get('predictandList')        
        startyr = int(config.get('startyr'))
        endyr = int(config.get('endyr'))
        missing = float(config.get('predictandMissingValue'))
        ExplainedVariance = float(config.get('ExplainedVariance'))
        gridsize = float(config.get('gridsize'))
        rotation = config.get('rotation').get('types')[config.get('rotation').get('indx')]
        period = config.get('period').get('season')[int(config.get('period').get('indx'))]
        composition = config.get('composition')
        interpolation = config.get('interpolation')
        base_vector = config.get('base_vector').get('file')
        nzones = config.get('zones')
        if nzones.isnumeric(): 
            nzones = int(nzones)
        else:
            nzones = None
        vname, _ = os.path.splitext(os.path.basename(base_vector))
        vname = fixname(vname)
        proj='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
              'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],' \
              'AUTHORITY["EPSG","4326"]]'
        
        # check if output directory exists
        if not os.path.exists(outdir):
            print("output directory does not exist")
            window.statusbar.showMessage("output directory does not exist")
            sys.exit()
        
        # create output directory
        zonepath = (outdir / vname).joinpath(period + '_' + str(startyr)  + '_' + str(endyr))
        os.makedirs(zonepath, exist_ok=True)
        
        # outputs 
        dst_layername = "izones"
        dst_zones = str(zonepath / (dst_layername + ".geojson"))
        zonejson = str(zonepath / (vname + "_zones.geojson"))
        zonejson_clean = str(zonepath / (dst_layername + "_clean.geojson"))
        zonejson_smooth = str(zonepath / (dst_layername + "_smooth.geojson"))
        zonetif = str(zonepath / "izones.tif")
        datacsv = str(zonepath / "data.csv")
        faw = str(zonepath / "components.csv")
        far = str(zonepath / ("components_" + rotation + ".csv"))
        farpng = str(zonepath / ("components_" + rotation + ".png"))
        izonespng = str(zonepath / ("izones.png"))
        
        # clean outputs
        if os.path.exists(zonejson):
            os.remove(zonejson)
            
        if os.path.exists(dst_zones):
            os.remove(dst_zones)
        
        if os.path.exists(zonetif):
            os.remove(zonetif)
        
        if os.path.exists(faw):
            os.remove(faw)
        
        if os.path.exists(far):
            os.remove(far)
            
            
        #######
        start_time = time.time()
        print("Start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        window.statusbar.showMessage("Start time: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        input_data = concat_csvs(csvs, missing)
        stations = list(input_data['ID'].unique())
        nstations = len(stations)
        lats, lons = [], []
        for n in range(nstations):
            station_data_all = input_data.loc[input_data['ID'] == stations[n]]
            lats.append(station_data_all['Lat'].unique()[0])
            lons.append(station_data_all['Lon'].unique()[0])
            
        trainyears = list(range(startyr,endyr+1))
        data = pd.DataFrame(columns=[stations], index=trainyears)
        
        for station in stations:
            station_data = stationdata(input_data, station, period)
            for year in trainyears:
                if composition == "Sum":
                    data.loc[[year], station] = season_cumulation(station_data, year, period)
                else:
                    data.loc[[year], period] = season_average(station_data, year, period)
        
        print('number of stations',len(stations))
        window.statusbar.showMessage('number of stations',len(stations))
        data.to_csv(datacsv)
        
        data.columns = ['s'+str(x) for x in range(len(stations))]
        X = StandardScaler().fit_transform(data)
        fa = PCA()
        fa.fit(X)
        
        print('explained variance')
        window.statusbar.showMessage('explained variance')
        explained_variance = fa.explained_variance_.T
        
        total_exp_var = explained_variance.sum()
        explained_variance_p, y = [], 0
        for x in explained_variance:
            y = y + x
            explained_variance_p.append(round(100. * (y / total_exp_var),1))
            
        print(np.array(explained_variance).round(2))
        print(explained_variance_p)
        n_comps = (np.array(explained_variance_p) <= ExplainedVariance).sum() + 1
        colnames = ['cmp'+str(x) for x in list(range(1,n_comps+1))]
        
        fa = FactorAnalysis()
        fa.set_params(n_components=n_comps)
        fa.fit(X)
        components = fa.components_.T
        df1 = pd.DataFrame(components,columns = [colnames])
        df1['station'] = stations
        df1['Lat'] = lats
        df1['Lon'] = lons
        df1.to_csv(faw)
        
        fa = FactorAnalysis(rotation=rotation)
        fa.set_params(n_components=n_comps)
        fa.fit(X)
        components = fa.components_.T
        df2 = pd.DataFrame(components,columns = [colnames])
        df2['station'] = stations
        df2['Lat'] = lats
        df2['Lon'] = lons
        df1.to_csv(far)
        
        # open base map to get bounds
        with open(base_vector, "r") as read_file:
            base_map = geojson.load(read_file)
        feature = base_map['features'][0]
        poly = feature['geometry']
        polygon = shape(feature['geometry'])
        minx, miny, maxx, maxy = polygon.bounds
        minx = int(minx-1) * 1.0
        maxx = round(maxx+0.5,0)
        miny = int(miny-1) * 1.0
        maxy = round(maxy+0.5,0)
        
        # target grid to interpolate to
        if minx > int(min(lons)-1) * 1.0: minx = int(min(lons)-1) * 1.0
        if maxx < round(max(lons)+0.5,0): maxx = round(max(lons)+0.5,0)
        if miny > int(min(lats)-1) * 1.0: miny = int(min(lats)-1) * 1.0
        if maxy < round(max(lats)+0.5,0): maxy = round(max(lats)+0.5,0)
        x = np.array(lons)
        y = np.array(lats)
        xs = np.arange(minx,maxx+gridsize,gridsize)
        ys = np.arange(miny,maxy+gridsize,gridsize)
        nx = len(xs)
        ny = len(ys)
        
        xi,yi = np.meshgrid(xs,ys)
        xi, yi = xi.flatten(), yi.flatten()
        comps = []
        
        c = 3
        r = int(len(colnames)/3)
        if (len(colnames)%3 != 0) or (r == 0): r+=1
        
        
        fig, axs = plt.subplots(r, c, figsize=(30, 6*r), facecolor='w', edgecolor='k')
        fig.tight_layout()
        axs = axs.ravel()
         
        for i in range(len(colnames)):
            col = colnames[i]
            z = np.ravel(df2[col].values)
            # interpolate
            if interpolation == 'idw':
                zi = simple_idw(x,y,z,xi,yi)
            else:
                zi = linear_rbf(x,y,z,xi,yi)
            zi = zi.reshape((ny, nx))
            comps.append(list(zi))
            # plot
            axs[i].imshow(zi, extent=(minx, maxx, maxy, miny))
            axs[i].plot(x,y,'k.')
            axs[i].add_patch(PolygonPatch(poly, fc=None, fill=False, ec='#8f8f8f', alpha=1., zorder=2))
            axs[i].title.set_text(col)
            axs[i].title.set_fontsize(10)
            axs[i].invert_yaxis()
            axs[i].set_xlim([minx, maxx])
            axs[i].set_ylim([miny, maxy])
        plt.savefig(farpng,dpi=100)
        plt.close(fig)
        
        
        
        comps = np.array(comps)    
        #comps[np.isnan(comps)] = 0
        comps = comps.reshape(len(colnames),len(xs)*len(ys))
        
        if nzones is None:
            # compute optimal number of zones
            print('compute optimal number of zones')
            window.statusbar.showMessage('compute optimal number of zones')
            wcss = []
            distances = []
            maxk = 16
            klist = list(range(1, maxk+1))
            for d in klist:
                kmeans = cluster.KMeans(n_clusters = d, random_state=42).fit(comps.T)
                wcss.append(kmeans.inertia_)
            
            # create linear coefficients for line joining first and last kmeans inertia
            a = (float(wcss[-1] - wcss[0]))/(klist[-1] - klist[0])
            b = -1
            c = wcss[0] - (klist[0] * a)
            
            # calculate perpendicular distances from each kmeans to line
            for e in klist:
                distances.append((e, shortest_distance(e, wcss[e-1], a, b, c)))
            
            # locate albow and get optimal number of zones
            distances.sort(key=lambda g: g[1])
            n_clusters = distances[-1][0]
        else:
            n_clusters = nzones
        
        print('clustering of components')
        window.statusbar.showMessage('clustering of components')
        db = cluster.KMeans(n_clusters = n_clusters, random_state=42).fit(comps.T)
        zi = db.labels_.reshape(len(ys), len(xs))
        print("number of zones ", len(np.unique(db.labels_)))
        window.statusbar.showMessage("number of zones ", len(np.unique(db.labels_)))
        # plot
        DPI = 150
        W = 750
        H = int(W * ny / nx)
        fig = plt.figure(figsize=(W / float(DPI), H / float(DPI)), frameon=True, dpi=DPI)
        ax = fig.gca()
        ax.imshow(zi, extent=(minx, maxx, maxy, miny))
        ax.plot(x,y,'k.')
        ax.add_patch(PolygonPatch(poly, fc=None, fill=False, ec='#8f8f8f', alpha=1., zorder=2))
        ax.title.set_text('zones')
        ax.title.set_fontsize(10)
        ax.invert_yaxis()
        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])
        plt.savefig(izonespng,dpi=DPI)
        plt.close(fig)
        
        # elbow plot
        if nzones is None:        
            if os.path.exists(fzonepath / 'elbowplot.png'):
                os.remove(zonepath / 'elbowplot.png')
            plt.figure()
            plt.plot(klist, wcss, 'bx-')
            plt.plot(n_clusters, wcss[klist.index(n_clusters)], 'r.')
            plt.xlabel('Values of K')
            plt.xticks(klist)
            plt.ylabel('Distortion')
            plt.title('The Elbow Method using Distortion')
            plt.savefig(zonepath / 'elbowplot.png',dpi=100)
            plt.close(fig)
        
        # pcreate a raster for the zones
        drv = gdal.GetDriverByName("GTiff")
        ds = drv.Create(zonetif, nx, ny, eType=gdal.GDT_Byte)
        xres = (maxx - minx) / float(nx)
        yres = (maxy - miny) / float(ny)
        geotransform = (minx, gridsize, 0, maxy, 0, -gridsize)
        ds.SetProjection(proj)
        ds.SetGeoTransform(geotransform)
        ds.GetRasterBand(1).WriteArray(np.flip(zi, axis=0))
        srcband = ds.GetRasterBand(1)
        
        # polygonize the zone raster into a geojson
        drv2 = ogr.GetDriverByName("GeoJSON")
        dst_ds = drv2.CreateDataSource(dst_zones)
        dst_layer = dst_ds.CreateLayer(dst_layername, srs = None )
        newField = ogr.FieldDefn('Zone', ogr.OFTReal)
        dst_layer.CreateField(newField)
        gdal.Polygonize(srcband, None, dst_layer, 0, [])
        
        # close the data sources
        ds = None
        dst_ds = None
        
        
        # clean the vector zone map
        print('cleaning the vector zone map')
        window.statusbar.showMessage('cleaning the vector zone map')
        with open(dst_zones, "r") as read_file:
            base_map = geojson.load(read_file)
        
        extents = []
        for feature in base_map['features']:
            extents.append(shape(feature['geometry']).bounds)
        smallerpolygons = []
        for i in range(len(extents)):
            for j in range(len(extents)):
                if i == j: continue
                zonei = base_map['features'][i]['properties']['Zone']
                zonej = base_map['features'][j]['properties']['Zone']
                if (zonei == zonej) and (bigger(extents[i], extents[j])):
                    smallerpolygons.append(j)
        
        containedpolygons = []
        for i in range(len(extents)):
            for j in range(len(extents)):
                if i == j: continue
                coordsi = base_map['features'][i]['geometry']['coordinates'][0]
                coordsj = base_map['features'][j]['geometry']['coordinates'][0]
                polyi = Polygon([(x,y) for x,y in coordsi])
                polyj = Polygon([(x,y) for x,y in coordsj])
                if polyi.contains(polyj):
                    containedpolygons.append(j)
                    
        containedpolygons = np.unique(containedpolygons)
        smallerpolygons = np.unique(smallerpolygons)
        smallerpolygons = list(set(smallerpolygons).intersection(containedpolygons))
        print('will clean out tiny polygons ', smallerpolygons)
        window.statusbar.showMessage('will clean out tiny polygons ' + str(smallerpolygons))
        
        zones = []
        coords = []
        # remove tiny polygons, reorder the zones and close holes
        for n in range(len(base_map['features'])):
            if n not in smallerpolygons:
                feature = base_map['features'][n]
                coords.append([feature['geometry']['coordinates'][0]])
                zones.append(feature['properties']['Zone'])
        
        uniquezones = np.unique(zones)
        featurecoords = [[] for x in range(len(uniquezones))]
        
        for n in range(len(uniquezones)):
            for m in range(len(zones)):
                if zones[m] == uniquezones[n]:
                    featurecoords[n].append(coords[m])
        
        features = []
        for n in range(len(uniquezones)):
            features.append(
                { 
                    "type": "Feature", 
                    "properties": 
                        { 
                            "Zone": uniquezones[n]
                                }, 
                            "geometry": 
                                { 
                                    "type": "MultiPolygon", 
                                    "coordinates": featurecoords[n]
                                    }
                                })
        
        new_map = geojson.feature.FeatureCollection(features)
        new_map["name"] = "Zone"
                
        with open(zonejson_clean, 'w') as fp:
            geojson.dump(new_map, fp, sort_keys=False, ensure_ascii=False)
        
        # open intermediate zone layer for clipping
        print('open intermediate zone layer for clipping')
        window.statusbar.showMessage('open intermediate zone layer for clipping')
        inDataSource = drv2.Open(zonejson_clean)
        inLayer = inDataSource.GetLayer()
        
        # open base map for clipping
        print('open base map for clipping')
        window.statusbar.showMessage('open base map for clipping')
        inClipSource = drv2.Open(base_vector)
        inClipLayer = inClipSource.GetLayer()
        
        # create output ector file for the final zones
        print('create output ector file for the final zones')
        window.statusbar.showMessage('create output ector file for the final zones')
        outDataSource = drv2.CreateDataSource(zonejson)
        outLayer = outDataSource.CreateLayer('Zones', srs = None )
        
        # perform smoothing if activated
        if config.get('smoothing', 0):
            # smooth the output vector
            print('smooth the output vector')
            smooth_size = gridsize * 2.0
            smooth(zonejson_clean,zonejson_smooth,dst_layername,smooth_size)
            # open base map for clipping
            print('open smoothed map for clipping')
            inDataSource = drv2.Open(zonejson_smooth)
            inLayer = inDataSource.GetLayer()
        
        # clip intermediate zone layer with base map
        print('clip intermediate zone layer with base map')
        window.statusbar.showMessage('clip intermediate zone layer with base map')
        ogr.Layer.Clip(inLayer, inClipLayer, outLayer)
        
        # close data sources
        print('close the data sources')
        
        inDataSource = None
        inClipSource = None
        outDataSource = None
        
        
        print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print('Done in ' + str(convert(time.time() - start_time)))
        window.statusbar.showMessage('Done in ' + str(convert(time.time() - start_time)))
        
    # --- Load values into the UI ---
    window.outdirlabel.setText(config.get('outdir'))
    window.inputlayerlabel.setText(os.path.basename(config.get('base_vector',{}).get('file')))
    if config.get('interpolation') == 'idw':
        window.radioButtonIDW.setChecked(True)
    else:
        window.radioButtonLinear.setChecked(True)
    window.PEV.setText(str(config.get('ExplainedVariance', 70)))
    window.gridsizelineEdit.setText(str(config.get('gridsize', 0.05)))
    for rotationalg in config.get('rotation').get('types'):
        window.rotationDcombobox.addItem(rotationalg)
    window.rotationDcombobox.setCurrentIndex(int(config.get('rotation').get('indx')))
    if config.get('inputFormat') == "CSV":
        window.CSVRadio.setChecked(True)
    else:
        window.NetCDFRadio.setChecked(True)
        # window.predictandparamcombobox.addItem(config.get('predictandattr', ''))
    window.missingvalueslineEdit.setText(str(config.get('predictandMissingValue', -9999)))  
    if config.get('composition') == "Sum":
        window.cumRadio.setChecked(True)
    if config.get('composition') == "Average":
        window.avgRadio.setChecked(True)
    for periodx in config.get('period').get('season'):
        window.periodComboBox.addItem(periodx)
    window.periodComboBox.setCurrentIndex(int(config.get('period').get('indx')))
    window.startyearLineEdit.setText(str(config.get('startyr')))
    window.endyearLineEdit.setText(str(config.get('endyr')))
    window.ZonesLineEdit.setText(str(config.get('zones', '')))
    for fileName in config.get('predictandList'):
        window.predictandlistWidget.addItem(os.path.basename(fileName))
    

    ## Signals
    window.outputButton.clicked.connect(getOutDir)
    window.InputLayerButton.clicked.connect(addBaseVector)
    window.radioButtonIDW.toggled.connect(change_interpolation)
    window.radioButtonLinear.toggled.connect(change_interpolation)
    window.CSVRadio.toggled.connect(change_format_type)
    window.NetCDFRadio.toggled.connect(change_format_type)
    window.cumRadio.toggled.connect(change_composition)
    window.avgRadio.toggled.connect(change_composition)
    window.browsepredictandButton.clicked.connect(addPredictands)
    window.clearpredictandButton.clicked.connect(clearPredictands)
    window.runButton.clicked.connect(launch_zoning_Thread)
    # window.stopButton.clicked.connect(closeapp)
    sys.exit(app.exec_())

    
