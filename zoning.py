#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:59:45 2022

@author: thembani
"""


from functions import *
import matplotlib.pyplot as plt
import numpy as np

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

scriptpath = os.path.dirname(os.path.realpath(__file__))
scriptpath = Path(scriptpath)
outdir = Path("/Users/thembani/Documents/PCA")
csvs = ["data/chirps_monthly_1981_2020_ZAF.csv","data/chirps_monthly_1981_2020_LSO.csv", 
        "data/chirps_monthly_1981_2020_SWZ.csv","data/chirps_monthly_1981_2020_BWA.csv",
        "data/chirps_monthly_1981_2020_NAM.csv","data/chirps_monthly_1981_2020_MOZ.csv"]

# csvs = ["data/chirps_monthly_1981_2020_ZAF.csv"]

# csvs = ["data/chirps_monthly_1981_2020_AGO.csv","data/chirps_monthly_1981_2020_BWA.csv",
#         "data/chirps_monthly_1981_2020_COD.csv","data/chirps_monthly_1981_2020_COM.csv",
#         "data/chirps_monthly_1981_2020_LSO.csv","data/chirps_monthly_1981_2020_MDG.csv",
#         "data/chirps_monthly_1981_2020_MOZ.csv","data/chirps_monthly_1981_2020_MWI.csv",
#         "data/chirps_monthly_1981_2020_NAM.csv","data/chirps_monthly_1981_2020_SWZ.csv",
#         "data/chirps_monthly_1981_2020_TZA.csv","data/chirps_monthly_1981_2020_ZAF.csv",
#         "data/chirps_monthly_1981_2020_ZMB.csv","data/chirps_monthly_1981_2020_ZWE.csv"]

startyr = 1981
endyr = 2010
missing = 999.9
ExplainedVariance = 80
gridsize = 0.05
rotation = "quartimax"
period = 'OND'
composition = "Sum"
interpolation = 'idw'
# base_vector = str(Path("/Users/thembani/Documents/Projects/cft/output/sadcboundary.geojson"))
base_vector = str(Path("/Users/thembani/Documents/Projects/cft/output/southafrica.geojson"))
vname, _ = os.path.splitext(os.path.basename(base_vector))
vname = fixname(vname)
proj='GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],' \
     'AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],' \
     'AUTHORITY["EPSG","4326"]]'

# check if output directory exists
if not os.path.exists(outdir):
    print("output directory does not exist")
    sys.exit()

# create output directory
zonepath = (outdir / vname).joinpath(period + '_' + str(startyr)  + '_' + str(endyr))
os.makedirs(zonepath, exist_ok=True)

# outputs 
dst_layername = "izones"
dst_zones = str(zonepath / (dst_layername + ".geojson"))
zonejson = str(zonepath / (vname + "_zones.geojson"))
zonejson_clean = str(zonepath / (dst_layername + "_clean.geojson"))
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
data.to_csv(datacsv)

data.columns = ['s'+str(x) for x in range(len(stations))]
X = StandardScaler().fit_transform(data)
fa = PCA()
fa.fit(X)

print('explained variance')
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

# compute optimal number of zones
print('compute optimal number of zones')
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
# n_clusters = 6

print('clustering of components')
db = cluster.KMeans(n_clusters = n_clusters, random_state=42).fit(comps.T)
zi = db.labels_.reshape(len(ys), len(xs))
print("number of zones ", len(np.unique(db.labels_)))
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
print('close the data sources')
ds = None
dst_ds = None


# clean the vector zone map
print('cleaning the vector zone map')
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
inDataSource = drv2.Open(zonejson_clean)
inLayer = inDataSource.GetLayer()

# open base map for clipping
print('open base map for clipping')
inClipSource = drv2.Open(base_vector)
inClipLayer = inClipSource.GetLayer()

# create voutput ector file for the final zones
print('create output ector file for the final zones')
outDataSource = drv2.CreateDataSource(zonejson)
outLayer = outDataSource.CreateLayer('Zones', srs = None )

# clip intermediate zone layer with base map
print('clip intermediate zone layer with base map')
ogr.Layer.Clip(inLayer, inClipLayer, outLayer)

# close data sources
print('close the data sources')

inDataSource = None
inClipSource = None
outDataSource = None


print("End time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print('Done in ' + str(convert(time.time() - start_time)))


