# A simple utlity to make a rudimentary static map and overlay a sample image and overlay

# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2021, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import sys,os, time, gc
sys.path.insert(1, '../src')

from glob import glob
import numpy as np
from tkinter import filedialog, messagebox
from tkinter import *
import matplotlib.pyplot as plt
from skimage.io import imread
import rasterio
from cartopy import config
import cartopy.crs as ccrs
import pyproj
from matplotlib.transforms import offset_copy
import cartopy.io.img_tiles as cimgt
from tqdm import tqdm
from joblib import Parallel, delayed

from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)

from datetime import datetime
import simplekml

##========================================================
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

##========================================================
def fromhex(n):
    """ hexadecimal to integer """
    return int(n, base=16)

##========================================================
def label_to_colors(
    img,
    mask,
    alpha,#=128,
    colormap,#=class_label_colormap, #px.colors.qualitative.G10,
    color_class_offset,#=0,
    do_alpha,#=True
):
    """
    Take MxN matrix containing integers representing labels and return an MxNx4
    matrix where each label has been replaced by a color looked up in colormap.
    colormap entries must be strings like plotly.express style colormaps.
    alpha is the value of the 4th channel
    color_class_offset allows adding a value to the color class index to force
    use of a particular range of colors in the colormap. This is useful for
    example if 0 means 'no class' but we want the color of class 1 to be
    colormap[0].
    """


    colormap = [
        tuple([fromhex(h[s : s + 2]) for s in range(0, len(h), 2)])
        for h in [c.replace("#", "") for c in colormap]
    ]

    cimg = np.zeros(img.shape[:2] + (3,), dtype="uint8")
    minc = np.min(img)
    maxc = np.max(img)

    for c in range(minc, maxc + 1):
        cimg[img == c] = colormap[(c + color_class_offset) % len(colormap)]

    cimg[mask==1] = (0,0,0)

    if do_alpha is True:
        return np.concatenate(
            (cimg, alpha * np.ones(img.shape[:2] + (1,), dtype="uint8")), axis=2
        )
    else:
        return cimg



root = Tk()
root.filename =  filedialog.askdirectory(initialdir = os.getcwd(),title = "Select directory of model output (NPZ) files")
data_path = root.filename
print(data_path)
root.withdraw()


root = Tk()
root.filename =  filedialog.askopenfilename(title = "Select reference geotiff",filetypes = (("geotiff files","*.tif"),("all files","*.*")))
geotiff = root.filename
print(geotiff)
root.withdraw()


#===============================================================
print('Reading GeoTIFF data ...')
if type(geotiff) is not list:
  geotiff = [geotiff]

## read all arrays
bs = []
for layer in geotiff:
  with rasterio.open(layer) as src:
     layer = src.read()[0,:,:]
  w, h = (src.width, src.height)
  xmin, ymin, xmax, ymax = src.bounds
  crs = src.crs #get_crs()
  del src
  raster =  rasterio.open(geotiff[0])
  gt = raster.get_transform()
  pixelSizeX = gt[1]
  pixelSizeY =-gt[-1]
  bs.append({'bs':layer, 'w':w, 'h':h, 'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax, 'crs':crs, 'pixelSizeX':pixelSizeX, 'pixelSizeX':pixelSizeY})

#get pyproj transformation object
trans = pyproj.Proj(init=bs[0]['crs']['init'])

## resize arrays so common grid
##get bounds
xmax = max([x['xmax'] for x in bs])
xmin = min([x['xmin'] for x in bs])
ymax = max([x['ymax'] for x in bs])
ymin = min([x['ymin'] for x in bs])
## make common grid
yp, xp = np.meshgrid(np.arange(xmin, xmax, pixelSizeX), np.arange(ymin, ymax, pixelSizeY))

## get extents in lat/lon
lonmin, latmin = trans(xmin, ymin, inverse=True)
lonmax, latmax = trans(xmax, ymax, inverse=True)


#blue,red, yellow,green, etc
class_label_colormap = ['#3366CC','#DC3912','#FF9900','#109618','#990099','#0099C6','#DD4477',
                        '#66AA00','#B82E2E', '#316395','#0d0887', '#46039f', '#7201a8',
                        '#9c179e', '#bd3786', '#d8576b', '#ed7953', '#fb9f3a', '#fdca26', '#f0f921']


# g_tiles = cimgt.GoogleTiles(style='satellite')
g_tiles = cimgt.QuadtreeTiles()

img_extent = (lonmin, lonmax, latmin, latmax)

def make_plot(f,class_label_colormap,img_extent,g_tiles,trans, pixelSizeX, pixelSizeY, xmin, ymin):
    #print(f)

    try:
        with np.load(f) as data:
            input_file = data['input_file']
            grey_label = data['grey_label'].astype('float')
            n_data_bands = int(data['n_data_bands'])
            NCLASSES = int(data['nclasses'])

        if NCLASSES>1:
            class_label_colormap = class_label_colormap[:NCLASSES]
        else:
            class_label_colormap = class_label_colormap[:NCLASSES+1]


        #img = imread(str(input_file))
        with np.load(str(input_file)) as data:
            img = data['arr_0']

        color_label = label_to_colors(grey_label.astype('uint8'),
                        img[:,:,0]==0, alpha=128, colormap=class_label_colormap,
                        color_class_offset=0, do_alpha=False)


        grey_label[grey_label!=2]=np.nan
        sy,sx = np.where(~np.isnan(np.flipud(grey_label)))

        a =[]
        for ii in np.unique(sy):
            w = np.where(sy==ii)[0]
            a.append((ii,np.max(sx[w])))
        syi,sxi=zip(*a)
        sxi=np.array(sxi)
        syi=np.array(syi)


        sxii = moving_average(sxi, 3)
        syii = moving_average(syi, 3)

        #
        # # remove duplicates
        # L = []
        # for x,y in zip(sxii, syii):
        #   if (x, y) not in L and (y, x ) not in L:
        #     L.append((x, y))
        # sxii,syii=zip(*L)
        # sxii=np.array(sxii)
        # syii=np.array(syii)


        sxii = xmin+(pixelSizeX*sxii)
        syii = ymin+(pixelSizeY*syii)

        shore_lon, shore_lat = trans(sxii, syii, inverse=True)

        if len(shore_lon)<1:
            shore_lon=np.nan
            shore_lat=np.nan


        ###=====================================
        fig = plt.figure(figsize=(8, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.use_sticky_edges = False
        # set a margin around the data
        ax.set_xmargin(0.18)
        ax.set_ymargin(0.18)

        ax.add_image(g_tiles, 12, zorder=0)

        ax.imshow(img[:,:,:3], origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), zorder=1, cmap='gray')

        if len(shore_lon)>1:
            ax.plot(shore_lon,shore_lat,'r', transform=ccrs.PlateCarree(), zorder=2)

        #ax.imshow(grey_label, origin='upper', alpha=0.5, extent=img_extent, transform=ccrs.PlateCarree(), zorder=2, cmap='bwr_r')
        # ax.coastlines(resolution='10m', color='black', linewidth=1)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, rotate_labels=30)

        gl.bottom_labels = False
        gl.left_labels = False
        #gl.xlines = False

        gl.ylocator = LatitudeLocator()
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
        gl.ylabel_style = {'color': 'black', 'weight': 'bold'}

        #add a markr
        ax.plot(-124.0842416,41.5245625, 'bo', markersize=7, transform=ccrs.PlateCarree())
        ax.text(-124.084, 41.524, 'Flint Rock', color='b', transform=ccrs.PlateCarree())

        ax.plot(-124.0786306,41.5222312, 'yo', markersize=7, transform=ccrs.PlateCarree())
        ax.text(-124.078,41.522, 'Old radar station', color='y',transform=ccrs.PlateCarree())

        ax.text(lonmin,latmin-.0012,f.split(os.sep)[-1].split('_rgb')[0] , color='r', fontsize=12, transform=ccrs.PlateCarree())

        plt.savefig(f.replace('.npz','.png'), dpi=200, bbox_inches='tight')
        plt.close('all')
        del fig, ax

        gc.collect()

        ####=====================================================================

        fig = plt.figure(figsize=(8, 12))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.use_sticky_edges = False
        # set a margin around the data
        ax.set_xmargin(0.2)
        ax.set_ymargin(0.20)

        ax.add_image(g_tiles, 12, zorder=0)

        ax.imshow(img[:,:,:3], origin='upper', extent=img_extent, transform=ccrs.PlateCarree(), zorder=1, cmap='gray')

        ax.imshow(color_label, origin='upper', alpha=0.5, extent=img_extent, transform=ccrs.PlateCarree(), zorder=2, cmap='bwr_r')
        # ax.coastlines(resolution='10m', color='black', linewidth=1)
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, rotate_labels=30)

        gl.bottom_labels = False
        gl.left_labels = False
        #gl.xlines = False

        gl.ylocator = LatitudeLocator()
        gl.xformatter = LongitudeFormatter()
        gl.yformatter = LatitudeFormatter()
        gl.xlabel_style = {'color': 'black', 'weight': 'bold'}
        gl.ylabel_style = {'color': 'black', 'weight': 'bold'}

        #add a markr
        ax.plot(-124.0842416,41.5245625, 'bo', markersize=7, transform=ccrs.PlateCarree())
        ax.text(-124.084, 41.524, 'Flint Rock', color='b', transform=ccrs.PlateCarree())

        ax.plot(-124.0786306,41.5222312, 'yo', markersize=7, transform=ccrs.PlateCarree())
        ax.text(-124.078,41.522, 'Old radar station', color='y',transform=ccrs.PlateCarree())

        if len(shore_lon)>1:
            ax.plot(shore_lon,shore_lat,'k', lw=2,transform=ccrs.PlateCarree(), zorder=3)

        ax.text(lonmin,latmin-.0012,f.split(os.sep)[-1].split('_rgb')[0] , color='r', fontsize=12, transform=ccrs.PlateCarree())

        plt.savefig(f.replace('.npz','_overlay.png'), dpi=200, bbox_inches='tight')
        plt.close('all')

        gc.collect()

        return shore_lon,shore_lat, f.split(os.sep)[-1].split('_rgb')[0]
    except:
        return np.nan, np.nan, np.nan



###===================================================
sample_filenames = sorted(glob(data_path+os.sep+'*.npz'))

w = Parallel(n_jobs=-1, verbose=0, max_nbytes=None)\
            (delayed(make_plot)(f,class_label_colormap,img_extent,g_tiles,trans, pixelSizeX, pixelSizeY, xmin, ymin) for f in tqdm(sample_filenames))

shore_lon, shore_lat, names = zip(*w)

datadict = {}
datadict['shore_lon'] = shore_lon
datadict['shore_lat'] = shore_lat
datadict['names'] = names

np.savez_compressed('out.npz', **datadict)

for counter, (lon,lat,name) in enumerate(zip(shore_lon,shore_lat, names)):
    if type(lon) is not float:
        kml=simplekml.Kml()
        style = simplekml.Style()
        style.labelstyle.color = simplekml.Color.red  # Make the text red
        style.labelstyle.scale = .2  # Make the text twice as big
        style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'

        for l,ll in zip(lon,lat):
          pnt = kml.newpoint(coords=[(l,ll)])
          pnt.style = style

        kml.save('{}.kml'.format(name))
