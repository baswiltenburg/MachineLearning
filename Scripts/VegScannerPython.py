### SCRIPT VEGETATION SCANNER ###

### Set-up for training a U-net convolutional neural network for sementic segmentation ###
import os
from pathlib import Path
work_directory = 'C:/Users/wba/Internship'
os.chdir(work_directory+'/MachineLearning/Scripts')
from DataPreprocessing import *
from DataCreation import *
import CreateResults as cr
from DemProcessing import *
from Unet import unet, unet_multiclass
from random import sample
from owslib.wms import WebMapService
import tensorflow
from rasterio.plot import reshape_as_raster, reshape_as_image
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as keras
from matplotlib import pyplot
os.chdir(work_directory)

# Set variables and directories
image_size = (256, 256) 
cell_size = 0.25 
epsg = 28992  

# Create processing objects
dc = N2000_Data(image_size = image_size, cell_size = cell_size, epsg = epsg)
dp = N2000_DataPreparation(image_size = image_size)
dmp = DEM_Processing(image_size = (256,256), cell_size = 0.25, epsg = 28992)

### CREATE 4 DIMENSIONAL IMAGE FROM CIR, RGB, HEIGHT: BLUE,GREEN, NDVI, HEIGHT ###
def create4dimensionalImage2(self, rgb_path, cir_path, height_path, dest_folder, name = 'manueel4Channels.tif'):
    # path_rgb_data = folder with rgb images
    # path_cir_data = folder with cir images (should have the same id's and extents)
    # dest folder = destination folder of 4 channel (Green, blue, NDVI, Height) rasters
    # general name of the raster images (prefixed by known id)

    # Import normalization function
    work_directory = 'C:/Users/wba/Internship'
    os.chdir(work_directory+'/MachineLearning/Scripts')
    from DemProcessing import *
    os.chdir(work_directory)
    dmp = DEM_Processing(image_size = (256,256), cell_size = 0.25, epsg = 28992)

    rgb_images = os.listdir(rgb_path)
    cir_images = os.listdir(cir_path)
    height_images = os.listdir(height_path)
    for i in range(len(rgb_images)):
        print(i)
        if rgb_images[i].endswith('.tif'):
            img = rgb_images[i]
            img_id = f"{img.split('_')[0]}_{img.split('_')[1]}"
            for j in range(len(cir_images)):
                if cir_images[j].endswith('.tif'):
                    img2 = cir_images[j]
                    img_id2 = f"{img2.split('_')[0]}_{img2.split('_')[1]}"
                    if img_id == img_id2:
                        for k in range(len(height_images)):
                            if height_images[k].endswith('.tif'):
                                img3 = height_images[k]
                                img_id3 = f"{img3.split('_')[0]}"
                                if img_id3 == img_id2.split('_')[0]:
                                    with rio.open(rgb_path + "/" + rgb_images[i]) as src:
                                        blue = src.read(3)
                                        green = src.read(2)
                                        red = src.read(1)
                                    with rio.open(cir_path + "/" + cir_images[j]) as src2:
                                        nir = src2.read(1)
                                    with rio.open(height_path + "/" + height_images[k]) as src3:
                                        height = src3.read(1)
                                        if np.max(height) > 1:
                                            print(np.max(height), i)
                                    
                                    check = np.logical_and ( red > 0, nir > 0 )
                                    ndvi = np.where (check,  (nir - red ) / ( nir + red ), 0.5) 
                                    ndvi_normalized = dmp.NormalizeData(ndvi, -1, 1)
                                    bands = [blue, green, ndvi_normalized, height]
                                    # Update meta to reflect the number of layers
                                    meta = src.meta
                                    meta.update(count = 4)

                                    # Read each layer and write it to stack
                                    with rio.open(dest_folder + "/" + img_id + "_" + name, 'w', **meta) as dst:
                                        for id, layer in enumerate(bands, start=1):
                                            dst.write_band(id, layer)



