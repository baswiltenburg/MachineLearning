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

### OPTION 1. GENERATE BOUNDING BOXES BASED ON EXISTING BOUNDING BOXES ###
json_file = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/stedelijkCirRgb256px.json"
image_dir = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/training_corrected"
bounding_boxes, ids = dp.getBoundingBoxesofImages(json_file = json_file, image_dir = image_dir)

### OPTION 2. GENERATE BOUNDING BOXES BASED ON SHAPEFILE OR AREA EXTENT ###
area = [118210.000, 454000.000, 123000.000,  465500.000]
bounding_boxes, ids = dc.getBoundingBoxesAreaExtent(area)

### DOWNLOAD CIR AND RGB IMAGES ###
url = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/infrarood/wms?&request=GetCapabilities&service=WMS')
url_rgb = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?&request=GetCapabilities&service=WMS')
dest_folder_cir = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/cir"
dest_folder_rgb = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/rgb"
for i in range(len(ids)):
    name_2016 = f"{ids[i]}_2016_cir256px.tif"
    #name_2017 = f"{ids[i]}_2017_cir256px.tif"
    rgb_name_2016 = f"{ids[i]}_2016_rgb256px.tif"
    #rgb_name_2017 = f"{ids[i]}_2017_rgb256px.tif"
    bb = bounding_boxes[i]
    valid = False
    while valid == False:
        try:
            dc.downloadTrainingImages(bb, url, '2016_ortho25IR', dest_folder_cir, name = name_2016)
            #dc.downloadTrainingImages(bb, url, '2017_ortho25IR', dest_folder_cir, name = name_2017)
            dc.downloadTrainingImages(bb, url_rgb, '2016_ortho25', dest_folder_rgb, name = rgb_name_2016)
            #dc.downloadTrainingImages(bb, url_rgb, '2017_ortho25', dest_folder_rgb, name = rgb_name_2017)
            print(f"Succesfully downloaded image id: {i}")
            valid = True
        except:
            print("Error, try again...")


### READ CIR IMAGES IN 3 DIMENSIONAL NUMPY ARRAY ###
cir_path = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/cir"
tiles = os.listdir(cir_path)
array_list = []
for tile in tiles:
    if tile.endswith('.tif') and not tile.endswith('_mask.tif'):
        tile_path = f"{cir_path}/{tile}"
        array = rio.open(tile_path, 'r').read() 
        array = array.transpose((1, 2, 0))  
        array = np.expand_dims(array, axis=0)
        array_list.append(array)
# Resulting array (3 dimensional)
cir_array = np.concatenate(array_list) 
### READ RGB IMAGES IN 3 DIMENSIONAL NUMPY ARRAY ###
rgb_path = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/rgb"
tiles = os.listdir(rgb_path)
array_list = []
for tile in tiles:
    if tile.endswith('.tif') and not tile.endswith('_mask.tif'):
        tile_path = f"{rgb_path}/{tile}"
        array = rio.open(tile_path, 'r').read()    
        array = array.transpose((1, 2, 0))  
        array = np.expand_dims(array, axis=0)
        array_list.append(array)
# Resulting array (3 dimensional)
rgb_array = np.concatenate(array_list) 

### NORMALIZE CIR AND RGB IMAGES AND SAVE NORMALIZATION STATISTICS (OPTIONAL) ###
csv_folder_rgb = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/rgb"
csv_folder_cir = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/cir"
rgb_normalized, stats_rgb = dp.NormalizeTrainingData(rgb_array, csv_folder_rgb, 'NormalizationRGB.csv')
cir_normalized, stats_cir = dp.NormalizeTrainingData(cir_array, csv_folder_cir, 'NormalizationCIR.csv')

### APPLY NORMALIZATION AND WRITE NORMALIZED RGB AND CIR IMAGES TO FILE ###
rgb_path = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/rgb"
cir_path = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/cir"
dest_folder_rgb = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/rgb/normalized"
dest_folder_cir = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/cir/normalized"
### FOLDER NORMALIZATION STATS ### 
csv_rgb = work_directory + "/Data/5_TrainingData/Gras/Images/BGT/2016_2017_RgbCirAhn/rgb/NormalizationRGB.csv"
csv_cir = work_directory + "/Data/5_TrainingData/Gras/Images/BGT/2016_2017_RgbCirAhn/cir/NormalizationCIR.csv"
tiles = os.listdir(rgb_path)
for tile in tiles:
    if tile.endswith('.tif'):
        tile_path = f"{rgb_path}/{tile}"
        dp.NormalizeImage(tile_path, tile, dest_folder_rgb, csv_rgb)
tiles = os.listdir(cir_path)
for tile in tiles:
    if tile.endswith('.tif'):
        tile_path = f"{cir_path}/{tile}"
        dp.NormalizeImage(tile_path, tile, dest_folder_cir, csv_cir)

### DOWNLOAD DSM AND DTM ###
url = "https://geodata.nationaalgeoregister.nl/ahn3/wcs"
dest_folder_dsm = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/dsm"
dest_folder_dtm = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/dtm"
dsm = 'ahn3_05m_dsm'
dtm = 'ahn3_05m_dtm'
for i in range(len(ids)):
    name_dsm = f"{ids[i]}_unknown_dsm256px.tif"
    name_dtm = f"{ids[i]}_unknown_dtm256px.tif"
    bb = bounding_boxes[i]
    dc.downloadAhn3Images(server = url, layer = dsm, bounding_box = bb, dest_folder = dest_folder_dsm, name = name_dsm)
    dc.downloadAhn3Images(server = url, layer = dtm, bounding_box = bb, dest_folder = dest_folder_dtm, name = name_dtm)

### CALCULATE VEGETATION HEIGHT AND NORMALIZE ###
name = "unknown_vegetationHeight.tif"
dest_folder_heigth = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/height"
dtm_path = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/dtm"
dsm_path = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/dsm"
dmp.calculateVegetationHeight(dsm_path = dsm_path, dtm_path = dtm_path, dest_path = dest_folder_heigth, name = name)

### CALCULATE SLOPE AND NORMALIZE ###
dest_folder_slope = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/slope"
path = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/dsm"
tiles = os.listdir(path)
for tile in tiles:
    if tile.endswith('.tif'):
        tile_path = f"{path}/{tile}"  
        name = f"{(tile.split('_')[0])}_{(tile.split('_')[1])}_slope.tif" 
        dmp.calculate_slope(tile_path, dest_folder_slope, name)  

### CREATE 6 DIMENSIONAL IMAGE ###
dest_folder = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/training_data"
dc.create6dimensionalImage(dest_folder_rgb, dest_folder_cir, dest_folder_heigth, dest_folder_slope, dest_folder, name = 'bgt6Channels.tif')

### READ HEIGHT IMAGES IN 3 DIMENSIONAL NUMPY ARRAY ###
dest_path = work_directory + "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/training_data"
tiles = os.listdir(dest_path)
array_list = []
for tile in tiles:
    tile_path = f"{dest_path}/{tile}"
    array = rio.open(tile_path, 'r').read()
    array = array.transpose((1, 2, 0))  
    array = np.expand_dims(array, axis=0)    
    array_list.append(array)

# Resulting array (3 dimensional) --> normalized
total_array = np.concatenate(array_list)

### COPY MASK IMAGES TO FOLDER -- ONLY FOR TRAINING DATA ###
path_training_data = work_directory + "/Data/5_TrainingData/Gras/Images/BGT/2016_2017_RgbCirAhn/training_data"
path_mask_data = work_directory + "/Data/5_TrainingData/Gras/Images/BGT/2016_2017_RgbCirAhn/training_corrected"
path_mask_data2 = work_directory + "/Data/5_TrainingData/Gras/Images/BGT/2016_2017_RgbCirAhn/mask"
tiles = os.listdir(path_training_data)
masks = os.listdir(path_mask_data2)
for tile in tiles:
    if tile.endswith('.tif'):
        file_id = f"{tile.split('_')[0]}_{tile.split('_')[1]}"
        for mask in masks:
            if mask.endswith('mask.tif'):
                mask_id = f"{mask.split('_')[0]}_{mask.split('_')[1]}"
                if mask_id == file_id:
                    mask_name = f"{tile.split('.')[0]}_mask.tif"
                    shutil.copy(f"{path_mask_data2}/{mask}", (path_training_data+"/"+mask_name)) 
        
### NORMALIZE MASK DATA -- ONLY FOR TRAINING DATA ###
tiles = os.listdir(path_mask_data2)
for tile in tiles:
    tile_path = f"{path_mask_data2}/{tile}"
    with rio.open(tile_path, 'r+') as data:
        meta = data.meta
        meta.update({"dtype":"uint8"})
        array = data.read()
        array = array.reshape(256,256) 
        array_normalized = dp.NormalizeGroundTruth(array)
        meta.update({"dtype":"float32"})
        with rio.open(tile_path, 'w', **meta) as dst:
            dst.write_band(1, array_normalized)


### REMOVE IMAGES WITHOUT MASK -- ONLY FOR TRAINING DATA  (OPTIONAL) ###
path_training_data = work_directory + "/Data/5_TrainingData/Gras/Images/BGT/2016_2017_RgbCirAhn/training_data"
tiles = os.listdir(path_training_data)
import os.path
for tile in tiles:
    if tile.endswith('.tif') and not tile.endswith('_mask.tif'):
        mask_name = f"{tile.split('.')[0]}_mask.tif"
        mask_path = f"{path_training_data}/{mask_name}"
        if not os.path.exists(mask_path):
            # Delete training image if mask not exists
            path_training_image = f"{path_training_data}/{tile}"
            os.unlink(path_training_image)

### CREATE H5 FILES FOR TRAINING ###
test_data = work_directory +  "/Data/5_TrainingData/Gras/Images/Stedelijk_testing/2016_2017_RgbCirAhn/training_data"
dp.CreateH5_files(test_data, name_file = "stedelijk6ChannelsTest.h5")

 
