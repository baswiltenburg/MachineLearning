### Set-up for training a U-net convolutional neural network for sementic segmentation ###
import os
work_directory = 'C:/Users/wba/Internship'
os.chdir(work_directory+'/MachineLearning/Scripts')
import VegScannerPython as vs
from DataCreation import *
from DataPreprocessing import *
#from DataAugmentation import *
from DataNormalization import *
import CreateResults as cr
from DemProcessing import *
import numpy as np
import rasterio as rio
from rasterio.plot import reshape_as_raster, reshape_as_image
from random import sample
from owslib.wms import WebMapService
os.chdir(work_directory)

# Create processing objects
dc = N2000_Data(image_size = (256, 256), cell_size = 0.25, epsg = 28992)
dp = N2000_DataPreparation(image_size = (256, 256))
dmp = DEM_Processing(image_size = (256,256), cell_size = 0.25, epsg = 28992)
dn = DataNormalization(image_size = (512,512))

### OPTION 1. GENERATE BOUNDING BOXES BASED ON EXISTING BOUNDING BOXES ###
json_file = work_directory + "/Data/5_TrainingData/Gras/Images/From_model/2016_2017_RgbCirAhn/stedelijkCirRgb256px.json"
image_dir = work_directory + "/Data/5_TrainingData/Gras/Images/From_model/2016_2017_RgbCirAhn/training_corrected"
bounding_boxes, ids = dp.getBoundingBoxesofImages(json_file = json_file, image_dir = image_dir)

### OPTION 2. GENERATE BOUNDING BOXES BASED ON SHAPEFILE OR AREA EXTENT ###
area = [93000.000, 444000.000, 95000.000,  446000.000]
bounding_boxes, ids = dc.getBoundingBoxesAreaExtent(area)

### DOWNLOAD CIR AND RGB IMAGES ###
url_cir = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/infrarood/wms?&request=GetCapabilities&service=WMS')
url_rgb = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?&request=GetCapabilities&service=WMS')
cir_path = work_directory + "/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCirAhn/cir"
rgb_path = work_directory + "/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCirAhn/rgb"
for i in range(len(ids)):
    name_2016 = f"{ids[i]}_2016_cir256px.tif"
    name_2017 = f"{ids[i]}_2017_cir256px.tif"
    rgb_name_2016 = f"{ids[i]}_2016_rgb256px.tif"
    rgb_name_2017 = f"{ids[i]}_2017_rgb256px.tif"
    bb = bounding_boxes[i]
    valid = False
    while valid == False:
        try:
            dc.downloadTrainingImages(bb, url_cir, '2016_ortho25IR', cir_path, name = name_2016)
            dc.downloadTrainingImages(bb, url_cir, '2017_ortho25IR', cir_path, name = name_2017)
            dc.downloadTrainingImages(bb, url_rgb, '2016_ortho25', rgb_path, name = rgb_name_2016)
            dc.downloadTrainingImages(bb, url_rgb, '2017_ortho25', rgb_path, name = rgb_name_2017)
            print(f"Succesfully downloaded image id: {i}")
            valid = True
        except:
            print("Error, try again...")

### WRITE JSON FILE (OPTIONAL) ###
dc.saveImageDataToJson(image_directory = rgb_path, bounding_boxes_images = bounding_boxes, file_name = "6dimensional.json")

### READ CIR AND RGB IMAGES IN 3 DIMENSIONAL NUMPY ARRAY ###
cir_array = dp.ImagesToArray(img_folder = cir_path, mask = False)
rgb_array = dp.ImagesToArray(img_folder = rgb_path, mask = False)

### NORMALIZE CIR AND RGB IMAGES AND SAVE NORMALIZATION STATISTICS (OPTIONAL) ###
stats_rgb = dn.CalculateNormalizationStatistics(np_array = rgb_array, csv_folder = rgb_path, csv_name= "NormalizationRGB.csv")
stats_cir = dn.CalculateNormalizationStatistics(np_array = cir_array, csv_folder = cir_path, csv_name= 'NormalizationCIR.csv')

### APPLY NORMALIZATION  (OPTIONAL --> AZURE) ###
rgb_normalized = dn.NormalizeData(stats_file = f"{rgb_path}/NormalizationRGB.csv", img_folder = rgb_path, write = True, dest_folder = f"{rgb_path}/normalized")
cir_normalized = dn.NormalizeData(stats_file = f"{cir_path}/NormalizationCIR.csv", img_folder = cir_path, write = True, dest_folder = f"{cir_path}/normalized")

### DOWNLOAD DSM AND DTM ###
url = "https://geodata.nationaalgeoregister.nl/ahn3/wcs"
dtm_path = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/dtm"
dsm_path= "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/dsm"
dsm = 'ahn3_05m_dsm'
dtm = 'ahn3_05m_dtm'
for i in range(len(ids)):
    name_dsm = f"{ids[i]}_unknown_dsm256px.tif"
    name_dtm = f"{ids[i]}_unknown_dtm256px.tif"
    bb = bounding_boxes[i]
    dc.downloadAhn3Images(server = url, layer = dsm, bounding_box = bb, dest_folder = dsm_path, name = name_dsm)
    dc.downloadAhn3Images(server = url, layer = dtm, bounding_box = bb, dest_folder = dtm_path, name = name_dtm)

### CALCULATE VEGETATION HEIGHT AND NORMALIZE ###
name = "unknown_vegetationHeight.tif"
dest_folder_heigth = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/height"
dmp.calculateVegetationHeight(dsm_path = dsm_path, dtm_path = dtm_path, dest_path = dest_folder_heigth, name = name, normalize=True)

### CALCULATE SLOPE AND NORMALIZE AND NORMALIZE ###
dest_folder_slope = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/slope"
tiles = os.listdir(dsm_path)
for tile in tiles:
    if tile.endswith('.tif'):
        tile_path = f"{dsm_path}/{tile}"  
        name = f"{(tile.split('_')[0])}_{(tile.split('_')[1])}_slope.tif" 
        dmp.calculate_slope(tile_path, dest_folder_slope, name, normalize = True)  

### CREATE 6 DIMENSIONAL IMAGE ###
path_training_data = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/training_data_6channels"
dc.CreateMultiDimensionalImage(rgb_path, cir_path, dest_folder_heigth, path_training_data, path_slope_data= dest_folder_slope, name = 'Test6Channels.tif')
vs.CreateRgbNirNdviHeightImage(rgb_path = rgb_path, cir_path =cir_path, height_path=dest_folder_heigth, dest_folder = path_training_data, name = 'RgbNirNdviHeight.tif')

### NORMALIZE MASK DATA  ###
path_mask_data = f"{work_directory}/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCir/mask"
mask_array = dn.NormalizeMaskData(mask_folder = path_mask_data, write = True)
mask_array = None
### COPY MASK IMAGES TO FOLDER -- ONLY FOR TRAINING DATA ###
dp.PrepareTrainingData(path_training_data= path_training_data, path_mask_data = path_mask_data)

### REMOVE IMAGES WITHOUT MASK -- ONLY FOR TRAINING DATA  (OPTIONAL) ###
dp.RemoveInvalidData(path_training_data)

### CREATE H5 FILES FOR TRAINING ###
dp.SaveTrainingData(path_training_data, name_file = "Manueel5Channels.h5")