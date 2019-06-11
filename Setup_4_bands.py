### Set-up for training a U-net convolutional neural network for sementic segmentation ###
import os
work_directory = 'C:/Users/wba/Internship'
os.chdir(work_directory+'/MachineLearning/Scripts')
from DataCreation import *
from DataPreprocessing import *
#from DataAugmentation import *
from DataNormalization import *
import numpy as np
from random import sample
from owslib.wms import WebMapService
import rasterio as rio
#from rasterio.plot import reshape_as_raster, reshape_as_image
os.chdir(work_directory)

# Create processing objects
dc = N2000_Data(image_size = (512,512), cell_size = 0.25, epsg = 28992)
dp = N2000_DataPreparation(image_size = (512,512))
dn = DataNormalization(image_size = (512,512))

### OPTION 1. GENERATE BOUNDING BOXES BASED ON EXISTING BOUNDING BOXES ###
json_file = work_directory + "/Data/5_TrainingData/H5130/Images/Totaal/h5130_totaal_512px.json"
image_dir = work_directory + "/Data/5_TrainingData/H5130/Images/Totaal/2016_CirRgb_testing/rgb"
bounding_boxes, ids = dp.GetBoundingBoxesofImages(json_file = json_file, image_dir = image_dir)

### OPTION 2. GENERATE BOUNDING BOXES BASED ON SHAPEFILE OR AREA EXTENT ###
area = [192000.000, 448000.000, 197000.000,  462000.000]
bounding_boxes, ids = dc.getBoundingBoxesAreaExtent(area)

### DOWNLOAD CIR AND RGB IMAGES ###
url_cir = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/infrarood/wms?&request=GetCapabilities&service=WMS')
url_rgb = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?&request=GetCapabilities&service=WMS')
cir_path = work_directory + "/Data/5_TrainingData/H5130/Images/Gelderland/2016CirRgb/cir"
rgb_path = work_directory + "/Data/5_TrainingData/H5130/Images/Gelderland/2016CirRgb/rgb"
for i in range(len(ids)):
    #name_2016 = f"{ids[i]}_2016_h5130.tif"
    name_2017 = f"{ids[i]}_2017_h5130.tif"
    #name_2018 = f"{ids[i]}_2018_h5130.tif"
    #rgb_name_2016 = f"{ids[i]}_2016_h5130.tif"
    rgb_name_2017 = f"{ids[i]}_2017_h5130.tif"
    bb = bounding_boxes[i]
    valid = False
    while valid == False:
        try:
            dc.downloadTrainingImages(bb, url_cir, '2016_ortho25IR', cir_path, name = name_2016)
            #dc.downloadTrainingImages(bb, url_cir, '2017_ortho25IR', cir_path, name = name_2017)
            #dc.downloadTrainingImages(bb, url_cir, '2018_ortho25IR', cir_path, name = name_2018)
            dc.downloadTrainingImages(bb, url_rgb, '2016_ortho25', rgb_path, name = rgb_name_2016)
            #dc.downloadTrainingImages(bb, url_rgb, '2017_ortho25', rgb_path, name = rgb_name_2017)
            print(f"Succesfully downloaded image id: {i}")
            valid = True
        except:                      
            print("Error, try again...")

### WRITE JSON FILE (OPTIONAL) ###
dc.saveImageDataToJson(image_directory = rgb_path, bounding_boxes_images = bounding_boxes, file_name = "h5130Gelderland.json")

### READ CIR AND RGB IMAGES IN 3 DIMENSIONAL NUMPY ARRAY ###
cir_array = dp.ImagesToArray(img_folder = cir_path, mask = False)
rgb_array = dp.ImagesToArray(img_folder = rgb_path, mask = False)

### NORMALIZE CIR AND RGB IMAGES AND SAVE NORMALIZATION STATISTICS (OPTIONAL) ###
stats_rgb = dn.CalculateNormalizationStatistics(np_array = rgb_array, csv_folder = rgb_path, csv_name= "NormalizationRGB.csv")
rgb_array = None
stats_cir = dn.CalculateNormalizationStatistics(np_array = cir_array, csv_folder = cir_path, csv_name= 'NormalizationCIR.csv')
cir_array = None 

### APPLY NORMALIZATION  (OPTIONAL --> AZURE) ###
rgb_normalized = dn.NormalizeData(stats_file = f"{rgb_path}/NormalizationRGB.csv", img_folder = rgb_path, write = True, dest_folder = f"{rgb_path}/normalized")
cir_normalized = dn.NormalizeData(stats_file = f"{cir_path}/NormalizationCIR.csv", img_folder = cir_path, write = True, dest_folder = f"{cir_path}/normalized")
cir_normalized = None
rgb_normalized = None

### CREATE 4 DIMENSIONAL IMAGE (RGB + NIR) ###
path_training_data = f"{work_directory}/Data/5_TrainingData/H5130/Images/Totaal/2016_CirRgb_testing/training_data"
dc.create4dimensionalImage(path_rgb_data = f"{rgb_path}/normalized", path_cir_data = f"{cir_path}/normalized",  dest_folder = path_training_data, name = 'h5130.tif')

### NORMALIZE MASK DATA  ###
path_mask_data = f"{work_directory}/Data/5_TrainingData/H5130/Images/Totaal/2016_CirRgb_testing/mask"
mask_array = dn.NormalizeMaskData(mask_folder = path_mask_data, write = True)
mask_array = None 

### COPY IMAGES FROM MASK TO TRAINING DATA FOLDER ###
dp.PrepareTrainingData(path_training_data= path_training_data, path_mask_data = path_mask_data)
        
### REMOVE INVALID DATA
dp.RemoveInvalidData(path_training_data = path_training_data)

### CREATE H5 FILES FOR TRAINING ### MEMORY PROBLEM!
dp.SaveTrainingData(path_training_data, name_file = "2016_2017_2018_H5130.h5")

path_mask_data = f"{work_directory}/Data/5_TrainingData/Gras/Images/BGT/2016_2017_RgbCirAhn/mask2"
files = os.listdir(path_mask_data)
for file in files:
    split = file.split("_")
    split[2] = "Bgt5Channels"
    newname = f"{split[0]}_{split[1]}_{split[2]}_mask.tif"
    os.rename(f"{path_mask_data}/{file}", f"{path_mask_data}/{newname}")