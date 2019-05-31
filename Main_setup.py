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

# Data aquisition
# - Download aerial images based on known polygons
# - Create raster masks (binary and non-binary)
# - Create and rename check images
# - Create folder of valid training images with corresponding raster mask
# - Save image data (bounding boxes) to JSON 

# RUN FUNCTIONS #
#wms ='https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?&request=GetCapabilities&service=WMS'
#wms_ir 'https://geodata.nationaalgeoregister.nl/luchtfoto/infrarood/wms?&request=GetCapabilities&service=WMS'
#shapeLocation = work_directory + "/data/n2000_project/shape/gras_manueel_ahn_2016.shp"
#bounding_boxes_features, areas = dc.getBoundingBoxes(shapeLocation = shapeLocation)
#bounding_boxes_images = dc.createImageBoundingBoxes(bounding_boxes_features, areas)
#bounding_boxes_images = sample(bounding_boxes_images,1000)
#bounding_boxes_images = bounding_boxes_images[:75]
#bounding_boxes_images_list = []
#shapeLocation = work_directory+"/data/n2000_project/shape/Gras_bos_handmatig.shp"
#bounding_boxes_images = dc.createImageBoundingBoxes2(shapeLocation = shapeLocation)
#bounding_boxes_images = sample(bounding_boxes_images,1000)
#bounding_boxes_images_list.append(bounding_boxes_images)

#shapeLocation = work_directory + "/data/n2000_project/shape/gras_manueel_ahn_2016.shp"
#bounding_boxes_images = dc.createImageBoundingBoxes2(shapeLocation = shapeLocation)
#bounding_boxes_images = sample(bounding_boxes_images, 700)
# bounding_boxes_images_list.append(bounding_boxes_images)

# bounding_boxes_images_totaal = []
# for i in range(len(bounding_boxes_images_list)):
#     for bb in bounding_boxes_images_list[i]:
#         bounding_boxes_images_totaal.append(bb)

#dc.downloadTrainingImages(bounding_boxes_images, path_rgb_data, name = "grasManueelAhn256pxRGB", ir = False, years = ['2016'])
#dc2.downloadTrainingImages(bounding_boxes_images, path_cir_data, name = "grasManueelAhn256pxIR", ir = True, years = ['2016'])
#dc.saveImageDataToJson(image_directory = path_rgb_data, bounding_boxes_images = bounding_boxes_images, file_name = "grasManueelAhn256px.json", image_size = (256,256))
#mask = work_directory+"/data/n2000_project/shape/gras_manueel_ahn_2016_clipLayer2.shp"
#dc.createRasterMasks(path_rgb_data, path_mask_data, mask)  

#dc.convertMaskToBinaryMask(src_folder = path_mask_data, dst_folder = path_mask_data_binair)
#mask = work_directory+"/data/n2000_project/shape/Gras_bos_handmatig_clipLayer_dissolve.shp"
#dc.createCheckingImages(path_original_data, path_check_images, mask)
#dc.createZipfile(path_check_images, filename = "grasManueel256pxIR_check.gzip")
#dc.createZipfile(path_training_data, filename = "bgt_Gras_256pxIR.gzip")

#dc.create4dimensionalImage(path_rgb_data, path_cir_data, path_training_data, name = 'grasManueelAhn256pxRGB.tif')
#dc.createZipfile(path_training_data, filename = "bgt_Gras_256pxIR.gzip")

# ZIP FOLDERS AND FILES
#import shutil
#zip_name = work_directory + '/notebooks/Bas/Scripts_20190527.zip'
#directory_name = work_directory + '/notebooks/Bas'
# Create 'path\to\zip_file.zip'
#shutil.make_archive(zip_name, 'zip', directory_name)

# Data preprocessing 
# - Clean up data
# - Devide image data in training- validation and test set
# - Normalize image data
# - Perform data augmentation

# RUN FUNCTIONS # 

#dp.RenameCheckImages()
#dp.PrepareTrainingData()
#dp.RemoveInvalidData()
#dp.CreateH5_files("grasManueelAhn256pxCirRgb.h5")
#x_train, y_train, x_val, y_val, x_test, y_test, filenamesTest = dp.DevideData2(path_dataset = (path_training_data+"/bgt_Gras_256pxIR.h5"))
#x_train, y_train, x_val, y_val, x_test, y_test, stats =  dp.NormalizeData(x_train, y_train, x_val, y_val, x_test, y_test)

# Perform data augmentation to improve + increase training dataset
# Create N2000_DataAugmentation object
#da = N2000_DataAugmentation(x_train, y_train)
#x_train_hf, y_train_hf = da.HorizontalFlip(batch_size = 200)
#x_train_vf, y_train_vf = da.VerticalFlip(batch_size = 200)
#x_train_rr, y_train_rr = da.RandomRotation(batch_size = 200)

# Merge original training data with data augmentation
#x_train_total = np.concatenate([x_train,x_train_hf, x_train_vf, x_train_rr])
#y_train_total = np.concatenate([y_train,y_train_hf, y_train_vf, y_train_rr])