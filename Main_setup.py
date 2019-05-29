### Set-up for training a U-net convolutional neural network for sementic segmentation ###
import os
from pathlib import Path
work_directory = 'C:/Users/wba/Internship'
os.chdir(work_directory+'/MachineLearning/Scripts')
from DataPreprocessing import *
from DataCreation import *
import CreateResults as cr
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
wms = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?&request=GetCapabilities&service=WMS')
wms_ir = WebMapService('https://geodata.nationaalgeoregister.nl/luchtfoto/infrarood/wms?&request=GetCapabilities&service=WMS')
# path_training_data = work_directory + "/data/n2000_project/training_images/gras_manueel_ahn/training_corrected"
# path_mask_data_binair = work_directory+ "/data/n2000_project/training_images/gras_manueel_ahn/binary_mask"
# path_mask_data = work_directory + "/data/n2000_project/training_images/gras_manueel_ahn/mask"
# path_cir_data = work_directory + "/data/n2000_project/training_images/gras_manueel_ahn/training_cir"
# path_rgb_data = work_directory + "/data/n2000_project/training_images/gras_manueel_ahn/training_rgb"
# folder_checkpoints = work_directory

# Create N2000_Data object
dc = N2000_Data(image_size = image_size, cell_size = cell_size, epsg = epsg)

# Create N2000_DataPreparation object
dp = N2000_DataPreparation(image_size = image_size)

# Data aquisition
# - Download aerial images based on known polygons
# - Create raster masks (binary and non-binary)
# - Create and rename check images
# - Create folder of valid training images with corresponding raster mask
# - Save image data (bounding boxes) to JSON 

# RUN FUNCTIONS #
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

# CHECK VARIABLES # CHECK FOLDER

#dc.convertMaskToBinaryMask(src_folder = path_mask_data, dst_folder = path_mask_data_binair)
#mask = work_directory+"/data/n2000_project/shape/Gras_bos_handmatig_clipLayer_dissolve.shp"
#dc.createCheckingImages(path_original_data, path_check_images, mask)
#dc.createZipfile(path_check_images, filename = "grasManueel256pxIR_check.gzip")
#dc.createZipfile(path_training_data, filename = "bgt_Gras_256pxIR.gzip")


#dc.create4dimensionalImage(path_rgb_data, path_cir_data, path_training_data, name = 'grasManueelAhn256pxRGB.tif')
# dc.createZipfile(path_training_data, filename = "bgt_Gras_256pxIR.gzip")

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


# Train U-net convolutional neural network for semantic segmentation
# Initialize Unet model
model = unet(input_size = (256, 256, 3), drop_out = 0.2, lr = 0.00005)

# Checkpoints
checkpoint = ModelCheckpoint((folder_checkpoints + '/Run1_weights_best.h5'), monitor='val_acc', verbose = 1, save_best_only=True, mode = "max", period = 1)
checkpoint2 = ModelCheckpoint((folder_checkpoints + '/Run1_weights_best_loss.h5'), monitor='val_loss', verbose = 1, save_best_only=True, mode = "min", period = 1)
tensorboard = TensorBoard(log_dir='tensorboard/', write_graph=True, write_images=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 4)
# Reduce learning rate on plateu
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0000001)
# CSV Logger
csv_logger = CSVLogger(folder_checkpoints + '/Run1_training.csv', append = True, separator = ',')

calbacks_list = [checkpoint, checkpoint2, tensorboard, es, reduce_lr]

# Model training
history =  model.fit(x_train_total, y_train_total, validation_data = (x_val, y_val), batch_size=8, epochs=30, verbose=2, shuffle=True, callbacks=calbacks_list)


# Initialize Unet model
model = unet(input_size = (256, 256, 3), drop_out = 0.2, lr = 0.00005)


with open(folder_checkpoints + '/Run1_trainHistory', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
# import json
# with open('file.json', 'w') as f:
#     json.dump(history.history, f)

# evaluate the model
_, train_acc = model.evaluate(x_train_total, y_train_total, verbose=0)
_, test_acc = model.evaluate(x_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))


# tensorboard --logdir=/full_path_to_your_logs
# plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# evaluate the model
scores = model.evaluate(x_train_total, y_train_total, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

model = unet_multiclass(n_classes = 2, input_size = (256, 256, 3), drop_out = 0.2, lr = 0.00005)
model.summary()






### DOWNLOAD DIGITAL SURFACE MODEL AND DIGITAL TERRAIN MODEL BASED ON EXISTING BOUNDING BOXES ###
dsm = 'ahn3_05m_dsm'
dtm = 'ahn3_05m_dtm'
json_file = work_directory + "/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCirAhn/grasManueel256pxIR.json"
image_dir = work_directory + "/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCirAhn/training_corrected"
bounding_boxes, ids = dp.getBoundingBoxesofImages(json_file = json_file, image_dir = image_dir)
dest_folder = work_directory + "/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCirAhn/dsm"
url = "https://geodata.nationaalgeoregister.nl/ahn3/wcs"
for i in range(len(ids)):
    name = f"{ids[i]}_unknown_Ahn256px"
    bb = bounding_boxes[i]
    dc.downloadAhn3Images(server = url, layer = dsm, bounding_box = bb, dest_folder = dest_folder, name = name)

dest_folder = work_directory + "/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCirAhn/dtm"
for i in range(len(ids)):
    name = f"{ids[i]}_unknown_Ahn256px"
    bb = bounding_boxes[i]
    dc.downloadAhn3Images(server = url, layer = dtm, bounding_box = bb, dest_folder = dest_folder, name = name)


### CALCULATE VEGETATION HEIGHT ###
name = "unknown_vegetationHeight"
dest_path = work_directory + "/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCirAhn/height"
dtm_path = work_directory + "/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCirAhn/dtm"
dsm_path = work_directory + "/Data/5_TrainingData/Gras/Images/Manueel/2016_2017_RgbCirAhn/dsm"
dc.calculateHeight(dsm_path, dtm_path, dest_path, name)

### READ HEIGHT IMAGES IN 3 DIMENSIONAL NUMPY ARRAY ###
tiles = os.listdir(dest_path)
array_list = []
for tile in tiles:
    tile_path = f"{dest_path}/{tile}"
    array = rio.open(tile_path, 'r').read()    
    array_list.append(array)
# Resulting array (3 dimensional)
result_array = np.concatenate(array_list)

### NORMALIZE TRAINING DATA ###
x_train, stats = dp.NormalizeTrainingDataMinMax(result_array, dest_path, "normalizationStatsVegetationHeight.csv")   

### CREATE IMAGE OF 5 CHANNELS ###

### CALCULATE SLOPE AND ASPECT ###

### CREATE IMAGE OF 7 CHANNELS ###
    