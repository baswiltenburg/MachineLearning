### Run sementic segmentation model U-net with pretrained weights ###
import os
work_directory = 'C:/Users/wba/Internship'
os.chdir(work_directory+'/MachineLearning/Scripts')
from pathlib import Path
from DataPreprocessing import *
from DataCreation import N2000_Data
from Unet import unet
import CreateResults as cr
import tensorflow
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as keras
os.chdir(work_directory)

# Set variables and directories
# Set variables and path directories
image_size = (512, 512) 
cell_size = 0.25 
epsg = 28992    
path_training_data = work_directory + "/Data/testing"
path_mask_data_binair = work_directory 
path_original_data = work_directory
h5Path = work_directory + "/Data/h5130/testing/H5130_Drenthe.h5"
json_path = work_directory + "/Data/h5130/testing/H5130_Drenthe.json"
stats_file = work_directory + "/Data/h5130/NormalizationStatsH5130.csv"
geotiff_folder = work_directory + "/Data/h5130/predictions"
model_path = work_directory + '/Data/h5130/checkpoints/JsonModels/Run2_H5130.json'
model_weights = work_directory + '/Data/h5130/checkpoints/Run2_weights_best.h5'
# Create N2000_DataPreparation object
dp = N2000_DataPreparation(image_size, path_training_data, path_mask_data_binair, path_original_data)

# Load and normalize test data with training data statistics
# Normalize test dataset
x_test, y_test = dp.ApplyNormalization(h5Path, stats_file)

# Run the U-net convolutional neural network on unseen data

# Open saved U-net model from json
json_file = open(model_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
# Convert Json model to Tensorflow ConvNetmodel
loaded_model = model_from_json(loaded_model_json)
# Load weights into the model
loaded_model.load_weights(model_weights)
print("Loaded model from disk")

# Predict images and plot predicted images (code on Azure)
# evaluate the model
loaded_model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
scores = loaded_model.evaluate(x_test, y_test, verbose=0, batch_size = 4) 
print("%s: %.2f%%" % (loaded_model.metrics_names[1], scores[1]*100))

# Make predictions and write prediction to geotiff
images, masks, filenames = cr.readH5File(h5Path)
x_test, y_test = dp.ApplyNormalization(h5Path, stats_file)
for i in range(len(x_test)):
    j = i+1
    image = x_test[i:j]
    filename = filenames[i:j][0]
    prediction = loaded_model.predict(image, batch_size = 4)
    prediction = np.squeeze(prediction, axis = 0)
    cr.writeGeoTiff(json_path, prediction, filename, geotiff_folder)


cr.vectorizePrediction(geotiff_folder, cat = 'h5130_prob90', prob_treshold = 0.9)


# SHOW RESULTS ON TEST SET --> OUD --> NOT GEOTIFF
import scipy
predictions = loaded_model.predict(x_test, batch_size = 4)
predictions = predictions[0:20]

for i in range(len(predictions)):
    pred =  predictions[i].reshape(512,512)
    img = x_test[i]
    gt = y_test[i].reshape(512,512)

    scipy.misc.imsave(geotiff_folder + '/Test_img_' +str(i)+".tif", img)
    scipy.misc.imsave(geotiff_folder + '/Test_gt_' + str(i)+".tif", gt)
    scipy.misc.imsave(geotiff_folder + '/Test_prediction_'+str(i)+".tif", pred)


