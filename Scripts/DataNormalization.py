import os
import io
import shutil
import gdal, osr
import numpy as np
import h5py
import rasterio as rio
from PIL import Image
import json
import random

class DataNormalization():   

    def __init__(self, image_size): 
        self.image_size = image_size

    # NORMALIZE TRAINING DATA AND GROUND TRUTH DATA AND APPLY THE STATISTICS TO VALIDATION AND TEST DATA
    def CalculateNormalizationStatistics(self, np_array, csv_folder, csv_name = "NormalizationStats.csv", write_stats = True):
        if self.image_size[0] <260:        
            if len(np_array) > 1200:
                indexes_np_array = list(np.random.choice(len(np_array), 1200))
                np_array_norm = np_array[indexes_np_array]
            else:
                np_array_norm = np_array
        else:
            if len(np_array) > 500:
                indexes_np_array = list(np.random.choice(len(np_array), 500))
                np_array_norm = np_array[indexes_np_array]
            else:
                np_array_norm = np_array

        np_array_norm = np_array_norm.astype('float32')
        # Calculate normalization statistics
        mean = np.mean(np_array_norm)  # mean for data centering (derived from training data !)
        std = np.std(np_array_norm)  # std for data normalization (derived from training data !)
        np_array_normalized = np_array_norm - mean
        np_array_normalized /= std
        max_value = np.max(np_array_normalized)
        min_value = np.min(np_array_normalized)
        min_value_original = np.min(np_array_norm)
        max_value_original = np.max(np_array_norm)

        stats = [mean, std, min_value, max_value, min_value_original, max_value_original]
        def format(value):
            return "%.3f" % value        
        stats_formatted = [format(v) for v in stats]
        
        if write_stats == True:
            with open(f"{csv_folder}/{csv_name}",'w') as resultFile:
                for stat in stats_formatted:
                    print(stat)
                    resultFile.write(stat)
                    resultFile.write("\n")
        return (stats)

    def NormalizeData(self, stats_file, img_folder = "", h5_file = None, mask = False, write = False, dest_folder = "", array_result = False):
        # stats file:               csv- file with mean, sigma, min and max
        # img_folder:               folder location of tif images
        # h5_file:                  path to h5 file with numpy arrays
        # mask:                     Whether the h5-file contains also maks arrays
        # write:                    write images to file (only possilble when reading from img_folder)
        # dest_folder:              Location of normalized images to be written (only applicable when write == True)
        if write == False and array_result == False:
            return('Either write or array_result must be true..')
        # Read statsfile
        stats_list = []
        with open(stats_file, 'r') as stats:
            for stat in stats:
                stats_list.append(float(stat))
        mean = stats_list[0]
        std = stats_list[1]
        min_value = stats_list[2]
        max_value = stats_list[3]
   
        # If data is stored in an Array
        if h5_file is not None:
            img_array = []
            # Load training data
            file = h5py.File(h5_file, 'r')
            img_x = np.array(file.get('images'))
            for i in range(len(img_x)):
                array = img_x[i]
                array = array.astype('float32')
                array -= mean # mean subtraction
                array /= std  # normalization  
                array = (array-min_value)/(max_value-min_value)  
                img_array.insert(len(img_array), array)
            img_x = np.array(img_array)
            if mask == True:
                mask_array = []
                img_y = np.array(file.get('masks')) 
                for j in range(len(img_y)):
                    array - img_y[j]
                    array = array.astype('float32')
                    array /= 255
                    mask_array.insert(len(mask_array), array)
                img_y = np.array(mask_array)                
                return(img_x, img_y)
            return(img_x)
        # If images are in folder (.tif file)
        else:
            array_normalized = []
            files = os.listdir(img_folder)
            for file in files:
                array = None
                if file.endswith('.tif') and not file.endswith('_mask.tif'):
                    file_path = f"{img_folder}/{file}"
                    # Read image as numpy array
                    with rio.open(file_path, "r") as data:
                        array = data.read()   
                        meta = data.meta   
                    array = array.astype('float32')  
                    # Normalize image
                    array -= mean # mean subtraction
                    array /= std  # normalization    
                    # Rescale normalized images to 0 - 1
                    array = (array-min_value)/(max_value-min_value)
                    if array_result == True and write == True:
                        array_normalized.insert(len(array_normalized), array)
                    if array_result == True and write == False:
                        array = array.transpose((1, 2, 0)) 
                        array_normalized.insert(len(array_normalized), array)
                    if write == True:
                        # Write normalized image to file
                        meta.update({"dtype":"float32"})
                        with rio.open(f"{dest_folder}/{file}", "w", **meta) as dst:
                            dst.write(array)
                            dst.close()
                            data.close()
            if array_result == True:
                array_normalized = np.array(array_normalized)
            return(array_normalized)

    def NormalizeDataMinMax(self, stats_file, np_array, csv_folder, csv_name = "NormalizationStats.csv"):
        array_list = []
        with open(stats_file, 'r') as stats:
            for stat in stats:
                stats_list.append(float(stat))
        min_value = stats_list[4] # original min value
        max_value = stats_list[5] # original max value
        for i in range(len(np_array)):
            array = np_array[i]
            array = (array-min_value)/(max_value-min_value)
            array_list.insert(len(array_list), array)
        result_array = np.array(array_list)
        return(result_array)

    # Convert from 0-255 to scale 0-1
    def NormalizeMaskData(self, mask_folder = "", y_train = None, write = False):
        mask_array = []
        # If masks are read from array:
        if y_train is not None:
            for i in range(len(y_train)):
                array = y_train[j]
                array = array.astype('float32')    
                array /= 255
                mask_array.insert(len(mask_array), array)
            mask_array = np.array(mask_array)
            return(mask_array) 
        else:                
            files = os.listdir(mask_folder)
            for file in files:
                array = None
                if file.endswith('_mask.tif'):
                    file_path = f"{mask_folder}/{file}"
                    # Read image as numpy array
                    with rio.open(file_path, "r+") as data:
                        array = data.read()     
                        meta = data.meta
                        meta.update({"dtype":"float32"})
                    array = array.astype('float32')  
                    array /= 255
                    mask_array.insert(len(mask_array), array)
                    if write == True:                            
                        with rio.open(file_path, 'w', **meta) as dst:
                            dst.write(array)
            mask_array = np.array(mask_array)
            return(mask_array)       
            
 