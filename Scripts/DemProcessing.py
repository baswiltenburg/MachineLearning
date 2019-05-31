import matplotlib.pyplot as plt
import os
import io
import shutil
import gdal, osr
import os, struct
import numpy as np
import h5py
import rasterio as rio
import os
from PIL import Image
import json
from osgeo import gdal

class DEM_Processing(object):          
    def __init__(self, image_size = (256,256), cell_size = 0.25, epsg = 28992):
        # Variables        
        self.image_size = image_size
        self.cell_size = cell_size
        self.epsg = epsg


    def createHeightModel(self, dsm, dtm):
        '''Calculate height from integer arrays'''
        # Make all values positive by adding 1000 meter
        dsm = np.add(dsm, 1000)
        dtm = np.add(dtm, 1000)
        height = dsm - dtm
        return (height)

     # Generate vegetation height data of AHN information (based on DSM and DTM)
    def calculateVegetationHeight(self, dsm_path, dtm_path, dest_path, name, treshold = 5, normalize = True):
        # dsm_path: directory of dsm tiles with ID 
        # dtm_path: directory of dtm tiles with ID
        # dest_path: write images to directory
        # name: name of the file. Format: ID_'name'.tif   
        # treshold: heights higher then the treshold are set to threshold value (for proper normalization)
        # If treshold is None: no data manipulation

        # List all tiles of both dtm and dsm
        dsm_files = os.listdir(dsm_path)
        dtm_files = os.listdir(dtm_path)
        for i in range(len(dsm_files)):
            img_id = dsm_files[i].split("_")[0]
            for j in range(len(dtm_files)):
                img_id2 = dtm_files[j].split("_")[0]
                if img_id == img_id2:
                    # If ID's are identical: read the tiles of both dsm and dtm
                    with rio.open(dsm_path + "/" + dsm_files[i]) as src:
                        dsm = src.read()
                        meta = src.meta
                    with rio.open(dtm_path + "/" + dtm_files[j]) as src2:
                        dtm = src2.read()
                    # Replace 0 values by -9999. 0 values are holes in the data
                    dsm[dsm==0] = -9999
                    dtm[dtm==0] = -9999     
                    # Calculate the height of the vegetation (holes will be returned as negative)       
                    height = self.createHeightModel(dsm, dtm)
                    # Set negative values (holes that were set to -9999) to 5m
                    # Set very heigh objects to 5 to ensure a fixed scale of 0 - 5]
                    if treshold is not None:
                        height[height>treshold] = treshold
                        height[height<0] = treshold
                    else:
                        height[height<0] = np.nan

                    height.resize(self.image_size)
                    # Create filename
                    fn = f"{img_id}_{name}"

                    if normalize == True:
                        height = self.NormalizeData(height, 0, treshold)
                        print(np.max(height))
                        print(np.min(height))

                    # Write height tile to file
                    with rio.open(dest_path + "/" + fn, 'w', **meta) as dst:
                        dst.write_band(1, height)

    # Calculate slope from digital surface model
    def calculate_slope(self, file_path, dest_folder, file_name, normalize = True, max_value = 90, min_value = 0):
        gdal_dem = gdal.Open(file_path)
        gdal.DEMProcessing(f"{dest_folder}/{file_name}", gdal_dem, 'slope')
        if normalize == True:
            with rio.open(f"{dest_folder}/{file_name}", "r") as dataset:
                slope=dataset.read(1)
                meta = dataset.meta
                slope_normalized = self.NormalizeData(slope, min_value, max_value)
                slope_normalized[slope_normalized < 0] = 0.5
                slope_normalized[slope_normalized > 1] = 0.5
                dataset.close()                
            with rio.open(f"{dest_folder}/{file_name}", "w", **meta) as dst:                
                dst.write_band(1, slope_normalized)
                dst.close()


    # Calculate aspect from digital surface model
    def calculate_aspect(self, file_path, dest_folder, file_name, normalize = True, max_value = 360, min_value = 0):
        gdal_dem = gdal.Open(file_path)
        gdal.DEMProcessing(f"{dest_folder}/{file_name}", gdal_dem, 'aspect')

    # NORMALIZE TRAINING DATA DATA MIN-MAX
    def NormalizeData(self, array, min_value, max_value):
        array = array.astype('float32')
        # Rescale normalized images to 0 - 1  
        normalized_array = (array-min_value)/(max_value-min_value)
        return (normalized_array)

        