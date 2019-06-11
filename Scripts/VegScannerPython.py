### SCRIPT VEGETATION SCANNER ###
import os
work_directory = 'C:/Users/wba/Internship'
os.chdir(work_directory+'/MachineLearning/Scripts')
# from DataPreprocessing import *
# from DataCreation import *
# import CreateResults as cr
from DemProcessing import *
from random import sample
from owslib.wms import WebMapService
from rasterio.plot import reshape_as_raster, reshape_as_image
from matplotlib import pyplot
os.chdir(work_directory)

### CREATE 4 DIMENSIONAL IMAGE FROM CIR, RGB, HEIGHT: BLUE,GREEN, NDVI, HEIGHT ###
def createBGNdviHeightImage(rgb_path, cir_path, height_path, dest_folder, name = 'manueel4Channels.tif'):
    # path_rgb_data = folder with rgb images
    # path_cir_data = folder with cir images (should have the same id's and extents)
    # dest folder = destination folder of 4 channel (Green, blue, NDVI, Height) rasters
    # general name of the raster images (prefixed by known id)

    # Import normalization function
    work_directory = 'C:/Users/wba/Internship'
    os.chdir(work_directory+'/MachineLearning/Scripts')
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
                                    red = red.astype('float32')
                                    nir = nir.astype('float32')
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


### CREATE 3 DIMENSIONAL IMAGE FROM HEIGHT, SLOPE AND NDVI ###
def createHSNdviImage(rgb_path, cir_path, height_path, slope_path, dest_folder, name = 'manueel4Channels.tif'):
    # path_rgb_data = folder with rgb images
    # path_cir_data = folder with cir images (should have the same id's and extents)
    # dest folder = destination folder of 4 channel (Green, blue, NDVI, Height) rasters
    # general name of the raster images (prefixed by known id)

    # Import normalization function
    work_directory = 'C:/Users/wba/Internship'
    os.chdir(work_directory+'/MachineLearning/Scripts')    
    os.chdir(work_directory)
    #dmp = DEM_Processing(image_size = (256,256), cell_size = 0.25, epsg = 28992)

    rgb_images = os.listdir(rgb_path)
    cir_images = os.listdir(cir_path)
    height_images = os.listdir(height_path)
    slope_images = os.listdir(slope_path)
    for i in range(len(rgb_images)):
        print(i)
        if rgb_images[i].endswith('.tif'):
            img = rgb_images[i]
            img_id = f"{img.split('_')[0]}_{img.split('_')[1]}"
            with rio.open(rgb_path + "/" + rgb_images[i]) as src:
                red = src.read(1)
        for j in range(len(cir_images)):
            if cir_images[j].endswith('.tif'):
                img2 = cir_images[j]
                img_id2 = f"{img2.split('_')[0]}_{img2.split('_')[1]}"
                if img_id == img_id2:                    
                    with rio.open(cir_path + "/" + cir_images[j]) as src2:
                        nir = src2.read(1)
                    break
        for k in range(len(height_images)):
            if height_images[k].endswith('.tif'):
                img3 = height_images[k]
                img_id3 = img3.split("_")[0] 
                if img_id3 == img_id.split('_')[0]:
                    with rio.open(height_path + "/" + height_images[k]) as src3:
                        height = src3.read(1)
                    break             
        for l in range(len(slope_images)):
            if slope_images[l].endswith('.tif'):
                img4 = slope_images[l]
                img_id4 = f"{img4.split('_')[0]}"  
                if img_id4 == img_id.split("_")[0]:                                                            
                    with rio.open(slope_path + "/" + slope_images[l]) as src4:
                        slope = src4.read(1) 
                    break   
                                  
        red = red.astype('float32')
        nir = nir.astype('float32')
        check = np.logical_and ( red > 0, nir > 0 )
        ndvi = np.where(check,  (nir - red ) / ( nir + red ), 0) 
        ndvi = ndvi.astype('float32')
        #ndvi_normalized = dmp.NormalizeData(ndvi, -1, 1)
        bands = [ndvi, height, slope]
        # Update meta to reflect the number of layers
        meta = src.meta
        meta.update({'count' : 3, 'dtype':'float32'})
        
        # Read each layer and write it to stack
        with rio.open(dest_folder + "/" + img_id + "_" + name, 'w', **meta) as dst:
            for id, layer in enumerate(bands, start=1):
                dst.write_band(id, layer)

def createGrassMask(img_folder, mask_folder):
    mask_folder = work_directory +  "/Data/5_TrainingData/Gras/Images/From_model/2016_2017_RgbCirAhn/mask"
    images = os.listdir(img_folder)
    for img in images:
            name = f"{img.split('.')[0]}_mask.tif"
            tile_path = f"{img_folder}/{img}"
            with rio.open(tile_path, 'r') as src:
                    array_height = src.read(2)
                    array_height[array_height > 0.35] = 9999
                    array_height[array_height <= 0.35] = 1  
                    array_height[array_height > 1] = 0               
                    meta_copy = src.meta
                    meta_copy.update(count=1)
            array_ndvi = rio.open(tile_path, 'r').read(1)
            array_ndvi[array_ndvi< 0.05] = 0
            array_ndvi[array_ndvi > 0] = 1
            array_slope = rio.open(tile_path, 'r').read(3)
            array_slope[array_slope<=45] = 1
            array_slope[array_slope > 45 ] = 0
            result_array = array_height * array_slope * array_ndvi
            with rio.open(f"{mask_folder}/{name}", 'w', **meta_copy) as dst:
                    dst.write_band(1, result_array)