from pathlib import Path
import os
import h5py
import json
import rasterio as rio
from rasterio import crs
import json
import geopandas as gpd
import numpy as np
import rasterio.features
from affine import Affine
from shapely.geometry import shape, mapping, Polygon
    
### Read h5 file ###
def readH5File (h5_path):
    # Open h5 file 
    h5 = h5py.File(h5_path, 'r')
    # Get array of images
    if 'images' in h5.keys():
        images = h5['images'][:]
    else:
        images = []

    # Get array of corresponding masks
    if 'masks' in h5.keys():
        masks = h5['masks'][:]
    else:
        masks = []

    # Get list of filenames of corresponding images and masks!
    if 'filenames' in h5.keys():
        filenames = list(h5['filenames'][:])
        filenames = [n.decode('UTF-8') for n in filenames]
    else:
        filenames= []
    h5.close()

    # Return the data
    return(images, masks, filenames)

### write predictions / arrays to geotiff files ###
def writeGeoTiff(json_path, numpy_array, filename, geotiff_folder, epsg = 28992): 
    #json path:         path to json file with metadata of image
    #numpy array:       predicted images (as numpy array)
    #filename:          corresponding filename from h5 file
    #geotiff_folder:    destination folder
    import json
    # Open original json with image-information (bounding box coordinates etc.)
    with open(json_path) as json_file:
        data = json.load(json_file)
        for key in data.keys():
            bb = data[key]["bounding_box"]
            image_size = data[key]["images_size"]
            pixel_size = data[key]["pixel_size"]
            epsg = data[key]["epsg"]
            # Loop trough all the files in the filenames list and match it with corresponding image in json-file
            fn = filename
            image_id = fn.split("_")[0]
            if key == image_id:
                image = numpy_array
                image = image.astype('float32')
                # Reshape image to write probability in geotiff format (spatial!)
                image = image.reshape(image.shape[0], image.shape[1])

                # Transform image to position and define spatial reference
                transform = Affine(pixel_size, 0.0, bb[0], 0.0, -pixel_size, bb[3])
                crs = rio.crs.CRS({"init": ("epsg:"+str(epsg))}) 
                out_meta = ({"driver": "GTiff", "height": image.shape[0], "width": image.shape[0], "dtype" : str(image.dtype), "count":1, "transform": transform, "crs": crs})

                # Write prediction image to file
                with rio.open(geotiff_folder+"/" + fn + "_prediction.tif", mode = 'w', **out_meta) as dst:
                    dst.write(image, 1)
                    dst.close()           
        json_file.close()

def vectorizePrediction(geotiff_folder, cat = 'h5130', epsg = 28992, prob_treshold = 0.5):
    #geotiff_folder:    folder with geotiff images of predictions
    #cat:               class of object (i.e. habitat type class)

    # Create an empty dataframe (will be filled with polygons of positive class)
    out_gdf = gpd.GeoDataFrame(columns = ['img_id', 'file_name', 'class'], crs = {'init': 'epsg:'+str(epsg)})

    # List all georeferenced prediction files and loop trough this list
    files = os.listdir(geotiff_folder)
    index = 0
    for file in files:
        if file.endswith(".tif"):
            image_id = file.split("_")[0]
            name = file.split("_")[0] + "_" + file.split("_")[1] + "_" + file.split("_")[2]
            # Open the georeferenced probability rasters (output of deep learning model)
            with rio.open(geotiff_folder+"/" + file, mode = 'r') as src:
                src_img = src.read()
                transform = src.meta['transform']
                crs = src.meta['crs']
                # If probablitiy of a pixel is larger then 0,5: positive class. Else: negative class
                src_img[src_img >= prob_treshold] = 1

                # Vectorize probability raster
                for shp, val in rasterio.features.shapes(src_img, transform=transform):
                    # Add polygons (shape) to vector list when raster value = 1 (positive class)
                    if val == 1:                  
                        out_gdf.loc[index,'geometry'] = shape(shp)
                        out_gdf.loc[index,'img_id'] = image_id
                        out_gdf.loc[index,'file_name'] =  name
                        out_gdf.loc[index,'class'] =  cat
                        index += 1
                    else:
                        continue

    # Write geopandas dataframe to file
    out_gdf.to_file(driver = 'ESRI Shapefile', filename= geotiff_folder+"/"+cat+".shp")