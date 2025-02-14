import os, random
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from owslib.wms import WebMapService   
import geopandas as gpd
import matplotlib
import shapely
from shapely.geometry import MultiPolygon, Point, Polygon, Point, LineString
import ogr, osr
from osgeo import gdal 
import shutil
import fiona
import rasterio as rio
from rasterio import crs
from rasterio import mask
import pycrs
from affine import Affine
import tarfile
import json
import requests
from PIL import Image
from io import BytesIO

class N2000_Data():     
    #instance variables
    def __init__(self, image_size, cell_size, epsg):
        self.image_size = image_size
        self.cell_size = cell_size
        self.epsg = epsg
     
    # Create image bounding boxes based on a shapefile of ground-truth locations
    # This funtion creates random intersection points with ground truth-polygons. The intersection point represent the center of 
    # the image bounding box. The list of created bounding boxes can be used to download training images.
    def createImageBoundingBoxes(self, shapeLocation):
        # Read shapefile
        shape = gpd.read_file(shapeLocation)
        polygons = [] # List of polygons in shapefile
        areas = [] # List of areas (the area of the polygons defines the number of generated bounding boxes)
        bb_image_patches = [] # List of final bounding boxes
        # Iterate over features (polygons) in shapefile 
        for gdf in shape.iterrows():
            poly = gdf[1]['geometry']
            polygons.append(poly)
            area = gdf[1]['SHAPE_Area']
            areas.append(area)

        # Calculate the spatial extend of the image based on image size and resolution
        image_area = (self.image_size[0]*self.cell_size) * (self.image_size[1]*self.cell_size)
        result_coords = []
        # Loop trough the polygons list and create random intersection points
        for i in range(len(polygons)):
            poly = polygons[i]
            min_x, min_y, max_x, max_y = poly.bounds
            area = areas[i]
            # Calculate how many intersection points (number of samples) need to created based on the area of the polygon
            number_of_samples = int(np.ceil(area / image_area))  
            min_x, min_y, max_x, max_y = poly.bounds
            count = 1
            while count <= number_of_samples:
                x = random.uniform(min_x, max_x)
                x_line = LineString([(x, min_y), (x, max_y)])      
                try:
                    # Create intersecting line with polygon
                    x_line_intercept_min, x_line_intercept_max = x_line.intersection(poly).xy[1].tolist()
                    # Pick a random point on this intersection line and covert it to coordinates (x,y)
                    y = random.uniform(x_line_intercept_min, x_line_intercept_max)
                    c = (x,y)
                    result_coords.append(c)
                    # Convert intersection point (center of bounding box) to image bounding boxes.
                    xmin = x - ((self.image_size[0]/2)* self.cell_size)
                    xmax = x + ((self.image_size[0]/2)* self.cell_size)
                    ymin = y - ((self.image_size[1]/2)* self.cell_size)
                    ymax = y + ((self.image_size[1]/2)* self.cell_size)
                    bb_image_patch = [xmin, ymin, xmax, ymax]
                    # Add bounding box to result list of bounding boxes
                    bb_image_patches.append(bb_image_patch)  
                    count += 1
                except:
                    'Error: continue'
                    continue
        return (bb_image_patches)
    
    # Function to download training images based om image bounding boxes (above function) and (wms)server 
    # layer is the layer-name of the wms-server, so both ortho as well as CIR can be downloaded
    def downloadTrainingImages(self, bounding_box, server, layer, store_path, name = "image"):
        # Set coordinate projection systems
        crs = rio.crs.CRS({"init": ("epsg:"+str(self.epsg))}) 
        proj = pycrs.parse.from_epsg_code(self.epsg).to_proj4()
        # Get data from wms server
        wms = server
        data = wms.getmap(layers=[layer], styles=[], srs=('EPSG:'+str(self.epsg)), crs=('EPSG:'+str(self.epsg)), bbox=bounding_box,  size=self.image_size, format='image/tiff', transparent=True, stream = True) 

        # Define filenames and write image to disk
        filename = f"{store_path}/{name}" 
        out = open(filename, 'wb')
        out.write(data.read())
        out.close()  

        # Geo-reference the image based on projection system and bounding box coordinates
        transform = Affine(self.cell_size, 0.0, bounding_box[0], 0.0, -self.cell_size, bounding_box[3])
        with rio.open(filename, mode = 'r+') as src:
            src.transform = transform
            src.crs = crs
            src.close()              

    # Download ahn3 images, based on servr and bounding box.
    def downloadAhn3Images(self, server, layer, bounding_box, dest_folder, name = 'unknown_ahn3'):
        # Create service ULR
        bb_param = f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}"
        serviceUrl = f"{server}?service=wcs&request=GetCoverage&format=geotiff_float32&BoundingBox={bb_param},urn:ogc:def:crs:EPSG::{str(self.epsg)}&width={str(self.image_size[0])}&height={str(self.image_size[0])}&version=1.0.0&coverage={layer}"
        # Download AHN images
        with requests.Session() as session:
            try:
                response = session.get(serviceUrl, headers={'Content-type': "image/tif"})
                if response.status_code == 200: # request is good
                    result = Image.open(BytesIO(response.content))
                    if result is not None:
                        if result.mode == 'P':
                            result = result.convert('RGB')
            except Exception as ex:
                    raise ex
        # Convert result to numpy array, set projection  and georeference
        result = np.array(result)
        crs = rio.crs.CRS({"init": ("epsg:"+str(self.epsg))}) 
        transform = Affine(self.cell_size, 0.0, bounding_box[0], 0.0, -self.cell_size, bounding_box[3])
        filename = dest_folder + "/" + name  
        meta = {'driver': 'GTiff', 'dtype': 'float32', 'nodata': None, 'width': self.image_size[0], 'height': self.image_size[1], 'count': 1, 'crs': crs, 'transform': transform}

        # Write result if tile consists of data. If tile has no data: location is not covered by AHN 3
        all_zeros = not np.any(result)
        if all_zeros == False:
            with rio.open(filename, mode = 'w', **meta) as src:
                src.write_band(1, result.astype(rio.float32))
                src.close()
        else:
            print('Location not covered by AHN3.. No results written.')

    # Create raster mask based on image and polygon                
    def createRasterMasks(self, store_path, store_path_mask, shapeLocation):
        shape = gpd.read_file(shapeLocation)
        shape.loc[:,'Dissolve'] = 1
        files = [f for f in listdir(store_path) if isfile(join(store_path, f))]
        crs = rio.crs.CRS({"init": ("epsg:"+str(self.epsg))}) 
        count = 0
        for file in files:
            if file.endswith(".tif"):
                count += 1
                print(str(count)+" out of: "+str(len(files)))
                # Define name and path of rastermask image
                name_mask_file = file.split('.')[0]+"_mask."+file.split('.')[1]
                path_mask_file = store_path_mask +"/"+name_mask_file
                # file path of the original training image (RGB)
                file_path = store_path+"/"+file
                # Copy the original image and set all values to [0,0,0]    
                image_copy = shutil.copy(file_path,path_mask_file)

                # Create single polygon as mask layer 
                clip_polygon = shape.dissolve(by = 'Dissolve') # dissolve based on product field -> change this in order to dissolve in all cases
                clip_polygon.crs = crs
                # Function to convert mask polygon to coordinats for mask operation with rasterio
                def getFeatures(gdf):
                    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
                    import json
                    return [json.loads(gdf.to_json())['features'][0]['geometry']]
                # Coordinates of mask layer
                coords = getFeatures(clip_polygon) # clip_polygon was polygons ?

                def getProj4(epsg):
                    try:
                        crs_metadata = pycrs.parse.from_epsg_code(epsg).to_proj4()
                        return crs_metadata
                    except:
                        print ("Failed: trying again")
                        return getProj4(epsg)                   

                # Open training mask images
                with rio.open(path_mask_file, mode = 'r+') as src_mask:
                    # Create mask image
                    try:
                        out_img, out_transform = mask.mask(src_mask, shapes=coords, crop=True, invert = False)
                        out_meta = src_mask.meta.copy()
                        #crs_metadata = getProj4(self.epsg)
                        crs_metadata = rio.crs.CRS({"init": ("epsg:"+str(self.epsg))}) 
                        out_meta.update({"driver": "GTiff", "height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform, "crs": crs_metadata})
                        out_img[out_img > 0] = 255
                        src_mask.close()
                        with rio.open(path_mask_file, "w", **out_meta) as dest:
                            dest.write(out_img)
                            dest.close()
                    except:
                        print ("No overlap, everything set to 0")
                        data = src_mask.read()
                        out_meta = src_mask.meta.copy()
                        crs_metadata = rio.crs.CRS({"init": ("epsg:"+str(self.epsg))}) 
                        #crs_metadata = getProj4(self.epsg)
                        out_meta.update({"driver": "GTiff", "height": self.image_size[0], "width": self.image_size[1], "crs": crs_metadata})
                        data[data>0]=0
                        src_mask.close()
                        with rio.open(path_mask_file, "w", **out_meta) as dest:
                            dest.write(data)
                            dest.close()
                        #os.unlink(file_path)
                        #os.unlink(path_mask_file)
                        print(file_path)
                        #print("Trainig image removed from dataset")
                        continue

    # Convert raster mask (function above) to binary raster mask (only 1 channel instead of 3)            
    def convertMaskToBinaryMask(self, src_folder, dst_folder):
        # src_folder: Folder with tif-files of 3-dimensional mask
        # dst_folder: Folder of destination

        def getProj4(epsg):            
            try:
                crs_metadata = pycrs.parse.from_epsg_code(epsg).to_proj4()
                return (crs_metadata)
            except:
                print ("Failed: trying again")
                return (getProj4(epsg))

        # Convert mask of three channels into binary raster mask
        count = 0
        for filename in os.listdir(src_folder): 
            count += 1
            if filename.endswith(".tif"):
                pathname = src_folder + "/" + filename        
                pathname_dest = dst_folder + "/" + filename
                with rio.open(pathname, mode = 'r') as src_check:
                    # Read the first band only (values in first, second and third band are the same)
                    out_img = src_check.read(1)
                    out_meta = src_check.meta.copy()              
                    #crs_metadata = getProj4(self.epsg)   
                    crs_metadata = rio.crs.CRS({"init": ("epsg:"+str(self.epsg))})              
                    out_meta.update({"driver": "GTiff", "height": out_img.shape[0], "width": out_img.shape[1], "crs": crs_metadata, "count" : 1})
                    src_check.close()
                    # Write binary mask to file (only the first band)
                    with rio.open(pathname_dest, 'w', **out_meta) as dst:
                        dst.write_band(1, out_img)
                        dst.close

    # Create checking images by creating polygon delaniations-mask           
    def createCheckingImages(self, store_path, store_path_check, shapeLocation):
        shape = gpd.read_file(shapeLocation)
        crs = rio.crs.CRS({"init": ("epsg:"+str(self.epsg))}) 
        proj = pycrs.parse.from_epsg_code(self.epsg).to_proj4()
        
        # Create new images for manual checking (polygon boundaries as mask)
        files = [f for f in listdir(store_path) if isfile(join(store_path, f))]

        # Create geodataframe of polygons which respresent the edges of habitat contour lines 
        # --> convert habitat delaniations (polygons) to linestrings 
        # --> convert linestrings to polygons to use these polygons as image-mask
        out_gdf = gpd.GeoDataFrame(columns = ['dissolve'])
        for index, row in shape.iterrows():
            # Create linestring object of each habitat polygon
            boundary = row.geometry.boundary
            # Buffer the line to convert it to a polygon
            buffer_line = boundary.buffer(0.4)  
            # Add the polygons to the (empty) geodataframe
            # If the polygon is type MultiPolygon: Expolode the polgyon and add individual polygons to geodataframe
            if buffer_line.geom_type == 'MultiPolygon':
                recs = len(buffer_line)
                multdf = gpd.GeoDataFrame(columns = ['dissolve'])
                multdf = multdf.append([buffer_line]*recs, sort=True)
                for geom in range(recs):       
                    multdf.loc[geom,'geometry'] = buffer_line.geoms[geom]
                    multdf.loc[geom,'dissolve'] = 'yes'
                multdf = multdf[['geometry', 'dissolve']][:] 
                out_gdf.append(multdf)
            else:
                index = len(out_gdf)
                out_gdf.loc[index, 'geometry'] = buffer_line
                out_gdf.loc[index, 'dissolve'] = 'yes'
        # Dissolve the entire geodataframe with habitat countours (as polygons)
        # This is the mask layer of the 'check-images'
        boundaries_dissolve = out_gdf.dissolve(by = 'dissolve')
        # Function to convert mask polygon to coordinats for mask operation with rasterio
        def getFeatures(gdf):
            """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
            import json
            return [json.loads(gdf.to_json())['features'][0]['geometry']]  
        # Coordinates of mask layer
        coords = getFeatures(boundaries_dissolve)
        
        def getProj4(epsg):            
            try:
                crs_metadata = pycrs.parse.from_epsg_code(epsg).to_proj4()
                return (crs_metadata)
            except:
                print ("Failed: trying again")
                return (getProj4(epsg))

        count = 0
        for file in files:
            if file.endswith(".tif"):
                count += 1
                print(str(count)+" out of: "+str(len(files)))
                # Define name and path of rastermask image
                name_check_file = file.split('.')[0]+"_check."+file.split('.')[1]
                path_check_file = store_path_check  +"/"+name_check_file
                # file path of the original training image (RGB)
                file_path = store_path+"/"+file
                # Copy the original image and set all values to [0,0,0]    
                image_copy = shutil.copy(file_path,path_check_file)  

                # Open 'check' images
                with rio.open(path_check_file, mode = 'r') as src_check:        
                    # Create mask image
                    out_img, out_transform = mask.mask(src_check, shapes=coords, crop=False, invert = True)
                    out_meta = src_check.meta.copy()              
                    crs_metadata = getProj4(self.epsg)                
                    out_meta.update({"driver": "GTiff", "height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform, "crs": crs_metadata})
                    #out_img[out_img > 0] = 255
                    src_check.close()
                    with rio.open(path_check_file, "w", **out_meta) as dest:
                        dest.write(out_img)
                        dest.close()

    # Create checking images just by masking                    
    def createCheckingImages2(self, store_path, store_path_check, shapeLocation):
        shape = gpd.read_file(shapeLocation)
        crs = rio.crs.CRS({"init": ("epsg:"+str(epsg))}) 
        proj = pycrs.parse.from_epsg_code(epsg).to_proj4()

        def getProj4(epsg):
            try:
                crs_metadata = pycrs.parse.from_epsg_code(epsg).to_proj4()
                return crs_metadata
            except:
                print ("Failed: trying again")
                return getProj4(epsg)

        # Create new images for manual checking (polygon boundaries as mask)
        files = [f for f in listdir(store_path) if isfile(join(store_path, f))]

        shape.loc[:]["dissolve"] = 'yes'
        shape_dissolve = shape.dissolve(by = 'dissolve')
        def getFeatures(gdf):
            """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
            import json
            return [json.loads(gdf.to_json())['features'][0]['geometry']]  
        # # Coordinates of mask layer
        coords = getFeatures(shape_dissolve)
        
        count = 0
        for file in files:
            if file.endswith(".tif"):
                count += 1
                print(str(count)+" out of: "+str(len(files)))
                # Define name and path of rastermask image
                name_check_file = file.split('.')[0]+"_check."+file.split('.')[1]
                path_check_file = store_path_check  +"/"+name_check_file
                # file path of the original training image (RGB)
                file_path = store_path+"/"+file
                # Copy the original image and set all values to [0,0,0]    
                image_copy = shutil.copy(file_path,path_check_file)  

                # Open 'check' images
                with rio.open(path_check_file, mode = 'r') as src_check:        
                    # Create mask image
                    out_img, out_transform = mask.mask(src_check, shapes=coords, crop=True, invert = False)
                    out_meta = src_check.meta.copy()              
                    crs_metadata = getProj4(epsg)                
                    out_meta.update({"driver": "GTiff", "height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform, "crs": crs_metadata})
                    #out_img[out_img > 0] = 255
                    src_check.close()
                    with rio.open(path_check_file, "w", **out_meta) as dest:
                        dest.write(out_img)
                        dest.close()
                    
    def createZipfile(self, folder_path, filename = "images.gzip"):
        # Create zip-files of images 
        tar_name = folder_path + "/" + filename
        files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

        with tarfile.open(tar_name, "w:gz") as tar_handle:   
            for file in files:
                file = folder_path + "/" + file
                tar_handle.add(file)

    # Save the metadata of downloaded training images in json-file (very important to georeference them later on)       
    def saveImageDataToJson(self, image_directory, bounding_boxes_images, file_name):
        # Image_directory is the folder of the downloaded images
        # Bounding_boxes_images is the list of bounding boxes
        # filename is desired name of json-file
        
        image_ids = []
        for filename in os.listdir(image_directory): 
            filename_split = filename.split("_")
            if filename.endswith(".tif"):
                image_id = int(filename_split[0])
                if image_id not in image_ids:
                    image_ids.append(image_id)

        image_ids.sort()
        dictionary_data = {}
        for i in range(len(bounding_boxes_images)):
            image_id = image_ids[i]
            dictionary_data[image_id] = {"bounding_box": bounding_boxes_images[i], "images_size" : self.image_size, "pixel_size" : self.cell_size, "epsg": str(self.epsg)}

        # save json to disk
        store_json = image_directory + "/" + file_name
        with open (store_json, 'w') as fp:
            json.dump(dictionary_data, fp)
            
    # Create 4-dimensional image from CIR and RGB image
    def create4dimensionalImage(self, path_rgb_data, path_cir_data, dest_folder, name = '4channel_data.tif'):
        # path_rgb_data = folder with rgb images
        # path_cir_data = folder with cir images (should have the same id's and extents)
        # dest folder = destination folder of 4 channel (B,G,R,NIR) rasters
        # general name of the raster images (prefixed by known id)

        for file in os.listdir(path_rgb_data):
            if file.endswith('.tif'):
                file_id = file.split("_")[0] + "_" + file.split("_")[1]
                file = path_rgb_data + "/" + file    
                for file2 in os.listdir(path_cir_data):
                    if file2.endswith('.tif'):
                        file2_id = file2.split("_")[0] + "_" + file2.split("_")[1]
                        file2 = path_cir_data + "/" + file2 
                        if file_id == file2_id and not file2.endswith('mask.tif'):  
                            with rio.open(file, mode = 'r') as rgb_img:
                                blue = rgb_img.read(1)
                                #image_blue = reshape_as_image(blue)                
                                green = rgb_img.read(2)
                                #image_green = reshape_as_image(green)                
                                red = rgb_img.read(3)
                                #image_red = reshape_as_image(red)

                            with rio.open(file2, mode = 'r') as cir_img:
                                nir = cir_img.read(1)
                                #image_nir = reshape_as_image(nir)

                            bands = [blue, green, red, nir]

                            # Update meta to reflect the number of layers
                            meta = rgb_img.meta
                            meta.update(count = 4)

                            # Read each layer and write it to stack
                            with rio.open(dest_folder + "/" + file_id + "_" + name, 'w', **meta) as dst:
                                for id, layer in enumerate(bands, start=1):
                                    dst.write_band(id, layer)

     # Create 5-or 6-dimensional image from CIR and RGB image
    def CreateMultiDimensionalImage(self, path_rgb_data, path_cir_data, path_height_data, dest_folder, path_slope_data = None, name = '6channel_data.tif'):
        # path_rgb_data = folder with rgb images
        # path_cir_data = folder with cir images (should have the same id's and extents)
        # path_height_data = folder with height images
        # path_slope data = folder with slope images
        # dest folder = destination folder of 4 channel (B,G,R,NIR) rasters
        # general name of the raster images (prefixed by known id)

        for rgb_img in os.listdir(path_rgb_data):
            height = None 
            red = None
            blue = None 
            green = None
            slope = None
            nir = None
            if rgb_img.endswith('.tif'):
                file_id = rgb_img.split("_")[0] + "_" + rgb_img.split("_")[1]
                id_without_year = rgb_img.split("_")[0]
                rgb_path = path_rgb_data + "/" + rgb_img 
                with rio.open(rgb_path, mode = 'r') as img_rgb:
                    blue = img_rgb.read(1)
                    blue = blue.astype('float32') 
                    #image_blue = reshape_as_image(blue)                
                    green = img_rgb.read(2)
                    green = green.astype('float32')
                    #image_green = reshape_as_image(green)                
                    red = img_rgb.read(3)
                    red = red.astype('float32')
                    #image_red = reshape_as_image(red)    

                # Get Near infrared band
                for cir_img in os.listdir(path_cir_data):
                    if cir_img.endswith('.tif'):
                        file2_id = cir_img.split("_")[0] + "_" + cir_img.split("_")[1]
                        cir_path = path_cir_data + "/" + cir_img 
                        if file_id == file2_id:                                              
                            with rio.open(cir_path, mode = 'r') as img_cir:
                                nir = img_cir.read(1)
                                nir = nir.astype('float32')
                # Get height band
                for h_img in os.listdir(path_height_data):
                    if h_img.endswith('.tif'):
                        file3_id = h_img.split("_")[0] 
                        
                        h_path = path_height_data + "/" + h_img 
                        if id_without_year == file3_id:                                        
                            with rio.open(h_path, mode = 'r') as img_h:
                                height = img_h.read(1)
                                height = height.astype('float32')
                if path_slope_data is not None:
                # Get slope band
                    for slope_img in os.listdir(path_slope_data):
                        if slope_img.endswith('.tif'):
                            file4_id = slope_img.split("_")[0]
                            slope_path = path_slope_data + "/" + slope_img
                            if id_without_year == file4_id:                                              
                                with rio.open(slope_path, mode = 'r') as img_slope:
                                    slope = img_slope.read(1)  
                                    slope = slope.astype('float32')    

                if height is not None and slope is not None and nir is not None and red is not None and green is not None and blue is not None:
                    bands = [blue, green, red, nir, height, slope]
                elif height is not None and nir is not None and red is not None and green is not None and blue is not None: 
                    bands = [blue, green, red, nir, height]
                else:
                    continue              
                
                # Update meta to reflect the number of layers
                meta = img_rgb.meta
                meta.update({"count":len(bands), "dtype":"float32"})

                # Read each layer and write it to stack
                with rio.open(dest_folder + "/" + file_id + "_" + name, 'w', **meta) as dst:
                    for id, layer in enumerate(bands, start=1):
                        dst.write_band(id, layer)
                        
                                    
    # Download entire area based on bounding box extent
    def getBoundingBoxesAreaExtent(self, area):
        # area is a bounnding box of a entire study area ([xmin, ymin, xmax, ymax])
        # Function returns a list of (inner-)bounding boxes of specified size
        bb_list = []
        xmin_original = area[0]
        ymin_original = area[1]
        xmax_original = area[2]
        ymax_original = area[3]
        xmin = area[0]
        ymin = area[1]
        xmax = area[2]
        ymax = area[3]
        bb_id = 0
        ids = []
        while xmin <= xmax_original and ymin <= ymax_original:   
            bb = [xmin, ymin, xmin+(self.cell_size*self.image_size[0]), ymin+(self.cell_size*self.image_size[0])]
            bb_list.append(bb)
            ids.insert(len(ids), bb_id)
            bb_id += 1
            xmin = xmin + (self.cell_size*self.image_size[0])
            xmax = xmax + (self.cell_size*self.image_size[0]) 
            if xmin >= xmax_original:
                xmin = xmin_original
                xmax = xmax_original
                ymin = ymin + (self.cell_size*self.image_size[0]) 
                ymax = ymax + (self.cell_size*self.image_size[0]) 
        return(bb_list, ids)