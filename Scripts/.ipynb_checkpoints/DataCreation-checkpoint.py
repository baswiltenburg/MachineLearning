import os, random
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from owslib.wms import WebMapService   
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import MultiPolygon, Point
import ogr, osr
from osgeo import gdal 
import shutil
import fiona
import rasterio as rio
from rasterio import crs
from rasterio import mask
import pycrs
from affine import Affine
from rasterio.plot import show
import tarfile
import json

class N2000_Data(): 
    
    #instance variables
    def __init__(self, wms, image_size, cell_size, epsg):
        self.wms = wms
        self.image_size = image_size
        self.cell_size = cell_size
        self.epsg = epsg
    
    # Create a list of bounding boxes of habitat type features 
    def getBoundingBoxes(self, shapeLocation):       
        shape = gpd.read_file(shapeLocation)
        bounding_box_list = []
        areas_list = []
        for i in range(len(shape)):
            bb_feature = list(shape['geometry'][i].bounds)            
            # Add the new boundingbox to the result list of bounding boxes
            bounding_box_list.append(bb_feature)

        bounding_box_areas = []
        # Calculate the area of each generated bounding box (number of image patches per bounding box are based on the area of the bounding box)
        for j in bounding_box_list:
            x_meter = j[2] - j[0]
            y_meter = j[3] - j[1]
            area = x_meter * y_meter
            bounding_box_areas.append(area)         
        return(bounding_box_list, bounding_box_areas) 
    
    
    
    
    
    def createImageBoundingBoxes(self, bounding_box_list, bounding_box_areas):
        bb_image_patches = []
        # Calculate the number of samples (training images) per habitat bounding box
        for i in range(len(bounding_box_list)):
            bb = bounding_box_list[i]
            bb_area = bounding_box_areas[i]
            image_area = (self.image_size[0]*self.cell_size) * (self.image_size[1]*self.cell_size)
            number_of_samples = int(np.ceil(bb_area / image_area))    
            # Get random x and y points within the bounding box and create image patch bounding box (this x and y point represent centre of image patch)
            for j in range(number_of_samples):
                x = random.uniform(bb[0], bb[2])
                y = random.uniform(bb[1], bb[3])
                # Calculate new bounding box based on a random center point
                xmin = x - ((self.image_size[0]/2)* self.cell_size)
                xmax = x + ((self.image_size[0]/2)* self.cell_size)
                ymin = y - ((self.image_size[1]/2)* self.cell_size)
                ymax = y + ((self.image_size[1]/2)* self.cell_size)
                bb_image_patch = [xmin, ymin, xmax, ymax]
                bb_image_patches.append(bb_image_patch) 
        # Return a list of lists with bounding box coordinates of training images to be downloaded!
        return(bb_image_patches)
    
    def createImageBoundingBoxes2(self, shapeLocation):
        from shapely.geometry import Polygon, Point, LineString
        shape = gpd.read_file(shapeLocation)
        polygons = []
        areas = []
        bb_image_patches = []
        for gdf in shape.iterrows():
            poly = gdf[1]['geometry']
            polygons.append(poly)
            area = gdf[1]['SHAPE_Area']
            areas.append(area)

        image_area = (self.image_size[0]*self.cell_size) * (self.image_size[1]*self.cell_size)
        result_coords = []
        for i in range(len(polygons)):
            poly = polygons[i]
            min_x, min_y, max_x, max_y = poly.bounds
            area = areas[i]
            number_of_samples = int(np.ceil(area / image_area))  
            min_x, min_y, max_x, max_y = poly.bounds
            count = 1
            while count <= number_of_samples:
                x = random.uniform(min_x, max_x)
                x_line = LineString([(x, min_y), (x, max_y)])      
                try:
                    x_line_intercept_min, x_line_intercept_max = x_line.intersection(poly).xy[1].tolist()
                    y = random.uniform(x_line_intercept_min, x_line_intercept_max)
                    c = (x,y)
                    result_coords.append(c)
                    xmin = x - ((self.image_size[0]/2)* self.cell_size)
                    xmax = x + ((self.image_size[0]/2)* self.cell_size)
                    ymin = y - ((self.image_size[1]/2)* self.cell_size)
                    ymax = y + ((self.image_size[1]/2)* self.cell_size)
                    bb_image_patch = [xmin, ymin, xmax, ymax]
                    bb_image_patches.append(bb_image_patch) 
                    count += 1
                except:
                    'Error: continue'
                    continue
        return (bb_image_patches)
    
    
    def downloadTrainingImages(self, bb_image_patches, store_path, name = "image", ir = False, years = ['2016','2017','2018']):
        crs = rio.crs.CRS({"init": ("epsg:"+str(self.epsg))}) 
        proj = pycrs.parse.from_epsg_code(self.epsg).to_proj4()
        
        # Loop trough the bounding box coordinates of training images need to be downloaded
        for i in range(len(bb_image_patches)):
            print(str(i) + " out of: " + str(len(bb_image_patches)))
            if ir == False:
                if '2018' in years:
                    img_2018 = self.wms.getmap(layers=['2018_ortho25'], styles=[], srs='EPSG:28992', crs='EPSG:28992', bbox=bb_image_patches[i],  size=self.image_size, format='image/tiff', transparent=True) # stream = True verwijderd  
                if '2017' in years:
                    img_2017 = self.wms.getmap(layers=['2017_ortho25'], styles=[], srs='EPSG:28992', crs='EPSG:28992', bbox=bb_image_patches[i],  size=self.image_size, format='image/tiff', transparent=True)  
                if '2016' in years:
                    img_2016 = self.wms.getmap(layers=['2016_ortho25'], styles=[], srs='EPSG:28992', crs='EPSG:28992', bbox=bb_image_patches[i],  size=self.image_size, format='image/tiff', transparent=True)  
            else:
                if '2018' in years:
                    img_2018 = self.wms.getmap(layers=['2018_ortho25IR'], styles=[], srs='EPSG:28992', crs='EPSG:28992', bbox=bb_image_patches[i],  size=self.image_size, format='image/tiff', transparent=True) # stream = True verwijderd  
                if '2017' in years:
                    img_2017 = self.wms.getmap(layers=['2017_ortho25IR'], styles=[], srs='EPSG:28992', crs='EPSG:28992', bbox=bb_image_patches[i],  size=self.image_size, format='image/tiff', transparent=True)  
                if '2016' in years: 
                    img_2016 = self.wms.getmap(layers=['2016_ortho25IR'], styles=[], srs='EPSG:28992', crs='EPSG:28992', bbox=bb_image_patches[i],  size=self.image_size, format='image/tiff', transparent=True)  
                

            # Define filenames
            filename_2018 = store_path + "/" + str(i) + "_2018_" + name + ".tif"  
            filename_2017 = store_path + "/" + str(i) + "_2017_" + name + ".tif"      
            filename_2016 = store_path + "/" + str(i) + "_2016_" + name + ".tif"      

            # Write images disk (as tiff files with spatial information)
            files = []
            if '2018' in years:
                out = open(filename_2018, 'wb')
                out.write(img_2018.read())
                out.close()  
                files.append(filename_2018)
            if '2017' in years:
                out = open(filename_2017, 'wb')
                out.write(img_2017.read())
                out.close()
                files.append(filename_2017)
            if '2016' in years: 
                out = open(filename_2016, 'wb')
                out.write(img_2016.read())
                out.close() 
                files.append(filename_2016)
                
            # List written files, update projetion and move tile to spatial position
            for file in files:
                # SET PROJECTION AND MOVE TILE TO POSITION #
                dataset = gdal.Open(file,1)
                
                # Get raster projection
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(self.epsg)
                dest_wkt = srs.ExportToWkt()
                
                # Set projection
                dataset.SetProjection(dest_wkt)
                
                gt =  dataset.GetGeoTransform()
                gtl = list(gt)
                gtl[0] = bb_image_patches[i][0]
                gtl[1] = self.cell_size
                gtl[3] = bb_image_patches[i][3]
                gtl[5] = (-1 * self.cell_size)
                dataset.SetGeoTransform(tuple(gtl))
                dataset = None                
            
# THIS CODE DOES THE SAME AS THE CODE ABOVE BUT USES RASTERIO INSTEAD OF OGR/GDAL            
#             for file in files:          
#                 transform = Affine(self.cell_size, 0.0, bb_image_patches[i][0], 0.0, -self.cell_size, bb_image_patches[i][3])
#                 with rio.open(file, mode = 'r+') as src:
#                     src.transform = transform
#                     src.crs = crs
#                     src.close()
                    
    def createRasterMasks(self, store_path, store_path_mask, shapeLocation):
        shape = gpd.read_file(shapeLocation)
        shape.loc[:,'Dissolve'] = 1
        files = [f for f in listdir(store_path) if isfile(join(store_path, f))]
        crs = rio.crs.CRS({"init": ("epsg:"+str(self.epsg))}) 
        proj = pycrs.parse.from_epsg_code(self.epsg).to_proj4()
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
                with rio.open(path_mask_file, mode = 'r') as src_mask:
                    # Create mask image
                    try:
                        out_img, out_transform = mask.mask(src_mask, shapes=coords, crop=True, invert = False)
                        out_meta = src_mask.meta.copy()
                        crs_metadata = getProj4(self.epsg)
                        out_meta.update({"driver": "GTiff", "height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform, "crs": crs_metadata})
                        out_img[out_img > 0] = 255
                        src_mask.close()
                        with rio.open(path_mask_file, "w", **out_meta) as dest:
                            dest.write(out_img)
                            dest.close()
                    except:
                        print ("No overlap")
                        os.unlink(file_path)
                        os.unlink(path_mask_file)
                        print(file_path)
                        print("Trainig image removed from dataset")
                        continue
                    
    def convertMaskToBinaryMask(self, src_folder, dst_folder):
        ############ CONVERT 3 DIMENSIONAL MASK INTO 2-DIMENSIONAL BINARY MASK ##############
        # src_folder: Folder with tif-files of 3-dimensional mask
        # dst_folder: Folder of destination
        
        #epsg = 28992
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
                    crs_metadata = getProj4(self.epsg)                
                    out_meta.update({"driver": "GTiff", "height": out_img.shape[0], "width": out_img.shape[1], "crs": crs_metadata, "count" : 1})
                    src_check.close()
                    # Write binary mask to file (only the first band)
                    with rio.open(pathname_dest, 'w', **out_meta) as dst:
                        dst.write_band(1, out_img)
                        dst.close
                    
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
                
    def saveImageDataToJson(self, image_directory, bounding_boxes_images, file_name, image_size = (512,512), pixel_size = 0.25):
        ############ SAVE BOUNDING BOXES AND NAMES IN JSON ##############
        import json
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
            dictionary_data[image_id] = {"bounding_box": bounding_boxes_images[i], "images_size" : image_size, "pixel_size" : pixel_size, "epsg": "28992"}

        # save json
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
                                    
                                    
    # Download entire area
    #area = [230378.510, 479586.273, 232983.850,  482326.797]
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
        while xmin <= xmax_original and ymin <= ymax_original:   
            bb = [xmin, ymin, xmin+(self.cell_size*self.image_size), ymin+(self.cell_size*self.image_size)]
            bb_list.append(bb)
            xmin = xmin + (self.cell_size*self.image_size)
            xmax = xmax + (self.cell_size*self.image_size) 
            if xmin >= xmax_original:
                xmin = xmin_original
                xmax = xmax_original
                ymin = ymin + (self.cell_size*self.image_size) 
                ymax = ymax + (self.cell_size*self.image_size) 