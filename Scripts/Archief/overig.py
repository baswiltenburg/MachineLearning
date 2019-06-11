mask = work_directory+"/data/n2000_project/shape/Gras_bos_handmatig_clipLayer_dissolve.shp"
store_path = path_original_data
store_path_check = path_check_images
epsg = 28992
shapeLocation = mask
from rasterio import mask

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
                    

#####################################################################

                             
# Copy image and mask to new folder
for file in os.listdir(dest):
    if file.endswith('.tif'):
        filename = file.split("_")[0] + "_" + file.split("_")[1]
        for file2 in os.listdir(path_cir_data):
            if file2.endswith('.tif'):
                filename2 = file2.split("_")[0] + "_" + file2.split("_")[1]
                if filename == filename2 and file2.endswith("_mask.tif"):
                    file_path = dest + "/" + file
                    dest_path = path_training_data + "/" + file
                    image_copy = shutil.copy(file_path,dest_path)

                    file_path2 = cir_dir + "/" + file2
                    name = file.split("_")[-1].split(".")[0]
                    dest_path2 = path_training_data + "/" + filename+"_"+name+"_mask.tif"
                    image_copy = shutil.copy(file_path2, dest_path2)

                    
########################################################################

                             
# Function that performs all actions simultaniously
def getData(list_polygon_files, path_original_data, filename, sample_size = 1000, ir=False, years = ['2016', '2017', '2018'], 
            wms = wms, image_size = image_size, cell_size = cell_size, epsg = epsg):
    # list_polygons_files:     List of pathnames of polygon shapefiles of target data
    
    # Create N2000_Data object
    dc = N2000_Data(wms = wms, image_size = image_size, cell_size = cell_size, epsg = epsg)
    # Generate empty list of the bounding boxes of images
    bounding_boxes_images_list = []
    
    for shape_file in list_polygon_files:
        # Generate bounding boxes for training images based on target polygon locations
        bounding_boxes_images = dc.createImageBoundingBoxes2(shapeLocation = shape_file)
        # Get a sample based on sample size
        if len(bounding_boxes_images) > sample_size:
            bounding_boxes_images = sample(bounding_boxes_images, sample_size)
            
        # Add the bounding boxes to a list
        bounding_boxes_images_list.append(bounding_boxes_images)
    
    # Create one list of bounding boxes
    bounding_boxes_images_totaal = []
    for i in range(len(bounding_boxes_images_list)):
        for bb in bounding_boxes_images_list[i]:
            bounding_boxes_images_totaal.append(bb)
            
    # Download the data based on bounding boxes        
    dc.downloadTrainingImages(bounding_boxes_images_totaal, path_original_data, name = filename, ir = ir, years = years)
    dc.saveImageDataToJson(image_directory = path_original_data, bounding_boxes_images = bounding_boxes_images_totaal, file_name = filename+'.json', image_size = image_size)       
    
    

#########################################################################

    
path_data = work_directory + "/data/n2000_project/training_images/manueel_gras/training_corrected"
path_json = work_directory + "/data/n2000_project/training_images/manueel_gras/grasManueel256pxIR.json"
dest_folder = work_directory + "/data/n2000_project/training_images/destination"
# Download images based on bounding boxes of existing data
def getBoundingBoxesOfExistingImages(self, path_data, path_json, dest_folder):
    # List all files in existing training data 
    files = os.listdir(path_data)
    # fill list of IDS which had been selected for training
    id_list = []
    for file in files:
        if file.endswith('.tif'):
            img_id = file.split("_")[0]
            if img_id not in id_list:
                id_list.append(img_id)

    # Copy images and masks to new folder
    for file in files:
        if file.endswith('.tif'):
            file_path = path data + "/" + file
            dest_path = dest_folder + "/" + file
            image_copy = shutil.copy(file_path,dest_path)
            
    id_list = list(map(int, id_list))
    id_list.sort()
    # Read bounding boxes from  json
    bb_list = []
    with open(path_json) as json_file:  
        data = json.load(json_file)
        for img_id in id_list:
            img_id = str(img_id)
            bb = data[img_id]['bounding_box']
            bb_list.append(bb)
    return(bb_list)



#########################################################


# Rename images based on original CIR image
for file in os.listdir(path_rgb_data):
    if file.endswith('.tif'):
        file_path = path_rgb_data +"/" + file
        with rio.open(file_path, mode = 'r') as rgb_img:
            bb = [rgb_img.bounds[0], rgb_img.bounds[1], rgb_img.bounds[2], rgb_img.bounds[3]]
            with open(path_json) as json_file:  
                data = json.load(json_file)
                for img_id in data.keys():
                    if data[str(img_id)]['bounding_box'] == bb:
                        newName = file.split("_")
                        newName = str(img_id) + "_" + newName[1] + "_" + newName[2]
                        os.rename(path_rgb_data + "/" + file, path_rgb_data + "/" + newName)   