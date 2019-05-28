# Download RGB images based on bounding boxes of existing CIR data
# Folder of CIR images and JSON
path_cir_data = work_directory + "/data/n2000_project/training_images/manueel_gras/training_corrected"
path_json = work_directory + "/data/n2000_project/training_images/manueel_gras/grasManueel256pxIR.json"

# List all files in training data 
files = os.listdir(path_cir_data)
# fill list of IDS which had been selected for training
id_list = []
for file in files:
    if file.endswith('.tif'):
        img_id = file.split("_")[0]
        if img_id not in id_list:
            id_list.append(img_id)
            
            
# Copy images and masks of 2016 and 2017 to new folder
for file in files:
    if file.endswith('.tif'):
        year = file.split("_")[1]
        if year == '2016' or year == '2017':
            file_path = work_directory + "/data/n2000_project/training_images/manueel_gras/training_corrected/" + file
            dest_path = path_original_data + "/" + file
            image_copy = shutil.copy(file_path,dest_path)

            
id_list = list(map(int, id_list))
id_list.sort()
# Read bounding boxes from  json and download RGB images
bb_list = []
with open(path_json) as json_file:  
    data = json.load(json_file)
    for img_id in id_list:
        img_id = str(img_id)
        bb = data[img_id]['bounding_box']
        bb_list.append(bb)
print(len(bb_list))

path_rgb_data = work_directory + "/data/n2000_project/training_images/gras_manueel_4channels/training_rgb"
dc.downloadTrainingImages(bb_list, path_rgb_data, name = "bgtGrasManueel256pxRGB", ir = False, years = ['2016', '2017'])

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
                        
# Verwijder afbeeldingen waarvan geen CIR image
cir_dir = work_directory + "/data/n2000_project/training_images/gras_manueel_4channels/training_cir"
img_id_list = []
for f in os.listdir(cir_dir):
    img_id = f.split("_")[0] + "_" + f.split("_")[1]
    if img_id not in img_id_list:
        img_id_list.append(img_id)

count = 0
for j in os.listdir(path_rgb_data):
    img_id = j.split("_")[0] + "_" + j.split("_")[1]
    if img_id not in img_id_list:
        count += 1
        os.unlink(path_rgb_data + "/" + j)

cir_dir = work_directory + "/data/n2000_project/training_images/gras_manueel_4channels/training_cir"
dest = work_directory + "/data/n2000_project/training_images/gras_manueel_4channels/training"

# Create 4-dimensional image from CIR and RGB image
for file in os.listdir(path_rgb_data):
    if file.endswith('.tif'):
        file_id = file.split("_")[0] + "_" + file.split("_")[1]
        file = path_rgb_data + "/" + file    
        for file2 in os.listdir(cir_dir):
            if file2.endswith('.tif'):
                file2_id = file2.split("_")[0] + "_" + file2.split("_")[1]
                file2 = cir_dir + "/" + file2        
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
                    with rio.open(dest + "/" + file_id + '_CirRgb256px.tif', 'w', **meta) as dst:
                        for id, layer in enumerate(bands, start=1):
                            dst.write_band(id, layer)

# Copy image and mask to new folder
for file in os.listdir(dest):
    if file.endswith('.tif'):
        filename = file.split("_")[0] + "_" + file.split("_")[1]
        for file2 in os.listdir(cir_dir):
            filename2 = file2.split("_")[0] + "_" + file2.split("_")[1]
            if filename == filename2 and file2.endswith("_mask.tif"):
                file_path = dest + "/" + file
                dest_path = path_training_data + "/" + file
                image_copy = shutil.copy(file_path,dest_path)
                
                file_path2 = cir_dir + "/" + file2
                name = file.split("_")[-1].split(".")[0]
                dest_path2 = path_training_data + "/" + filename+"_"+name+"_mask.tif"
                image_copy = shutil.copy(file_path2, dest_path2)