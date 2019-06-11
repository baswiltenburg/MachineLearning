import os
import io
import shutil
import gdal, osr
import numpy as np
import h5py
import rasterio as rio
from PIL import Image
import json
from sklearn.model_selection import train_test_split
import random

class N2000_DataPreparation():          
    def __init__(self, image_size):
        # Variables        
        self.image_size = image_size

    # GET 'CHECK' IMAGES FROM BLOB STORAGE #
    # ALTERNATIVE https://github.com/agermanidis/pigeon
    def GetBlobData(self, path_training_data, config = '../config.txt', blobname = 'n2000-images-corrected'):
        block_blob = dimension.BlobConnect(config).block_blob_service()
        file_list = block_blob.list_blob_names(blobname)
        for item in file_list.items:
            block_blob.get_blob_to_path(blobname, item, (path_training_data+"/"+item))

    def ImagesToArray(self, img_folder, mask = True):
        ### READ IMAGES IN 3 DIMENSIONAL NUMPY ARRAY ###
        tiles = os.listdir(img_folder)
        array_list = []
        for tile in tiles:
            if mask == False:
                if tile.endswith('.tif') and not tile.endswith('_mask.tif'):
                    tile_path = f"{img_folder}/{tile}"
                    array = rio.open(tile_path, 'r').read() 
                    array = array.transpose((1, 2, 0))  
                    array_list.append(array)
            else:
                if tile.endswith('.tif'):
                    tile_path = f"{img_folder}/{tile}"
                    array = rio.open(tile_path, 'r').read() 
                    array = array.transpose((1, 2, 0))  
                    array_list.append(array)     
        # Resulting array 
        result_array = np.array(array_list) 
        return (result_array)
            
    # RENAME IMAGES: REMOVE THE WORD 'CHECK'  
    def RenameCheckImages(self, path_training_data):
        for filename in os.listdir(path_training_data): 
            filename_split = filename.split("_")
            if filename.endswith(".tif"):
                dst_name = filename_split[0]+"_"+filename_split[1]+"_"+filename_split[2]+".tif"
                os.rename((path_training_data+"/"+filename), (path_training_data+"/"+dst_name)) 
            else:
                continue
    
    # GENERATE FOLDER OF CORRECT TRAINING DATA (ORIGINAL IMAGE PLUS ITS MASK)
    # CHECK IMAGES (WITH MASKED POLYGON BOUNDARIES) ARE REPLACED BY ITS ORIGINAL IMAGE
    # MASK IMAGES ARE ADDED TO THE TRAININGSET FOLDER
    def PrepareTrainingData(self, path_training_data, path_mask_data, check_images = False, path_original_data = None):
        # Copy the mask image from the mask-folder to the training folder when image is in trainingfolder 
        # Replace check image (with masked countour lines, with original image)
        # If check images == True, training data are checked images (with boundary mask) and need to be replaced by the original data
        # Folder of original data need to be specified

        for filename in os.listdir(path_training_data): 
            filename_split = filename.split("_")            
            if filename.endswith(".tif"):
                # NAME SHOULD ALWAYS BE ID + YEAR
                #name = filename_split[0]+"_"+filename_split[1]+"_"+filename_split[2]
                name = filename_split[0]+"_"+filename_split[1]
                name = name.split(".")[0]  
                # If mask is true, copy the original mask image and put in the training folder
                #if filename_split[-1].startswith("mask"):
                for maskname in os.listdir(path_mask_data):
                    if maskname.endswith("mask_.tif"):
                        # Maskname should always be ID + YEAR
                        maskname_split = maskname.split("_")             
                        #maskname2 = maskname_split[0]+"_"+maskname_split[1]+"_" + maskname_split[2]
                        maskname2 = maskname_split[0]+"_"+maskname_split[1]
                        maskname2 = maskname2.split(".")[0]
                        # Copy mask image to folder with training images 
                        if maskname2 == name:
                            image_copy = shutil.copy((path_mask_data+"/"+maskname), (path_training_data+"/"+maskname)) 
                    else:
                        continue
                # Copy the original image and replace the 'check image'
                if check_images == True:
                    for original_name in os.listdir(path_original_data):
                        original_name_split = original_name.split("_")
                        if original_name.endswith(".tif"):
                            original_name2 = original_name_split[0]+"_"+original_name_split[1]
                            original_name2 = original_name2.split(".")[0]
                            # Copy original image to folder with training images 
                            if original_name2 == name:
                                image_copy = shutil.copy((path_original_data+"/"+original_name),(path_training_data+"/"+original_name))
                        else:
                            continue
            else:
                print("File is not a .tif file")
                continue
                
    # REMOVE IMAGES FROM TRAINING DATA OF WHICH ITS BINARY MASK IMAGE IS NOT OF CORRECT SHAPE
    # REMOVE IMAGES WITHOUT MASK 
    def RemoveInvalidData(self, path_training_data):
        wrong_items = []
        # Create list of mask files which do not have a correct shape
        for filename in os.listdir(path_training_data):
            filename_split = filename.split("_")
            if filename_split[-1].startswith("mask"):
                img = Image.open(path_training_data + "/" + filename)
                arr = np.array(img)
                if arr.shape != self.image_size:
                    image_name = filename_split[0] + "_" + filename_split[1] + "_" + filename_split[2]
                    wrong_items.append(image_name)

        # remove mask images and training images if mask was wrong
        for img_file in os.listdir(path_training_data):
            if img_file.endswith(".tif"):
                name_list = img_file.split("_")
                name = name_list[0] + "_" + name_list[1] + "_" + name_list[2]
                name = name.split(".")[0]  
                for wf in wrong_items:
                    if wf == name:
                        os.unlink(self.path_training_data + "/" + img_file)    

        # remove images which do not have a mask image
        tiles = os.listdir(path_training_data)
        for tile in tiles:
            if tile.endswith('.tif') and not tile.endswith('_mask.tif'):
                mask_name = f"{tile.split('.')[0]}_mask.tif"
                mask_path = f"{path_training_data}/{mask_name}"
                if not os.path.exists(mask_path):
                    # Delete training image if mask not exists
                    path_training_image = f"{path_training_data}/{tile}"
                    os.unlink(path_training_image)
                        
    # WRITE TRAINING AND MASK IMAGES TO H5 FORMAT AND ENSURE SIMILAR SHAPE FORMAT
    def SaveTrainingData(self, path_training_data, name_file = "Dataset_train.h5"):
        images = []
        masks = []
        filenames = []
        for filename in os.listdir(path_training_data):
            if filename.endswith(".tif"):
                filename_split = filename.split("_") 
                if filename_split[-1].startswith("mask"):
                    continue
                else:     
                    # Name/ID of the file
                    name = filename_split[0] + "_" + filename_split[1] + "_" + filename_split[2]
                    name = name.split(".")[0]
                    # Add file name to list of filenames
                    filenames.append(name)
                    
                    # Add image to the list of images
                    img = rio.open(path_training_data + "/" + filename)  
                    array = img.read()
                    array = array.transpose((1, 2, 0))  
                    images.append(array)

                    # Look for the corresponding mask-file with same name/id
                    for filename_mask in os.listdir(path_training_data):
                        filename_split_mask = filename_mask.split("_") 
                        if filename_split_mask[-1].startswith("mask"):
                            # Name/ID of the file
                            name_mask = filename_split_mask[0] + "_" + filename_split_mask[1] + "_" + filename_split_mask[2]
                            name_mask = name_mask.split(".")[0]
                            if name == name_mask:
                                # Add mask image to the list of mask images
                                img_mask = Image.open(path_training_data + "/" + filename_mask)
                                # Add a dimension to the mask-data, so the shape format becomomes: 512x512x1 instead of 512x5212
                                img_mask = np.expand_dims(img_mask, axis=-1) 
                                arr_mask = np.array(img_mask) 
                                masks.append(arr_mask)                                   

                        else:
                            continue # file is not a mask-file

        # Read lists of arrays as single array
        images = np.array(images)
        masks = np.array(masks)
        filenames = [n.encode("ascii", "ignore") for n in filenames]

        with h5py.File(path_training_data + "/" + name_file, 'w') as hdf:
            hdf.create_dataset('images', data=images, compression='gzip', compression_opts=9)
            hdf.create_dataset('masks', data=masks, compression='gzip', compression_opts=9)
            hdf.create_dataset('filenames', data=filenames, compression='gzip', compression_opts=9)

    # Merge multiple H5-files (with images, masks and filenames) into one set
    def MergeTrainingData(self, paths_list, dest_folder, name_file = 'merged_data.h5', write = True):
        # Paths_list:       list of paths linking to h5 files
        filenames_list = []
        count = 0
        for path_dataset in paths_list:      
            h5_file = h5py.File(path_dataset, 'r')   
            imgs_train = np.array(h5_file.get('images'))
            imgs_mask_train = np.array(h5_file.get('masks'))
            if count == 0:
                imgs_merge = imgs_train 
                masks_merge = imgs_mask_train
                count += 1
            else:
                imgs_merge = np.concatenate((imgs_merge, imgs_train))
                masks_merge = np.concatenate((masks_merge, imgs_mask_train))
                count += 1

            filenames = list(h5_file.get('filenames'))  
            #filenames = [n.decode('UTF-8') for n in filenames]# Load training data
            filenames_list.append(filenames)

        filenames_merge = []
        for i in range(len(filenames_list)):
            for filename in filenames_list[i]:
                filenames_merge.append(filename)
            
        if write == True:
            with h5py.File(dest_folder + "/" + name_file, 'w') as hdf:
                hdf.create_dataset('images', data=imgs_merge, compression='gzip', compression_opts=9)
                hdf.create_dataset('masks', data=masks_merge, compression='gzip', compression_opts=9)
                hdf.create_dataset('filenames', data=filenames_merge, compression='gzip', compression_opts=9)
      
        return(imgs_merge, masks_merge, filenames_merge)            
      
    # DEVIDE DATASETS IN VALIDATION, TEST EN TRAIN AND CREATE FILE WITH FILENAMES OF TESTSET
    # TEST SET ARE IMAGES WHICH ARE NOT IN TRAINING DATASET (ALSO NOT FROM ANOTHER YEAR)
    def DevideData(self, path_dataset, csv_folder, test_size = 0.1, validation_size = 0.15):
        # Load training data
        file = h5py.File(path_dataset, 'r')
        imgs_train = np.array(file.get('images'))
        imgs_mask_train = np.array(file.get('masks'))
        filenames = list(file.get('filenames'))
        filenames = [n.decode('UTF-8') for n in filenames]

        # Generate list of all unique image IDs
        ids = []
        for filename in filenames:
            filename_id = filename.split("_")[0]
            filename_id = int(filename_id)
            if filename_id not in ids:
                ids.append(filename_id)

        # Compute number of unique ids for testing (based on test_size variable)
        test_amount = int(test_size * len(ids))

        # Generate a random sample for the test-set (sample of image ids)
        ids_test = random.sample(ids, test_amount)

        ## COVERT IDS TO NUMPY ARRAY INDEXES ##
        # All indexes
        all_indexes = list(range(len(imgs_train)))

        # Get the numpy array indexes of the test-set ids
        indexes_test = []
        for i in range(len(filenames)):
            filename = filenames[i]
            filename_id = filename.split("_")[0]
            filename_id = int(filename_id)
            if filename_id in ids_test:
                indexes_test.append(i)

        # Create test dataset
        if len(indexes_test)>0 :
            x_test = imgs_train[indexes_test]
            y_test = imgs_mask_train[indexes_test]
            filenames_test = [filenames[i] for i in indexes_test]
        else:
            x_test = []
            y_test = []
            filenames_test = []        

        # Define indexes for validation and training 
        indexes_train = [x for x in all_indexes if x not in indexes_test]
        x = imgs_train[indexes_train]
        y = imgs_mask_train[indexes_train]

        # Split training+validation dataset in training dataset and validation dataset
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=validation_size, shuffle= True)

        # Write the filenames of the images of the testset to a csv file
        if len(filenames_test) >0 :
            with open((csv_folder + "/filenamesTestset.csv"),'w') as resultFile:
                for fn in filenames_test:
                    resultFile.write(fn)
                    resultFile.write("\n")

        return(x_train, y_train, x_val, y_val, x_test, y_test, filenames_test)
    
    # Write test-set to own h5 file
    def SaveTestData(self, h5_file, csv_testset, dest_h5_file):
        # h5_file:            original h5_file created before the splitting of data
        # csv_testset:        csv-file of image filenames which belong to the test set (images not used for training and validation)
        # dest_h5_file:       entire path and filename of h5_file of test set

        # Initialize result lists
        images_test = []
        masks_test = []
        filenames_test = []

        #Read original h5 file
        file = h5py.File(h5_file, 'r')
        images = np.array(file.get('images'))
        masks = np.array(file.get('masks'))
        filenames = list(file.get('filenames'))
        filenames = [n.decode('UTF-8') for n in filenames]

        # Open filenames of test-set
        data = pd.read_csv(test_set, sep = ',', header = None ) 
        filenames_testset = list(data[:][0])

        # Loop trough filenames in original h5 file and save them in new list when filename in filenames_test
        for i in range(len(filenames)):
            fn = filenames[i]
            if fn in filenames_testset:
                img = images[i]
                mask = masks[i]
                images_test.append(img)
                masks_test.append(mask)
                filenames_test.append(fn)
            else:
                continue

        # Read lists of arrays as single array
        images_test = np.array(images_test)
        masks_test = np.array(masks_test)
        filenames_test = [n.encode("ascii", "ignore") for n in filenames_test]

        with h5py.File(dest_h5_file, 'w') as hdf:
            hdf.create_dataset('images', data=images_test, compression='gzip', compression_opts=9)
            hdf.create_dataset('masks', data=masks_test, compression='gzip', compression_opts=9)
            hdf.create_dataset('filenames', data=filenames_test, compression='gzip', compression_opts=9)
        
        return (images_test, masks_test, filenames_test)

    # Load h5 file into numpy array
    def loadH5file(self, h5_file):
        h5_file= h5py.File(h5_file, 'r')   
        imgs_train = np.array(h5_file.get('images'))
        imgs_mask_train = np.array(h5_file.get('masks'))
        filenames = list(h5_file.get('filenames'))
        filenames = [n.decode('UTF-8') for n in filenames]
        return (imgs_train, imgs_mask_train, filenames)

    # Write normal numpy-arry to h5 file
    def writeH5file (self, images, dest_h5_file, filenames = None, masks = None):
        with h5py.File(dest_h5_file, 'w') as hdf:
            hdf.create_dataset('images', data=images, compression='gzip', compression_opts=9)
            if masks is not None:
                hdf.create_dataset('masks', data=masks, compression='gzip', compression_opts=9)
            if filenames is not None:
                hdf.create_dataset('filenames', data=filenames, compression='gzip', compression_opts=9)         

    # Get the bounding boxes of images which have been downloaded already
    def GetBoundingBoxesofImages(self, json_file, image_dir):
        # List all files in training data 
        files = os.listdir(image_dir)

        # fill list of IDS which had been selected for training
        id_list = []
        for file in files:
            if file.endswith('.tif'):
                img_id = file.split("_")[0]
                if img_id not in id_list:
                    id_list.append(img_id)

        id_list = list(map(int, id_list))
        id_list.sort()
        # Read bounding boxes from  json
        bb_list = []
        ids = []
        with open(json_file) as json_file:  
            data = json.load(json_file)
            for img_id in id_list:
                img_id = str(img_id)
                bb = data[img_id]['bounding_box']
                bb_list.insert(len(bb_list),bb)
                ids.insert(len(ids), img_id)
        print('Number of bounding-boxes: '+ str(len(bb_list)))
        return(bb_list, id_list)
    
