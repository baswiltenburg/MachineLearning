import tensorflow
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
from sklearn.model_selection import train_test_split
import random

class N2000_DataAugmentation():
    # Batch size is number of images you want to generate with specific augmentation technique
    
    def __init__(self, x_train, y_train):
        # Instance variables        
        self.x_train = x_train
        self.y_train = y_train
    
    def VerticalFlip(self, batch_size):   
        # Function to create list of random indices based on batch size
        def RandomIndices(start, end, batch_size): 
            import random 
            indices = []   
            for j in range(batch_size): 
                indices.append(random.randint(start, end))  
            return (indices)
        
        # Generate additional training data based on batch-size and VerticalFlip augmentation
        indices = RandomIndices(0, (len(self.x_train)-1), batch_size)
        x_train_sub = self.x_train[indices]
        y_train_sub = self.y_train[indices]
        
        x_train_vf = [np.flip(x, axis=0) for x in x_train_sub]
        x_train_vf = np.array(x_train_vf)
        y_train_vf = [np.flip(x, axis=0) for x in y_train_sub]
        y_train_vf = np.array(y_train_vf)
        return (x_train_vf, y_train_vf)
    
    def HorizontalFlip(self, batch_size):  
        # Function to create list of random indices based on batch size
        def RandomIndices(start, end, batch_size): 
            import random 
            indices = []   
            for j in range(batch_size): 
                indices.append(random.randint(start, end))  
            return (indices)
        
        # Generate additional training data based on batch-size and HorizontalFlip augmentation
        indices = RandomIndices(0, (len(self.x_train)-1), batch_size)
        x_train_sub = self.x_train[indices]
        y_train_sub = self.y_train[indices]
        
        x_train_hf = [np.flip(x, axis=1) for x in x_train_sub]
        x_train_hf = np.array(x_train_hf)
        y_train_hf = [np.flip(x, axis=1) for x in y_train_sub]
        y_train_hf = np.array(y_train_hf)
    
        return (x_train_hf, y_train_hf)
    
    def RandomRotation(self, batch_size):
        import random
        import cv2
        def RandomIndices(start, end, batch_size): 
            import random 
            indices = []   
            for j in range(batch_size): 
                indices.append(random.randint(start, end))  
            return (indices)

        rotated_images = []
        rotated_mask_images = []
        
        # Generate additional training data based on batch-size and RandomRotation augmentation
        indices = RandomIndices(0, (len(self.x_train)-1), batch_size)
        x_train_sub = self.x_train[indices]
        y_train_sub = self.y_train[indices]
        for i in range(len(x_train_sub)):
            random_degree = random.uniform(-30, 30)
            # Create new arrays of random rotated images
            x_train_rot = x_train_sub[i]
            y_train_rot = y_train_sub[i]
            x_rows, x_cols, x_ch = x_train_rot.shape  
            y_rows, y_cols, y_ch = y_train_rot.shape  

            # Rotate the training image
            x_rotation = cv2.getRotationMatrix2D((x_cols/2, x_rows/2), random_degree,1)
            x_img_rot = cv2.warpAffine(x_train_rot, x_rotation, (x_cols, x_rows))
            #x_img_rot = np.expand_dims(x_img_rot , axis=0) 
            rotated_images.append(x_img_rot)

            # Rotate the mask image
            y_rotation = cv2.getRotationMatrix2D((y_cols/2, y_rows/2), random_degree,1)
            y_img_rot = cv2.warpAffine(y_train_rot, y_rotation, (y_cols, y_rows))
            y_img_rot = np.expand_dims(y_img_rot , axis=-1) 
            #y_img_rot = y_img_rot.reshape(512,512) # RESHAPE FOR plt.imshow(y_img_rot)
            rotated_mask_images.append(y_img_rot)       

        # Read lists of arrays as single array
        x_train_rr = np.array(rotated_images)
        y_train_rr = np.array(rotated_mask_images)
        #print(y_train_rr.shape, ( " length: " + str(len(y_train_rr))))
        #print(x_train_rr.shape, (" length: " + str(len(x_train_rr))))
        
        return (x_train_rr, y_train_rr)
    

class N2000_DataPreparation():          
    def __init__(self, image_size):
        # Variables        
        self.image_size = image_size

    # GET 'CHECK' IMAGES FROM BLOB STORAGE #
    # ALTERNATIVE https://github.com/agermanidis/pigeon
    def getBlobData(self, path_training_data, config = '../config.txt', blobname = 'n2000-images-corrected'):
        block_blob = dimension.BlobConnect(config).block_blob_service()
        file_list = block_blob.list_blob_names(blobname)
        for item in file_list.items:
            block_blob.get_blob_to_path(blobname, item, (path_training_data+"/"+item))
            
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
                    if maskname.endswith(".tif"):
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
                # File is not a .tif file
                continue
                
    # REMOVE IMAGES FROM TRAINING DATA OF WHICH ITS BINARY MASK IMAGE IS NOT OF CORRECT SHAPE
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
                name_list = file.split("_")
                name = name_list[0] + "_" + name_list[1] + "_" + name_list[2]
                name = name.split(".")[0]  
                for wf in wrong_items:
                    if wf == name:
                        os.unlink(self.path_training_data + "/" + img_file)
                        
    # WRITE TRAINING AND MASK IMAGES TO H5 FORMAT AND ENSURE SIMILAR SHAPE FORMAT
    def CreateH5_files(self, path_training_data, name_file = "Dataset_train.h5"):
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
                    # img = Image.open(path_training_data + "/" + filename)            
                    # arr = np.array(img)
                    # images.append(arr)
                    img = rio.open(path_training_data + "/" + filename)  
                    array = img.read()
                    array = array.transpose((1, 2, 0))  
                    #array = np.expand_dims(array, axis=0)    
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
                            # If file is not a mask-file
                            continue

        # Read lists of arrays as single array
        images = np.array(images)
        masks = np.array(masks)
        filenames = [n.encode("ascii", "ignore") for n in filenames]
        #print(masks.shape, ( " length: " + str(len(masks))))
        #print(images.shape, (" length: " + str(len(images))))

        with h5py.File(path_training_data + "/" + name_file, 'w') as hdf:
            hdf.create_dataset('images', data=images, compression='gzip', compression_opts=9)
            hdf.create_dataset('masks', data=masks, compression='gzip', compression_opts=9)
            hdf.create_dataset('filenames', data=filenames, compression='gzip', compression_opts=9)
            
    # Load h5 file
    def loadH5file(self, h5_file):
        h5_file= h5py.File(h5_file, 'r')   
        imgs_train = np.array(h5_file.get('images'))
        imgs_mask_train = np.array(h5_file.get('masks'))
        filenames = list(h5_file.get('filenames'))
        filenames = [n.decode('UTF-8') for n in filenames]
        return (imgs_train, imgs_mask_train, filenames)

    # write h5 file
    def writeH5file (self, images, dest_h5_file, filenames = None, masks = None):
        with h5py.File(dest_h5_file, 'w') as hdf:
            hdf.create_dataset('images', data=images, compression='gzip', compression_opts=9)
            if masks is not None:
                hdf.create_dataset('masks', data=masks, compression='gzip', compression_opts=9)
            if filenames is not None:
                hdf.create_dataset('filenames', data=filenames, compression='gzip', compression_opts=9)
                    
    # DEVIDE TRAINING DATA IN TRAINING DATASET, VALIDATION DATASET AND TEST SET (70, 20, 10)                
    def DevideData(self, path_dataset):
        from sklearn.model_selection import train_test_split
        
        # Load training data
        file = h5py.File(path_dataset, 'r')
        imgs_train = np.array(file.get('images'))
        imgs_mask_train = np.array(file.get('masks'))
        
        # Split dataset into a training+validation dataset and test dataset
        x_train_val, x_test, y_train_val, y_test = train_test_split(imgs_train, imgs_mask_train, test_size=0.1, shuffle= True)
        # Split training+validation dataset in training dataset and validation dataset
        x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.15, shuffle= True)
        
        return (x_train, y_train, x_val, y_val, x_test, y_test)
    
    # DEVIDE DATASETS IN VALIDATION, TEST EN TRAIN AND CREATE FILE WITH FILENAMES OF TESTSET
    # TEST SET ARE IMAGES WHICH ARE NOT IN TRAINING DATASET (ALSO NOT FROM ANOTHER YEAR)
    def DevideData2(self, path_dataset, csv_folder, test_size = 0.1, validation_size = 0.15):
        from sklearn.model_selection import train_test_split
        import random

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
        x_test = imgs_train[indexes_test]
        y_test = imgs_mask_train[indexes_test]

        filenames_test = [filenames[i] for i in indexes_test]

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
    def testSet_to_H5file(self, h5_file, csv_testset, dest_h5_file):
        # h5_file:            original h5_file created before the splitting of data
        # csv_testset:        csv-file of image filenames which belong to the test set (images not used for training and validation)
        # dest_h5_file:       entire path and filename of h5_file of test set

        import pandas as pd 
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

    def mergeH5files(self, paths_list, dest_folder, name_file = 'merged_data.h5', write = True):
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

    # NORMALIZE TRAINING DATA AND GROUND TRUTH DATA AND APPLY THE STATISTICS TO VALIDATION AND TEST DATA
    def NormalizeData(self, x_train, y_train, x_val, y_val, x_test, y_test, csv_folder, csv_name = "NormalizationStats.csv"):
        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')
        y_train = y_train.astype('float32')
        y_val = y_val.astype('float32')
        y_test = y_test.astype('float32')

        mean = np.mean(x_train)  # mean for data centering (derived from training data !)
        std = np.std(x_train)  # std for data normalization (derived from training data !)

        x_train -= mean # mean subtraction
        x_train /= std  # normalization

        # Apply it to test and validation data
        x_val -= mean
        x_val /= std

        x_test -= mean
        x_test /= std

        # Rescale normalized images to 0 - 1  
        max_x_train = np.max(x_train)
        min_x_train = np.min(x_train)
        max_x_val = np.max(x_val)
        min_x_val = np.min(x_val)
        max_x_test = np.max(x_test)
        min_x_test = np.min(x_test)
        
        x_train = (x_train-min_x_train)/(max_x_train-min_x_train)

        # Rescale normalized images to 0 - 1  (NO USE STATS TRAINING DATA, CHECK IF THIS IS RIGHT)
        x_val = (x_val-min_x_val)/(max_x_val-min_x_val)

        # Rescale normalized images to 0 - 1 (NO USE STATS TRAINING DATA, CHECK IF THIS IS RIGHT)
        x_test = (x_test-min_x_test)/(max_x_test-min_x_test)

        # Normalize ground-truth data as well [0-1]
        y_train /= 255
        y_val /= 255
        y_test /= 255
        
        stats = [mean, std, min_x_train, max_x_train]
        def format(value):
            return "%.3f" % value
        
        stats_formatted = [format(v) for v in stats]
        
        with open((csv_folder + "/" + csv_name),'w') as resultFile:
            for stat in stats_formatted:
                print(stat)
                resultFile.write(stat)
                resultFile.write("\n")

        return (x_train, y_train, x_val, y_val, x_test, y_test, stats)

    # Normalize training dataset and save normalization statistics to file
    def NormalizeTrainingData(self, x_train, csv_folder, csv_name = 'NormalizationStats.csv'):
        x_train = x_train.astype('float32')

        mean = np.mean(x_train)  # mean for data centering (derived from training data !)
        std = np.std(x_train)  # std for data normalization (derived from training data !)

        x_train -= mean # mean subtraction
        x_train /= std  # normalization

        # Rescale normalized images to 0 - 1  
        max_x_train = np.max(x_train)
        min_x_train = np.min(x_train)
        
        x_train = (x_train-min_x_train)/(max_x_train-min_x_train)
       
        stats = [mean, std, min_x_train, max_x_train]
        def format(value):
            return "%.3f" % value
        
        stats_formatted = [format(v) for v in stats]
        
        with open((csv_folder + "/" + csv_name),'w') as resultFile:
            for stat in stats_formatted:
                print(stat)
                resultFile.write(stat)
                resultFile.write("\n")
        resultFile.close()

        return (x_train, stats)

    # Convert from 0-255 to scale 0-1
    def NormalizeGroundTruth(self, y_train):
            y_train = y_train.astype('float32')    
            # Normalize ground-truth data as well [0-1]
            y_train /= 255

            return (y_train)
       
    # NORMALIZE TRAINING AND GROUND-TRUTH DATA DATA MIN-MAX
    def NormalizeMinMax(self, x_train, y_train, x_val, y_val, x_test, y_test, csv_folder, csv_name = "NormalizationStats.csv"):
        x_train = x_train.astype('float32')
        x_val = x_val.astype('float32')
        x_test = x_test.astype('float32')
        y_train = y_train.astype('float32')
        y_val = y_val.astype('float32')
        y_test = y_test.astype('float32')

        # Rescale normalized images to 0 - 1  
        max_x_train = np.max(x_train)
        min_x_train = np.min(x_train)
        max_x_val = np.max(x_val)
        min_x_val = np.min(x_val)
        max_x_test = np.max(x_test)
        min_x_test = np.min(x_test)
        x_train = (x_train-min_x_train)/(max_x_train-min_x_train)
        x_val = (x_val-min_x_train)/(max_x_train-min_x_train)
        x_test = (x_test-min_x_train)/(max_x_train-min_x_train)

        # Normalize ground-truth data as well [0-1]
        y_train /= 255
        y_val /= 255
        y_test /= 255
        
        stats = [min_x_train, max_x_train]
        def format(value):
            return "%.3f" % value
        
        stats_formatted = [format(v) for v in stats]
        
        with open((csv_folder + "/" + csv_name),'w') as resultFile:
            for stat in stats_formatted:
                print(stat)
                resultFile.write(stat)
                resultFile.write("\n")
        
        return (x_train, y_train, x_val, y_val, x_test, y_test, stats)
    
    # NORMALIZE TRAINING DATA DATA MIN-MAX
    def NormalizeTrainingDataMinMax(self, x_train, csv_folder, csv_name = "NormalizationStats.csv"):
        x_train = x_train.astype('float32')
        # Rescale normalized images to 0 - 1  
        max_x_train = np.max(x_train)
        min_x_train = np.min(x_train)
        x_train = (x_train-min_x_train)/(max_x_train-min_x_train)

        stats = [min_x_train, max_x_train]
        def format(value):
            return "%.3f" % value
        
        stats_formatted = [format(v) for v in stats]
        
        with open((csv_folder + "/" + csv_name),'w') as resultFile:
            for stat in stats_formatted:
                print(stat)
                resultFile.write(stat)
                resultFile.write("\n")
        
        return (x_train, stats)
    
    # Apply normalization to unseen data
    def NormalizeArrayFromH5(self, h5_file, stats_file):
        # Load training data
        file = h5py.File(h5_file, 'r')
        x_test = np.array(file.get('images'))
        y_test = np.array(file.get('masks'))        
        
        x_test = x_test.astype('float32')
        y_test = y_test.astype('float32')
        
        # dest_stats: Folder met mean, sigma, minimum and maxium value of training data
        # Read statsfile
        stats_list = []
        with open(stats_file, 'r') as stats:
            for stat in stats:
                stats_list.append(float(stat))
                
        mean = stats_list[0]
        std = stats_list[1]
        min_value = stats_list[2]
        max_value = stats_list[3]
        
        x_test -= mean # mean subtraction
        x_test /= std  # normalization       

        # Rescale normalized images to 0 - 1 (NO USE STATS TRAINING DATA, CHECK IF THIS IS RIGHT)
        x_test = (x_test-min_value)/(max_value-min_value)

        # Normalize ground-truth data as well [0-1]
        y_test /= 255
        
        return (x_test, y_test)

    # Create training dataset of normalized images
    def NormalizeImage(self, file_path, file_name, dest_folder, stats_file):
        # Read image as numpy array
        with rio.open(file_path, "r") as data:
            array = data.read()   
            meta = data.meta   
        array = array.astype('float32')      

        # Read statsfile with normalization statistics
        stats_list = []
        with open(stats_file, 'r') as stats:
            for stat in stats:
                stats_list.append(float(stat))
        stats.close()
        # Get statistics      
        mean = stats_list[0]
        std = stats_list[1]
        min_value = stats_list[2]
        max_value = stats_list[3]
        # Normalize image
        array -= mean # mean subtraction
        array /= std  # normalization      
        # Rescale normalized images to 0 - 1
        array = (array-min_value)/(max_value-min_value)
        # Write normalized image to file
        meta.update({"dtype":"float32"})
        with rio.open(f"{dest_folder}/{file_name}", "w", **meta) as dst:
            dst.write(array)

    # Get the bounding boxes of images which have been downloaded already
    def getBoundingBoxesofImages(self, json_file, image_dir):

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
    
