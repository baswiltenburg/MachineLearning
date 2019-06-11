import matplotlib.pyplot as plt
import os
import io
import shutil
import gdal, osr
import numpy as np
import rasterio as rio
import json
from sklearn.model_selection import train_test_split
import random
import cv2

class N2000_DataAugmentation():
    # Batch size is number of images you want to generate with specific augmentation technique
    
    def __init__(self, x_train, y_train):
        # Instance variables        
        self.x_train = x_train
        self.y_train = y_train
    
    def VerticalFlip(self, batch_size):   
        # Function to create list of random indices based on batch size
        def RandomIndices(start, end, batch_size): 
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
        
        def RandomIndices(start, end, batch_size): 
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