from pathlib import Path
import os
import h5py
import json
import rasterio as rio
from rasterio import crs
import json
import geopandas as gpd
import numpy as np
from rasterio import features
from affine import Affine
from shapely.geometry import shape, mapping, Polygon
import sklearn
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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
                src_img = src.read(1)            
                src_img[src_img > prob_treshold] = 1.0     
                transform = src.meta['transform']              
                # = src.meta['crs']

                # Vectorize probability raster
                for shp, val in features.shapes(src_img, transform=transform):
                    # Add polygons (shape) to vector list when raster value = 1 (positive class)
                    if val == 1:  
                        print(image_id)            
                        out_gdf.loc[index,'geometry'] = shape(shp)
                        out_gdf.loc[index,'img_id'] = image_id
                        out_gdf.loc[index,'file_name'] =  name
                        out_gdf.loc[index,'class'] =  cat
                        index += 1
                    else:
                        continue

    # Write geopandas dataframe to file
    out_gdf.to_file(driver = 'ESRI Shapefile', filename= geotiff_folder+"/"+cat+".shp")

geotiff_folder = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run9"
vectorizePrediction(geotiff_folder = geotiff_folder, prob_treshold=0.5)

geotiff_folder = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run9"
import os
import shapely, rasterio
import geopandas as gpd
from rasterio import features
from shapely import geometry
import rasterio as rio
prob_treshold = 0.5
from shapely.geometry import shape, mapping, Polygon
files = os.listdir(geotiff_folder)
epsg = 28992
out_gdf = gpd.GeoDataFrame(columns = ['img_id', 'file_name', 'class'], crs = {'init': 'epsg:'+str(epsg)})
index = 0
for i in range(len(files)):
    if files[i].endswith('.tif'): 
        image_id = files[i].split("_")[0]
        name = files[i].split("_")[0] + "_" + files[i].split("_")[1] + "_" + files[i].split("_")[2]   
        cat = 'h5130'
        with rio.open(geotiff_folder+"/" + files[i], mode = 'r') as src:                
            src_img = src.read(1)            
            src_img[src_img > prob_treshold] = 1.0     
            transform = src.meta['transform']            
        for shp, val in features.shapes(src_img, transform=transform, connectivity=4):
            # Add polygons (shape) to vector list when raster value = 1 (positive class)
            if val == 1:                             
                out_gdf.loc[index,'geometry'] = shape(shp)
                out_gdf.loc[index,'img_id'] = image_id
                out_gdf.loc[index,'file_name'] =  name
                out_gdf.loc[index,'class'] =  cat
                print(len(out_gdf)) 
                index += 1
            else:
                continue

out_gdf.to_file(driver = 'ESRI Shapefile', filename= geotiff_folder+"/"+cat+".shp")

# Write prediction statistics  
# Confusion matrix: count total true negaties, false positives, false negatives and true positives
def CalculateConfusionMatrix(prediction_directory, gt_directory, csv_file):
    from sklearn.metrics import confusion_matrix
    import pandas as pd    
    predictions = os.listdir(prediction_directory)
    gt = os.listdir(gt_directory)
    tresholds = [0.25, 0.5, 0.75]
    for treshold in tresholds: 
        con_matrix = [0,0,0,0] # tn, fp, fn, tp
        for i in range(len(predictions)):
            pred_id = predictions[i].split("_")[0]
            for j in range(len(gt)):
                gt_id = gt[j].split("_")[0]
                if pred_id == gt_id:
                    p_path = f"{prediction_directory}/{predictions[i]}"
                    gt_path = f"{gt_directory}/{gt[j]}"
                    if p_path.endswith('.tif') and gt_path.endswith('.tif'):
                        prediction = rio.open(p_path, 'r').read() 
                        prediction = prediction.transpose((1, 2, 0)) 
                        prediction = prediction.reshape(prediction.shape[0], prediction.shape[1])      
                        prediction[prediction<treshold] = 0
                        prediction[prediction > 0] = 1      
                        prediction = prediction.astype("int8")
                        ground_truth = rio.open(gt_path, 'r').read() 
                        ground_truth = ground_truth.transpose((1, 2, 0)) 
                        ground_truth = ground_truth.reshape(ground_truth.shape[0], ground_truth.shape[1])
                        ground_truth = ground_truth.astype("int8")
                        pred = np.ndarray.tolist(prediction)
                        y_gt = np.ndarray.tolist(ground_truth)
                        y_pred = [p for sublist in pred for p in sublist]
                        y_true = [g for sublist in y_gt for g in sublist]
                        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel() # labes was 0,1
                        con_matrix[0]+= tn
                        con_matrix[1]+= fp
                        con_matrix[2]+= fn
                        con_matrix[3]+= tp
                        if tp > 100:
                            print(gt_id)
        # Compute recall, tp / (tp + fn)
        recall = con_matrix[3] / (con_matrix[3]+con_matrix[2])
        # Compute precision, tp / (tp + fp)
        precision = con_matrix[3] / (con_matrix[3]+con_matrix[1])
        # Compute f1-score: 2 tp / (2 tp + fp + fn)
        f1_score = 2 * ((precision * recall) / (precision + recall))
        if treshold == tresholds[0]:
            # Create dataframe        
            data = {'treshold': [treshold], 'true_negative': [con_matrix[0]], 'false_positive':[con_matrix[1]], 'false_negative': [con_matrix[2]], 'true_postive':[con_matrix[3]], 'recall':[recall], 'precision':[precision], 'f1_score':[f1_score]}
            data_df = pd.DataFrame(data=data)
        else:
            # Add to dataframe
            data = {'treshold': treshold, 'true_negative': con_matrix[0], 'false_positive':con_matrix[1], 'false_negative': con_matrix[2], 'true_postive':con_matrix[3], 'recall':recall, 'precision':precision, 'f1_score':f1_score}
            data_df = data_df.append(data, ignore_index=True)
        print(data_df)    
    data_df.to_csv(csv_file)
    return(data_df)

prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run9/"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Images/Gelderland/2016/mask_modified"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run9/ConvolutionMatrixRun9.csv"
results_run9 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)

prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run3/"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Images/Gelderland/2016/mask_modified"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run3/ConvolutionMatrixRun3.csv"
results_run3 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)

prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run7/"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Images/Gelderland/2016/mask_modified"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run7/ConvolutionMatrixRun7.csv"
results_run7 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)

prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run5/"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Images/Gelderland/2016/mask_modified"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run5/ConvolutionMatrixRun5.csv"
results_run5 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)

prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run11/"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Images/Gelderland/2016/mask_modified"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run11/ConvolutionMatrixRun11.csv"
results_run11 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)

prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run12/"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Images/Gelderland/2016/mask_modified"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/H5130/Predicions/Run12/ConvolutionMatrixRun12.csv"
results_run12 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)






prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run2/BGT"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/mask"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run2/BGT/ConvolutionMatrixRun2.csv"
results_run2 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)
    
prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run3/BGT"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/mask"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run3/BGT/ConvolutionMatrixRun3.csv"
results_run2 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)
    
prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run4/BGT"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/mask"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run4/BGT/ConvolutionMatrixRun4.csv"
results_run2 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)
    
prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run5/BGT"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/mask"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run5/BGT/ConvolutionMatrixRun5.csv"
results_run2 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)
    
prediction_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run6/BGT"
gt_directory = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Images/Testing_BGT/mask"
csv_file = "C:/Users/wba/Internship/Data/5_TrainingData/Gras/Predictions/Run6/BGT/ConvolutionMatrixRun6.csv"
results_run2 = CalculateConfusionMatrix(prediction_directory = prediction_directory, gt_directory = gt_directory, csv_file = csv_file)
    

#https://www.statisticshowto.datasciencecentral.com/false-positive-definition-and-examples/
