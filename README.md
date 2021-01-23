# WasteManagementClassifier

## Project: Classifying Waste into Organic and Inorganic
### Contributor: 
Phillip, Ridzwan and Mahfuzur

*Output Gist Can be viewed from:*  https://gist.github.com/mshaek/f89bcd760047c1fc44a836e6e83281c6

### Problem Statement:
Waste management is a big problem in our country. Most of the wastes end up in landfills. This leads to many issues like

* Increase in landfills
* Eutrophication
* Consumption of toxic waste by animals
* Leachate
* Increase in toxins
* Land, water and air pollution

APPROACH


Studied white papers on waste management
Analysed the components of household waste
Segregated into two classes (Organic and recyclable)
Automated the process by using IOT and machine learning
Reduce toxic waste ending in landfills

IMPLEMENTATION


Dataset is divided into train data (85%) and test data (15%)  

Training data - 22564 images  Test data - 2513 images

### Data Source: 
Data Collected from Kaggle. Link: https://www.kaggle.com/techsash/waste-classification-data

### Model Training
Input Image Size= 80 x 80 Color
Number of Classes = 2
Learning rate= 0.001
Image Tranformation= Flip, RandomCrop, Rotate
Batch Size= 32
Epochs= 10

### Training Evaluation Metrics
 #### of classes:    2
 Accuracy:        0.8039
 
 Precision:       0.8059
 
 Recall:          0.8007
 
 F1 Score:        0.8033
 

===Training Confusion Matrix=========================

  0    1

-----------

5650 1350 | 0 = 0
 

1395 5604 | 1 = 1
 
 
 ### Validation Metrics
 #### of classes:    2

Accuracy:        0.8183

Precision:       0.8135

Recall:          0.8260

F1 Score:        0.8197
 
 ===Validation Confusion Matrix=========================

0    1

-----------

2431  568 | 0 = 0

522 2478 | 1 = 1

## Total Training Time: Aprox 3:30 hours

