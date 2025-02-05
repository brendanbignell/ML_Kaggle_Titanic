![image](https://github.com/user-attachments/assets/88fb2c10-30de-4ae4-ad45-d24e8abb5c97)

# ML_Kaggle_Titanic

## Why

The main reason for doing this is to compare building an ML model in C#/.NET vs Python.

## Overview

This is a project to predict the survival of passengers on the Titanic. 

The dataset is from Kaggle and the goal is to predict whether a passenger survived or not. 
The dataset contains 891 rows and 12 columns.

Comparison of LightGBM and FastTree and a deep learning ensemble model consisting of:
	SDCA Logistic Regression
	FastForest
	LBFGS Logistic Regression

Usage of Nvidia GPU for training also tested although on this trivial ammount of data it is not needed.

## Example Output

`Checking CUDA availability...  
`CUDA found at: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6  
`Loading data...  

`Training Standard LightGBM...  
`Starting LightGBM training...  

`Training FastTree...  
`Starting FastTree training...  

`Training Deep Learning Model...  
`Training ensemble model...  

`Model Comparison Results:  
`Model Type           Accuracy        AUC             F1 Score        Training Time   
`Standard LightGBM    0.8799          0.9585          0.8366          0.20s  
`FastTree             0.8878          0.9604          0.8471          0.21s  
`Deep Learning        0.9910          0.9999          0.9883          0.88s  
