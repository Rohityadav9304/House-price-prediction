# Project Name :
## House-Price-Prediction :
This project is created to predict house prices in any area from given data based on house parameters.
# Parameters Are
1. Area in sq ft
2. Locality
3. Number of Bathrooms
4. Number of bedrooms 
In the peoject we Take Input 

# Project Type :
## Regression Project : 

# Steps Involved :
1.We first build a model using sklearn and linear regression using home prices dataset from kaggle.com.
# Model Building Envolves Various steps :
1. Importing Libraries :
     import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib 
matplotlib.rcParams["figure.figsize"] = (20,10)

2. Data Loading 
3. Matplotlib for data visualization
4. Data Cleaning :
   Numpy and Pandas for data cleaning.
5.outlier detection and removal.
6. feature engineering.
7. dimensionality reduction .
8. gridsearchcv for hyperparameter tunning.
9. k fold cross validation  


# Predicting Output :
For predicting Accurate output we will have to input follwing parameters serialwisw
1. Location : Example-'1st Phase JP Nagar'
2. sqft : Square ft : Ex- 1000
3. bath : Number of bathrooms : Example-2
4. bhk  : Bedroom Hall kichen : Example -2



price=predict_price('1st Phase JP Nagar',1000, 2, 2)
 print(price)
 output : Estimated price predicted by model


 

