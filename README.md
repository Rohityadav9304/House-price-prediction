# Project Name :
## House-Price-Prediction :
This project is created to predict house prices in any area from given data based on house parameters.
# Parameters Are
1. Area in sq ft
2. Locality
3. Number of Bathrooms
4. Number of bedrooms 
In the peoject we Take Input 
# Dataset : City-House-data.csv

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

2. Data Loading :
  1.Load  home prices into a dataframe .
  2.area_type	availability	location	size	society	total_sqft	bath	balcony	price
  3.Drop features that are not required to build our model.
  
3. Matplotlib for data visualization

4. Data Cleaning :
   Numpy and Pandas for data cleaning.
   Handle NA values
   
5. feature engineering: 
   1.Add new feature(integer) for bhk (Bedrooms Hall Kitchen)
   2.Add new feature called price per square feet
   
6. dimensionality reduction .
   Any location having less than 10 data points should be tagged as "other" location. This way number of categories can be reduced by 
   huge amount. Later on when we do one hot encoding, it will help us with having fewer dummy columns.
   
7. Outlier Removal Using Business Logic :
    As a data scientist when you have a conversation with your business manager (who has expertise in real estate), 
    he will tell you that normally square ft per bedroom is 300 (i.e. 2 bhk apartment is minimum 600 sqft. 
    If you have for example 400 sqft apartment with 2 bhk than that seems suspicious and can be removed as an outlier.
    We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft.
    We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom 
    apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk. 
    
8. Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment
9. Outlier Removal Using Bathrooms Feature :
 It is unusual to have 2 more bathrooms than number of bedrooms in a home
 Again the business manager has a conversation with you (i.e. a data scientist) that if you have 4 bedroom home and even if
 you have bathroom in all 4 rooms plus one guest bathroom, you will have total bath = total bed + 1 max. Anything above that 
 is an outlier or a data error and can be removed.
10. We can see that in 5 iterations we get a score above 80% all the time. This is pretty good but we want to test few other algorithms
    for regression to see if we can get even better score. We will use GridSearchCV for this purpose.
11. Find best model using GridSearchCV
   Based on above results we can say that LinearRegression gives the best score. Hence we will use that.
   Test the model for few properties



# Predicting Output :
For predicting Accurate output we will have to input follwing parameters serialwisw
1. Location : Example-'1st Phase JP Nagar'
2. sqft : Square ft : Ex- 1000
3. bath : Number of bathrooms : Example-2
4. bhk  : Bedroom Hall kichen : Example -2



price=predict_price('1st Phase JP Nagar',1000, 2, 2)
 print(price)
 output : Estimated price predicted by model


 

