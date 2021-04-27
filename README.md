# House price prediction :
In this project we will see that what kind of steps an challanges a data scientist working for a big community goes threw in his day today's life.
# Problem : 
Suppose we have to build a model that can predict property price based on on certain features such as sqare ft , location , bathrooms , kitchen etc.
# Project architecture :
We are going to take home price dataset of a city from kaggle.com .
by using this dataset we will build a machine learning  model which will predict prices.
# Steps in building a model :
### In model building we will cover some data science concepts :
Data cleaning , feature engineering ,dimensionality reduction , outlier detection and removal , GridSearchCV .
# Step 1 : Importing libraries and datasets and looking at the less important features
## Importing Libraries : 
In this step we will import some libraries which is used for making doing some operations in data like data loading as visualization and numerical operations.
### we will use following libraries
1. pandas : for loading and dividing datasets into features and labels.
2. numpy : used for numeric operation in dataset
3. Matplotlib : for visualization
## Importing Datasets :
In this section we will import our dataset by using pandas function.
![Screenshot (296)](https://user-images.githubusercontent.com/77377586/116209392-7dd86f00-a75f-11eb-91b2-84c03f561355.png)

## Dropping some less important features :
We will drop that features which are very less important by using  df1.drop([' ',' '] ,axis=column)
![Screenshot (297)](https://user-images.githubusercontent.com/77377586/116209500-99dc1080-a75f-11eb-9264-dd541da2a7e8.png)


# Step 2 : Data Cleaning : Most important step 
Data cleaning is the most important step 
we do data cleaning so that we can drop some values and modify some value so that our model performance will be better.
we do data cleaning to increase our model performance score and reduce errors.
### Dropping NA values :
Since our data is very big and NA values are very less in compared to data size therefore we can drop NA values it will not affect our model accuracy.
we can also find mean or average values and fit it into NA values but for this data we are not doing this.
### Seperating uniques and making a new feature :
since we can clearly see in data that our Size feature contains some valuese like
![Screenshot (298)](https://user-images.githubusercontent.com/77377586/116209762-da3b8e80-a75f-11eb-8504-9fd52f570027.png)

Here Bedroom and BHK features are same but names are different and also this can create error so we will make a new feature bhk which will contain only integer 
values of this size feature
'df3['bhk'] = df3['size'].apply(lambda x: int(x.split(' ')[0])' 
Now the new array will be 
![Screenshot (299)](https://user-images.githubusercontent.com/77377586/116209819-e9bad780-a75f-11eb-885a-042ca6ed39d1.png)
### Analysing the values in total sqare_ft section which are not float
we can analyse by using def method 
![Screenshot (309)](https://user-images.githubusercontent.com/77377586/116247641-b17bbf00-a788-11eb-85e7-21b3c093e954.png)

    
 ![Screenshot (300)](https://user-images.githubusercontent.com/77377586/116211003-0572ad80-a761-11eb-9514-1e4ae887d37f.png)
 
 so we in this data we will convert these types of data into float by making average of both numbers by using def.
 ![Screenshot (310)](https://user-images.githubusercontent.com/77377586/116247747-c8baac80-a788-11eb-88b0-fbfd79aac95a.png)

so the average of both has been settled in place of the previous 

# Step 2 : Feature Engineering 
In this step we will create new feature so that it can help in the outlier detection
![Screenshot (301)](https://user-images.githubusercontent.com/77377586/116212549-90a07300-a762-11eb-80bc-e8c07c26569b.png)

here we have created price per square ft feature.
### Problem with location :
Since we have so many locatios and if we do one hot encoding to convert location into dummy variable then we will be having very big data 
of dummy variables so to minimise this we will look at the loaction_stats and those which are less than 10 will be categorised into another 
section

# Step 3 : Outlier detection and Removal
### Outlier detection:
Outliers are generally those values which is not accepted or those can make our model prediction poor 
so firs we detects outliers 
#### Assuming a threshold value 
we will assume a threshold value of total_sqft_per_bedroom =300
'df5[df5.total_sqft/df5.bhk<300]'
we will only take those values which are greater than thresholad value
'df6=df5[~df5.total_sqft/df5.bhk<300]'
#### Creating a function for which (mean-standard-deviation)< price_per_sq_ft<=(mean+standard_deviation)
![Screenshot (311)](https://user-images.githubusercontent.com/77377586/116247877-ea1b9880-a788-11eb-877d-2d140813dc78.png)

#### Now We will check property price of 3bhk is greater than the price of 2bhk or not
this can be due to the location and services of that property the property may be at good location so that it's price is high
for this we will use def function and draw a scatter plot
![Screenshot (312)](https://user-images.githubusercontent.com/77377586/116247950-fdc6ff00-a788-11eb-8153-73e23b65bef1.png)


plot_scatter_chart(df7,"Rajaji Nagar")
#### Plot
![download](https://user-images.githubusercontent.com/77377586/116218364-14109300-a768-11eb-93ab-1836c3989096.png)

'plot_scatter_chart(df7,"Hebbal")'
# Hebbal Plot
![download (1)](https://user-images.githubusercontent.com/77377586/116218536-402c1400-a768-11eb-9a0c-95183594202e.png)
### Removing Outliers:
to remove outliers we will use function method
![Screenshot (313)](https://user-images.githubusercontent.com/77377586/116248013-0e777500-a789-11eb-8161-ac7be68a86d7.png)

### Checking Improvements
'plot_scatter_chart(df8,"Rajaji Nagar")'

![download (2)](https://user-images.githubusercontent.com/77377586/116219218-00b1f780-a769-11eb-944b-1cb918d1f81c.png)

'plot_scatter_chart(df8,"Hebbal")'
![download (3)](https://user-images.githubusercontent.com/77377586/116219307-17584e80-a769-11eb-957b-46d7c6ce8d67.png)
#### Here we have seen that majority of outliers has been removed
### Plotting Histogram
![Screenshot (314)](https://user-images.githubusercontent.com/77377586/116248092-24853580-a789-11eb-9925-9a46d6e21eaa.png)

![download (4)](https://user-images.githubusercontent.com/77377586/116219534-57b7cc80-a769-11eb-9644-23509fb34b10.png)

### Looking at the bathroom features 
Here we observe that some bathroom features are more than 10 values which are unusual so we will take only 
those bathroom features whose values are less than the bedrooms 
### Plotting bathroom histogram
![Screenshot (315)](https://user-images.githubusercontent.com/77377586/116248128-2c44da00-a789-11eb-8050-02d16ac75bd4.png)

![download (5)](https://user-images.githubusercontent.com/77377586/116220275-24297200-a76a-11eb-972a-fa360dddb86f.png)

### Dropping some features :
1 Price_per_sqft : we drop this because we used it for outlier detection 2 Size : we dropped this because we have already bhk feature
![Screenshot (307)](https://user-images.githubusercontent.com/77377586/116220672-84b8af00-a76a-11eb-9332-6ec88a958723.png)
 ### Now finally we have done all the steps in data cleaning and feature engineering 
 
 # Step 4 : Final Step
 # Building and selecting Best Regression Model :
 Befor selecting model we do some steps
 #### 1. One hot Encoding : to convert location features into dummy variables
    'dummies = pd.get_dummies(df10.location)'
 #### 2. Dropping feature : 'Other'
 we will drop the the other dummy variable since it is less important
 #### 3. Dropping Location Column:
 We will the drop the location column since it don't contain numeric values and we have already created dummmy variables of location column features
 which are in numeric for for further processing
 #### 4. Separating features and labels
     'X = df12.drop(['price'],axis='columns')'
     'y = df12.price'
 #### 5. Training Testing and splitting : We will take 20% test size 
![Screenshot (316)](https://user-images.githubusercontent.com/77377586/116248182-3f57aa00-a789-11eb-84d4-eff113587678.png)

 ## Training Regression Model 
![Screenshot (317)](https://user-images.githubusercontent.com/77377586/116248198-467eb800-a789-11eb-94d5-643be155543a.png)

### Shuffle split
use to shuffle the data into training and testing sets
![Screenshot (318)](https://user-images.githubusercontent.com/77377586/116248237-51394d00-a789-11eb-8108-8b66f1261f46.png)

# Creating Grid searchCv model to identify best model
![Screenshot (319)](https://user-images.githubusercontent.com/77377586/116248277-59918800-a789-11eb-82f7-e345e39084b5.png)

# Creating a function which detects best model on the basis of score
![Screenshot (320)](https://user-images.githubusercontent.com/77377586/116248341-6a41fe00-a789-11eb-9c91-9bbc5fc79d18.png)

 # Finding Best model :
 'find_best_model_using_gridsearchcv(X,y)'
 ![Screenshot (308)](https://user-images.githubusercontent.com/77377586/116223450-25a86980-a76d-11eb-92d1-1beb87b5d9c6.png)

 This will provide all the scores performed by all models
 we will select that model whose score is high in comparision to other models
 
 #### So our linear regression model have given best score hence we will choose this model for predictions 
 
 ## Making Prediction function :
 here we will create a prediction function which takes input as location , sqft,bath,bhk .
 ![Screenshot (321)](https://user-images.githubusercontent.com/77377586/116248363-70d07580-a789-11eb-903e-8ee6dd361ce4.png)

# Making Predictions :
Now our model is ready for predictions 
for example : If we want to predict price of house in indra nagar of 1000 sqft with 3 bathroom and of 3bhk.
We will give following command to our model :
'predict_price('Indira Nagar',1000, 3, 3)'
Output=184.5843020203347 
here we can say that property price is nearly 184.5843 lakh 
 


 












 

