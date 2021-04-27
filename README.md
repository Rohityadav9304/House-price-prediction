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
In this section we will import our dataset by using pandas function 
'df1=pd.read_csv('City_house_Data.csv')'
![Screenshot (296)](https://user-images.githubusercontent.com/77377586/116209392-7dd86f00-a75f-11eb-91b2-84c03f561355.png)

## Dropping some less important features :
We will drop that features which are very less important by using 
'df2 = df1.drop(['area_type','society','balcony','availability'],axis='columns')'
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
'def is_float(x):
    try:
        float(x)
    except:
        return False
    return True'
    
    
 ![Screenshot (300)](https://user-images.githubusercontent.com/77377586/116211003-0572ad80-a761-11eb-9514-1e4ae887d37f.png)
 
 so we in this data we will convert these types of data into float by making average of both numbers by using def.
 '''
 def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
 df4 = df3.copy()
df4.total_sqft = df4.total_sqft.apply(convert_sqft_to_num)
'''

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
'''
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
    
 df7 = remove_pps_outliers(df6)
 after calling this function will remove those outliers which are not in range
 '''
 
#### Now We will check property price of 3bhk is greater than the price of 2bhk or not
this can be due to the location and services of that property the property may be at good location so that it's price is high
for this we will use def function and draw a scatter plot
'''
def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")
'''
#### Plot
![download](https://user-images.githubusercontent.com/77377586/116218364-14109300-a768-11eb-93ab-1836c3989096.png)

'plot_scatter_chart(df7,"Hebbal")'
# Hebbal Plot
![download (1)](https://user-images.githubusercontent.com/77377586/116218536-402c1400-a768-11eb-9a0c-95183594202e.png)
### Removing Outliers:
to remove outliers we will use function method
'''
def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
            
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_bhk_outliers(df7)
    
'''
### Checking Improvements
'plot_scatter_chart(df8,"Rajaji Nagar")'

![download (2)](https://user-images.githubusercontent.com/77377586/116219218-00b1f780-a769-11eb-944b-1cb918d1f81c.png)

'plot_scatter_chart(df8,"Hebbal")'
![download (3)](https://user-images.githubusercontent.com/77377586/116219307-17584e80-a769-11eb-957b-46d7c6ce8d67.png)
#### Here we have seen that majority of outliers has been removed
### Plotting Histogram
'''
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df8.price_per_sqft,rwidth=0.8)
plt.xlabel("Price Per Square Feet")
plt.ylabel("Count")
'''
![download (4)](https://user-images.githubusercontent.com/77377586/116219534-57b7cc80-a769-11eb-9644-23509fb34b10.png)

### Looking at the bathroom features 
Here we observe that some bathroom features are more than 10 values which are unusual so we will take only 
those bathroom features whose values are less than the bedrooms 
### Plotting bathroom histogram
'''
df8.bath.unique()
plt.hist(df8.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")
'''
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
 #### 2. Separating features and labels
     'X = df12.drop(['price'],axis='columns')'
     'y = df12.price'
 #### 3. Training Testing and splitting : We will take 20% test size 
   '''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
'''
 ## Training Regression Model 
 '''
 from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)
'''
output=0.8452277697874312
### Shuffle split
use to shuffle the data into training and testing sets
'''
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)
'''
output =array([0.82430186, 0.77166234, 0.85089567, 0.80837764, 0.83653286])

# Creating Grid searchCv model to identify best model
'''
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
'''
# Creating a function which detects best model on the basis of score
'''
 def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])
    '''
    
 # Finding Best model :
 'find_best_model_using_gridsearchcv(X,y)'
 ![Screenshot (308)](https://user-images.githubusercontent.com/77377586/116223450-25a86980-a76d-11eb-92d1-1beb87b5d9c6.png)

 This will provide all the scores performed by all models
 we will select that model whose score is high in comparision to other models
 
 #### So our linear regression model have given best score hence we will choose this model for predictions 
 
 ## Making Prediction function :
 here we will create a prediction function which takes input as location , sqft,bath,bhk .
 '''
 def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]
    '''

# Making Predictions :
Now our model is ready for predictions 
for example : If we want to predict price of house in indra nagar of 1000 sqft with 3 bathroom and of 3bhk.
We will give following command to our model :
'predict_price('Indira Nagar',1000, 3, 3)'
Output=184.5843020203347 
here we can say that property price is nearly 184.5843 lakh 
 


 












 

