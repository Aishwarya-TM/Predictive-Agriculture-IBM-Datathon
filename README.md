# SVE-03 MODEL MAVENS TEAM
# Predictive-Agriculture-IBM-Datathon
## AIM:
To accurately predict the estimated quantity of crops that will be harvested from given agricultural area.
## ALGORITHM:
Step - 01: Import the necessary machine learning libraries.

Step - 02: Load the Dataset.

Step - 03: Preprocessing the Dataset.

Step - 04: Performing Exploratory Data Analysis.

Step - 05: Using Column Transformers.

Step - 06: Splitiing the Dataset for Training and Testing.

Step - 07: Training Multiple Models.

Step - 08: Calculating the Mean Squared Error, Mean Absolute Error and R2 score.

Step - 09: Selecting the Best Model.

Step - 10: Deploying the Notebook in LinuxOne.
## CODE:
### Importing the Required Libraries to handle the dataset:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```
### Importing the Dataset:
```
df = pd.read_csv("yield_df.csv")
df.head()
```
### Data Preprocessing:
```
df.drop('Unnamed: 0',axis=1,inplace=True)
df.head()
df.shape
df.isnull().sum()
df.info()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.duplicated().sum()
df.describe()
```
### Transforming average_rain_fall_mm_per_year:
```
df['average_rain_fall_mm_per_year']
def isStr(obj):
    try:
        float(obj)
        return False
    except:
        return True
to_drop = df[df['average_rain_fall_mm_per_year'].apply(isStr)].index
df = df.drop(to_drop)
df
df['average_rain_fall_mm_per_year'] = df['average_rain_fall_mm_per_year'].astype(np.float64)
```
### Graph Frequency vs Area:
```
plt.figure(figsize=(10,20))
sns.countplot(y=df['Area'])
```
### Yield Per Country Graph:
```
country = (df['Area'].unique())
yield_per_country = []
for state in country:
    yield_per_country.append(df[df['Area']==state]['hg/ha_yield'].sum())
yield_per_country
plt.figure(figsize=(10,20))
sns.barplot(y = country,x = yield_per_country)
```
### Yield of Items:
```
sns.countplot(y=df['Item'])
```
### Visualization:
```
sns.kdeplot(np.array(df['hg/ha_yield']),color='blue',fill=True)
x= np.array(df['Area'])
plt.hist(x)
plt.show()
sns.boxplot(x='hg/ha_yield',data=df)
sns.heatmap(df.corr(),annot=True, vmin=-1, vmax=1)
```
### Identifying the Unique Crops:
```
crops = (df['Item'].unique())
len(crops)
yield_per_item = []
yield_per_crop = []
for crop in crops:
    yield_per_crop.append(df[df['Item']==crop]['hg/ha_yield'].sum())
yield_per_crop
sns.barplot(y=crops,x = yield_per_crop)
```
### Splitting the Data:
### Identifying the Columns that are required to predict the Yield:
```
df
col = ['Year','average_rain_fall_mm_per_year','pesticides_tonnes','avg_temp','Area','Item','hg/ha_yield']
df = df[col]
df
X = df.drop('hg/ha_yield',axis=1)
y = df['hg/ha_yield']
```
### Using Train Test Split to Split the Data:
```
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state=100)
X_train.shape
X_test.shape
X_train
```
### Converting Categorical values to Numerical Values using One Hot Encoding:
```
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
one_hot_encode = OneHotEncoder(drop='first')
scaler = StandardScaler()
X_train.head(1)
preprocesssor = ColumnTransformer(transformers=[('onehotencode',one_hot_encode,[4,5]),('standrization',scaler,[0,1,2,3])],remainder='passthrough')
X_train_dummy = preprocesssor.fit_transform(X_train)
X_test_dummy = preprocesssor.fit_transform(X_test)
X_train_dummy
```
### Model Training:
```
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
models = {
    'Linear Regressor':LinearRegression(),
    'Lasso Regressor':Lasso(),
    'Ridge Regressor':Ridge(),
    'KNeighbors Regressor':KNeighborsRegressor(),
    'Decision Tree Regressor':DecisionTreeRegressor(),
    'Random Forest regressor': RandomForestRegressor(),
    'Gradient Boost Regressor': GradientBoostingRegressor(),
    'Ada Boost Regressor' : AdaBoostRegressor()
}

for name,mod in models.items():
    mod.fit(X_train_dummy,y_train)
    y_pred = mod.predict(X_test_dummy)

    print(f"{name} MSE : {mean_squared_error(y_test,y_pred)} MeanAbsoulte Error: {mean_absolute_error(y_test,y_pred)}  R2 Score: {r2_score(y_test,y_pred)}")
```
### Using a Function to Predict the Yield:
```
rf = RandomForestRegressor()
rf.fit(X_train_dummy,y_train)
y_predict=rf.predict(X_test_dummy)
df.head(1)
def Prediction(Year,average_rain_fall_m_per_year,pesticides_tonnes,avg_temp,Area,Item):
    feature = np.array([[Year,average_rain_fall_m_per_year,pesticides_tonnes,avg_temp,Area,Item]])
    transformed_features = preprocesssor.transform(feature)
    predicted_value = rf.predict(transformed_features).reshape(1,-1)
    return predicted_value[0]
Year = 1990
average_rain_fall_m_per_year = 1485.0
pesticides_tonnes = 121.0
avg_temp = 16.37
Area = 'Albania'
Item = 'Maize'
result = Prediction(Year,average_rain_fall_m_per_year,pesticides_tonnes,avg_temp,Area,Item)
y_test
```
## OUTPUT:

![WhatsApp Image 2023-10-21 at 10 36 38 PM](https://github.com/Aishwarya-TM/Predictive-Agriculture-IBM-Datathon/assets/127846109/294fd787-70a7-48d7-bc13-237ad520c533)

![WhatsApp Image 2023-10-22 at 8 09 21 AM](https://github.com/Aishwarya-TM/Predictive-Agriculture-IBM-Datathon/assets/127846109/55fed315-d846-4c28-a4cb-fe7909a076c1)

![WhatsApp Image 2023-10-21 at 10 39 58 PM](https://github.com/Aishwarya-TM/Predictive-Agriculture-IBM-Datathon/assets/127846109/53130248-8371-4192-a986-4cd602a4567b)

![WhatsApp Image 2023-10-22 at 8 11 03 AM](https://github.com/Aishwarya-TM/Predictive-Agriculture-IBM-Datathon/assets/127846109/d4f6369d-e274-4c88-abf7-ea5ca5d9661e)

![WhatsApp Image 2023-10-22 at 8 10 52 AM](https://github.com/Aishwarya-TM/Predictive-Agriculture-IBM-Datathon/assets/127846109/acb4cc54-3e7b-464d-98f0-86fca378778c)

![WhatsApp Image 2023-10-22 at 8 10 41 AM](https://github.com/Aishwarya-TM/Predictive-Agriculture-IBM-Datathon/assets/127846109/b2c8d2f1-823e-4765-9893-4e601ce0d9dd)

![WhatsApp Image 2023-10-22 at 8 10 28 AM](https://github.com/Aishwarya-TM/Predictive-Agriculture-IBM-Datathon/assets/127846109/16ebeaa3-8d92-4be5-80eb-b1d772c27cc5)

![WhatsApp Image 2023-10-22 at 7 27 10 AM](https://github.com/Aishwarya-TM/Predictive-Agriculture-IBM-Datathon/assets/127846109/cfd5b8ec-99ca-4ffd-b07a-e73753817fe0)

## RESULT:
Thus by using The Random Forest model and K-Neighnors model, our project has achieved the best performance for crop yield prediction with coefficient of determination (R²) of 0.93 and 0.98.
