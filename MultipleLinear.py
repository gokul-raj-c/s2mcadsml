# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
df=pd.read_csv("Housing_Price.csv")
# Print first five rows using head() function

print(df.head())
# Check if there are any null values. If any column has null values, treat them accordingly

print(df.isnull().sum())
# Replace all non-numeric values with numeric values.

df.replace(to_replace="yes", value=1, inplace=True)
df.replace(to_replace="no", value=0, inplace=True)
df.replace(to_replace="unfurnished", value=0, inplace=True)
df.replace(to_replace="semi-furnished", value=1, inplace=True)
df.replace(to_replace="furnished", value=2, inplace=True)
print()
print(df.head())
# Split the DataFrame into the training and test sets.

from sklearn.model_selection import train_test_split
x=df.drop('price',axis=1)
y=df['price']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# Build linear regression model

from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
# Print the value of the intercept
print("model intercept:",model.intercept_)

# Print the names of the features along with the values of their corresponding coefficients.

coef = model.coef_
f=x.columns
d=pd.DataFrame({'features':f,'regcoef':coef})
print(d)
# Predict the target variable values for training and test set

y_predicted=model.predict(x_test)

# Evaluate the linear regression model using the 'r2_score','mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
print("testing data")
y_predicted=model.predict(x_test)
print("mean squared error:",mean_squared_error(y_test,y_predicted))
print("mean absolute error:",mean_absolute_error(y_test,y_predicted))
print("r2 score:",r2_score(y_test,y_predicted))
print("training data")
y_predicted1=model.predict(x_train)
print("mean squared error:",mean_squared_error(y_train,y_predicted1))
print("mean absolute error:",mean_absolute_error(y_train,y_predicted1))
print("r2 score:",r2_score(y_train,y_predicted1))