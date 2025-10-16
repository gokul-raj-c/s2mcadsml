#import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error,root_mean_squared_error

#loading the datasets
df=pd.read_csv("insurance_dataset (1).csv")
# Print first five rows using head() function
print(df.head(10))
#Check if there are any null values. If any column has null values, treat them accordingly
print(df.isnull().sum())
# Create a regression plot between 'age' and 'charges'
plt.figure(figsize=(10,7))
plt.title("insurance")
sns.regplot(x="age",y="charges",data=df,color="green",line_kws={"color":"red"})
plt.xlabel("sales")
plt.ylabel("charges")
plt.grid()
plt.show()
# Split the DataFrame into the training and test sets
from sklearn.model_selection import train_test_split
X=df[["age"]]
y=df["charges"]
x_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# Create two-dimensional NumPy arrays for the feature and target variables.
x_train=np.reshape(x_train,(-1,1))
X_train=np.reshape(x_train,(-1,1))
X_test=np.reshape(X_test,(-1,1))
y_train=np.reshape(y_train,(-1,1))
# Print the shape or dimensions of these reshaped arrays
print(X_train.shape)
print(y_train.shape)
print(X)
print(y)
#model training
# 2. Deploy linear regression model using the 'sklearn.linear_model' module.
from sklearn.linear_model import LinearRegression
# Create an object of the 'LinearRegression' class.
model=LinearRegression()
# 3. Call the 'fit()' function
model.fit(X_train,y_train)
y_predicted=model.predict(X_test)
# Print the slope and intercept values
print()
print("model coefficient(slope):",model.coef_[0])
print("model intercept:",model.intercept_)
print()
# Predict the target variable values for both training set and test set
y_pred=model.predict(X_test)
# Call 'r2_score', 'mean_squared_error' & 'mean_absolute_error' functions of the 'sklearn' module.
#Calculate RMSE value by taking the
# Print these values for both training set and test set
res=r2_score(y_test,y_pred)
print("accuracy is ",round(res*100,2),'%')
print("Mean squared error is ",mean_squared_error(y_test,y_pred))
print("Root Mean squared error is ",root_mean_squared_error(y_test,y_pred))
print("Mean absolute error is ",mean_absolute_error(y_test,y_pred))