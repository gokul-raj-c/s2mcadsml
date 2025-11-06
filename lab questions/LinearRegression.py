import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,r2_score,mean_absolute_error,mean_squared_error,root_mean_squared_error

df=pd.read_csv("D:\DSML Lab\datasets-main\insurance_dataset.csv")
print(df.head())

print(df.isnull().sum())

import seaborn as sb
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.title("insurance")
sb.regplot(x="age",y="charges",data=df,color="green",line_kws={"color":"red"})
plt.xlabel("age")
plt.ylabel("charges")
plt.grid()
plt.show()

X=df[["age"]]
y=df["charges"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train=np.reshape(X_train,(-1,1))
X_test=np.reshape(X_test,(-1,1))

model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

print("model coefficient: ",model.coef_[0])
print("model intercept: ",model.intercept_)

res=r2_score(y_test,y_pred)
print("Accuracy: ",round(res*100,2))
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=root_mean_squared_error(y_test,y_pred)

print("Mean absolute error: ",mae)
print("mean squared error: ",mse)
print("root mean squared error: ",rmse)