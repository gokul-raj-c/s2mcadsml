import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("D:\DSML Lab\datasets-main\social-network-ads.csv")
print(df.head())

print(df.isnull().sum())

"""
df = df.fillna(0)
df['age'] = df['age'].fillna(df['age'].mean())
df['salary'] = df['salary'].fillna(df['salary'].median())
df['city'] = df['city'].fillna(df['city'].mode()[0])
"""

X=df.drop(["User ID","Purchased"],axis=1)
y=df["Purchased"]

X=pd.get_dummies(X)

ob=StandardScaler()
scaled=ob.fit_transform(X)
X_scaled=pd.DataFrame(scaled)
X_scaled.columns=X.columns

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

"""# Print the shape of the train and test sets. 
print("shape of x_train",X_train.shape) """

model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

ac=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)

print("accuracy score: ",ac)
print("confusion matrix: ",cm)
print("classification report: ",cr)
