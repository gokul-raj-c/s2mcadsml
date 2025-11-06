import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

df=pd.read_csv("D:\DSML Lab\datasets-main\IRIS.csv")
print(df.head())

print(df.isnull().sum())

X=df.drop(["species"],axis=1)
y=df["species"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

model=GaussianNB()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

ac=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)

print("accuracy score: ",ac)
print("confusion matrix: ",cm)
print("classification report: ",cr)
