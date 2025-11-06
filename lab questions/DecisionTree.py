import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
#from sklearn.datasets import load_iris

df=pd.read_csv("D:\DSML Lab\datasets-main\IRIS.csv")
print(df.head())

print(df.isnull().sum())

X=df.drop(["species"],axis=1)
y=df["species"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

model=DecisionTreeClassifier(criterion="entropy",min_samples_leaf=3,min_samples_split=50)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

ac=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred) 

print("accuracy score: ",ac)
print("confusion matrix: ",cm)
print("classification report: ",cr)

import matplotlib.pyplot as plt
import seaborn as sb

plt.figure(figsize=(10,5))
plt.title("confusion matrix")
sb.heatmap(cm,annot=True,cmap="Blues")
plt.xlabel('predicted Label')  
plt.ylabel('True label')  
plt.show()

plt.figure(figsize=(10,10)) 
plot_tree(model,filled=True) 
plt.show()