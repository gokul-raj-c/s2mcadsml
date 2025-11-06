import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,r2_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_csv("D:\DSML Lab\datasets-main\IRIS.csv")
print(df.head())

#check null values
print(df.isnull().sum())

X = df.drop(['species'], axis=1)
y=df['species']

X=pd.get_dummies(X)

ob=StandardScaler()
scaled=ob.fit_transform(X)
X_scaled=pd.DataFrame(scaled)
X_scaled.columns=X.columns

x_train,x_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=42)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)

y_test_pred=model.predict(x_test)
print("accuracy: ",accuracy_score(y_test_pred,y_test))
print("Classification Report-test set")
print(classification_report(y_test, y_test_pred))