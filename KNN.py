# Import all the necessary packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
# Load the dataset
df=pd.read_csv("social-network-ads.csv")

# Print first five rows using head() function
print(df.head())

# Check if there are any null values. If any column has null values, treat them accordingly
print(df.isnull().sum())
# Split the dataset into dependent and independent features
X = df.drop(['User ID', 'Purchased'], axis=1)
y=df['Purchased']

# Use 'get_dummies()' function to convert each categorical column in a DataFrame to numerical.
X=pd.get_dummies(X)
print(X)

ob=StandardScaler()
scaled=ob.fit_transform(X)
X_scaled=pd.DataFrame(scaled)
X_scaled.columns=X.columns
print(X_scaled.head())

# Use 'info()' function with the features DataFrame.
X_scaled.info()

# Split the DataFrame into the train and test sets.
# Perform train-test split using 'train_test_split' function.
X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.3,random_state=42)
# Print the shape of the train and test sets.
print("shape of x_train",X_train.shape)
print("shape of x_test",X_test.shape)
print("shape of y_train",y_train.shape)
print("shape of y_test",y_test.shape)

# Train kNN Classifier model
model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
# Perform prediction using 'predict()' function.
y_train_predict=model.predict(X_train)
y_test_predict=model.predict(X_test)

# Call the 'score()' function to check the accuracy score of the train set and test set.
from sklearn.metrics import accuracy_score
print("training-accuracy",accuracy_score(y_train,y_train_predict))
print("testing-accuracy",accuracy_score(y_test,y_test_predict))
# Display the precision, recall, and f1-score values.
from sklearn.metrics import classification_report
print("Classification Report-train set")
print(classification_report(y_train, y_train_predict))
print()
print("Classification Report-test set")
print(classification_report(y_test, y_test_predict))