#import the modules 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz 
from sklearn.model_selection import train_test_split 

#load dataset
df=pd.read_csv('https://raw.githubusercontent.com/gokul-raj-c/datasets/refs/heads/main/IRIS.csv')
df.head()
#Display the number of rows & columns in dataframe 
print(df.describe()) 
print(df.isnull().sum())
#Perform Train Test Split
x=df[['sepal_length','sepal_width','petal_length','petal_width']] 
y=df['species'] 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42) 

#Construct decission tree classifier with criterion='entropy' with min_samples_split to 50. Default value is 2
dt_model=DecisionTreeClassifier(criterion='entropy',min_samples_split=50,min_samples_leaf=3) 
dt_model.fit(x_train,y_train)

#Display Accuracy on test data
y_predict=dt_model.predict(x_test) 
cm=confusion_matrix(y_test,y_predict) 
plt.figure(figsize=(8,5)) 
sns.heatmap(cm,annot=True,cmap='Blues') 
plt.xlabel('predicted Label') 
plt.ylabel('True label') 
plt.title('CONFUSION MATRIX') 
plt.show() 

