#SIMPLE PROGRAM
#import libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# Define the small dataset
# We will classify texts as tech(0) or finance(1)
data = [
    "Apple launched a new iPhone with better neural engine.",  # tech
    "The stock market saw huge gains after the quarterly report.", # finance
    "Google's machine learning model achieved 90% accuracy.",  # tech
    "Investors are worried about rising interest rates and inflation.", # finance
    "Python libraries like scikit-learn are great for ML.", # tech
    "Bonds and treasury yields are highly volatile this week." # finance
]
#0 for tech 1 for finance
labels=[0,1,0,1,0,1]
target_names=['tech','finance']
# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print("Total Data Points:", len(data))
print("Training Data Points:", len(X_train))
print("Testing Data Points:", len(X_test))
# 3. Feature Extraction (TF-IDF)
vectorizer=TfidfVectorizer(stop_words="english")
# Convert training data
X_train_vectors=vectorizer.fit_transform(X_train)
# Convert testing data
X_test_vectors=vectorizer.transform(X_test)
# 4. Initialize and Train the SVM
svm_classifier=SVC(kernel="linear",random_state=42)
svm_classifier.fit(X_train_vectors,y_train)
# 5. Predict and Evaluate
y_pred=svm_classifier.predict(X_test_vectors)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
# 6. Simple Prediction
text = "OpenAI’s new model delivers more natural and context-aware responses."
text_vectors = vectorizer.transform([text])
prediction = svm_classifier.predict(text_vectors)
print(f'Prediction: {target_names[prediction[0]]}')