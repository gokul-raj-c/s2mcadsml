from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

data=[
    "Apple launched a new iPhone with better neural engine.",  # tech
    "The stock market saw huge gains after the quarterly report.", # finance
    "Google's machine learning model achieved 90% accuracy.",  # tech
    "Investors are worried about rising interest rates and inflation.", # finance
    "Python libraries like scikit-learn are great for ML.", # tech
    "Bonds and treasury yields are highly volatile this week." # finance
]

# 0 tech and 1 finance
labels=[0,1,0,1,0,1]
target_names=['tech','finance']

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print("Total Data Points:", len(data))
print("Training Data Points:", len(X_train))
print("Testing Data Points:", len(X_test))

vectorizer=TfidfVectorizer(stop_words="english")

x_train_vectors=vectorizer.fit_transform(X_train)
x_test_vectors=vectorizer.transform(X_test)

model=SVC(kernel="linear",random_state=42)
model.fit(x_train_vectors,y_train)

y_pred=model.predict(x_test_vectors)
ac=accuracy_score(y_test,y_pred)
print("accuracy: ",round(ac*100,2))

text="OpenAI’s new model delivers more natural and context-aware responses."
text_vectors=vectorizer.transform([text])
predcited=model.predict(text_vectors)
print(f"predcited value: {target_names[predcited[0]]}")


if predcited[0] == 0:
    print("\nPrediction: Tech")
else:
    print("\nPrediction: Finance")