# Problem Statement: Using the movie review sentiment classification model,
# evaluate predictions using a confusion matrix and calculate accuracy.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("data2.csv")
x = df["Review"]
y = df["Sentiment"]

vectorizer = CountVectorizer()
x_vectorized = vectorizer.fit_transform(x)

x_train, X_test, y_train, y_test = train_test_split(
    x_vectorized,
    y,
    test_size=0.3,
    random_state=42
)

model = MultinomialNB()
model.fit(x_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
).plot()

plt.title("Movie Review Sentiment Confusion Matrix")
plt.show()
