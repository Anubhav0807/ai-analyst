# Problem Statement: Collect movie reviews from friends and label them as Positive or Negative.
# Predict the sentiment of a new review. 

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data2.csv")
print(df)

x = df["Review"]
y = df["Sentiment"]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x)

model = MultinomialNB()
model.fit(x, y)

message = input("Enter a movie review: ")
message = vectorizer.transform([message])

prediction = model.predict(message)
print("Predicted Sentiment:", prediction[0])
