# Problem Statement: Collect short messages from classmates and label them as Academic or Personal.
# Build a model to classify new messages.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("data1.csv")
print(df)

x = df["Message"]
y = df["Category"]

vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x)

model = MultinomialNB()
model.fit(x, y)

message = input("Enter a message: ")
message = vectorizer.transform([message])

prediction = model.predict(message)
print("Predicted Category:", prediction[0])
