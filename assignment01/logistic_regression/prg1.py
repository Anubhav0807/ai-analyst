# Problem Statement: Collect data on study hours and pass/fail status from students.
# Predict whether a student will pass or fail. 

import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data1.csv")
print(df)

x = df[["Study_Hours"]]
y = df["Result"]

model = LogisticRegression()
model.fit(x, y)

study_hours = float(input("Enter the hours studied: "))
prediction = model.predict(pd.DataFrame({"Study_Hours": [study_hours]}))
result = ("Fail", "Pass")[prediction[0]]
print("Predicted Result:", result)
