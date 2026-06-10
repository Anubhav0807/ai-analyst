# Problem Statement: Collect data on study hours and marks from 15 students.
# Build a model to predict marks based on study hours. 

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data1.csv")
print(df)

x = df[["Study_Hours"]]
y = df["Marks"]

model = LinearRegression()
model.fit(x, y)

study_hours = float(input("Enter the hours studied: "))
prediction = model.predict(pd.DataFrame({"Study_Hours": [study_hours]}))
print("Marks predicted:", prediction[0])
