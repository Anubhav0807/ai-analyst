# Problem Statement: Collect data on study hours and attendance percentage.
# Classify students as Pass or Fail.

import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv("data1.csv")
print(df)

x = df[["Study_Hours", "Attendance_Percentage"]]
y = df["Result"]

model = SVC(kernel="linear")
model.fit(x, y)

study_hours = float(input("Enter the hours studied: "))
attendance_percentage = float(input("Enter the attendance percentage: "))

prediction = model.predict(pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance_Percentage": [attendance_percentage]
}))

print("Prediction:", prediction[0])
