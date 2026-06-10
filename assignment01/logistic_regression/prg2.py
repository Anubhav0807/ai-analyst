# Problem Statement: Collect data on attendance percentage and assignment completion status.
# Predict whether a student is eligible for placement (Yes/No).

import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data2.csv")
print(df)

x = df[["Attendance_Percentage", "Assignments_Completed"]]
y = df["Eligible"]

model = LogisticRegression()
model.fit(x, y)

attendance_percentage = float(input("Enter the attendance_pencentage: "))
assignments_completed = float(input("Enter the assignments completed: "))

prediction = model.predict(pd.DataFrame({
    "Attendance_Percentage": [attendance_percentage],
    "Assignments_Completed": [assignments_completed]
}))

result = ("No", "Yes")[prediction[0]]
print("Predicted Eligibility:", result)
