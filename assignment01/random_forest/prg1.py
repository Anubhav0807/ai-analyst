# Problem Statement: Collect data on study hours, attendance, assignment score,
# and pass/fail status. Use Random Forest to predict pass/fail.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data1.csv")
print(df)

x = df[["Study_Hours", "Attendance_Percentage", "Assignment_Score"]]
y = df["Result"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x, y)

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
assignment_score = float(input("Enter assignment score: "))

prediction = model.predict(pd.DataFrame({
    "Study_Hours": [study_hours],
    "Attendance_Percentage": [attendance],
    "Assignment_Score": [assignment_score]
}))

print("Prediction:", prediction[0])
