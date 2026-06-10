# Problem Statement: Collect data on attendance, assignment submission, and marks.
# Build a decision tree to determine whether a student will pass or fail.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data1.csv")
print(df)

x = df[["Attendance_Percentage", "Assignment_Submitted", "Marks"]]
y = df["Result"]

model = DecisionTreeClassifier()
model.fit(x, y)

attendance = float(input("Enter attendance percentage: "))
assignment = ("no", "yes").index(input("Assignment submitted? ").lower())
marks = float(input("Enter marks: "))

prediction = model.predict(pd.DataFrame({
    "Attendance_Percentage": [attendance],
    "Assignment_Submitted": [assignment],
    "Marks": [marks]
}))

print("Prediction:", prediction[0])
