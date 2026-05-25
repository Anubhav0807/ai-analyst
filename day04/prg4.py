import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data4.csv")
print(df)

le_attendance = LabelEncoder()
le_result = LabelEncoder()
df["Attendance"] = le_attendance.fit_transform(df["Attendance"])
df["Result"] = le_result.fit_transform(df["Result"])
# print(df)

x = df[["Study_Hours", "Attendance"]]
y = df["Result"]

model = DecisionTreeClassifier()
model.fit(x, y)

hours = input("Enter hours studied: ")
attnedance = input("Enter attendance [Low/Medium/High]: ")
sample = pd.DataFrame({
  "Study_Hours": [hours],
  "Attendance": le_attendance.transform([attnedance])
})
prediction = model.predict(sample)
result = le_result.inverse_transform(prediction)
print("Result:", result[0])