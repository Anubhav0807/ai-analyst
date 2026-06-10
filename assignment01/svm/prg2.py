# Problem Statement: Collect data on sleep hours and stress level. 
# Classify people as Low Stress or High Stress.

import pandas as pd
from sklearn.svm import SVC

df = pd.read_csv("data2.csv")
print(df)

x = df[["Sleep_Hours", "Stress_Level"]]
y = df["Category"]

model = SVC(kernel="linear")
model.fit(x, y)

study_hours = float(input("Enter the hours slept: "))
attendance_percentage = float(input("Enter the stress level: "))

prediction = model.predict(pd.DataFrame({
    "Sleep_Hours": [study_hours],
    "Stress_Level": [attendance_percentage]
}))

print("Prediction:", prediction[0])
