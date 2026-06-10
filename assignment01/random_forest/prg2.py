# Problem Statement: Collect data on sleep hours, exercise hours, and stress level.
# Predict whether a person has a healthy lifestyle.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data2.csv")
print(df)

x = df[["Sleep_Hours", "Exercise_Hours_Per_Week", "Stress_Level"]]
y = df["Healthy_Lifestyle"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x, y)

sleep_hours = float(input("Enter sleep hours: "))
exercise_hours = float(input("Enter exercise hours per week: "))
stress_level = int(input("Enter stress level (1-10): "))

prediction = model.predict(pd.DataFrame({
    "Sleep_Hours": [sleep_hours],
    "Exercise_Hours_Per_Week": [exercise_hours],
    "Stress_Level": [stress_level]
}))

print("Prediction:", prediction[0])
