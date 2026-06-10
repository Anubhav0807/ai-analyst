# Problem Statement: Collect data on budget, battery preference, and camera preference.
# Predict whether a person will buy a particular smartphone.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("data2.csv")
print(df)

x = df[["Budget_Thousands", "Battery_Preference", "Camera_Preference"]]
y = df["Buy"]

model = DecisionTreeClassifier()
model.fit(x, y)

budget_thousands = float(input("Enter your budget (in thousands): "))
battery_preferrence = ("no", "yes").index(input("Bettery preference: ").lower())
camera_preferrence = ("no", "yes").index(input("Camera preference: ").lower())

prediction = model.predict(pd.DataFrame({
    "Budget_Thousands": [budget_thousands],
    "Battery_Preference": [battery_preferrence],
    "Camera_Preference": [camera_preferrence]
}))

print("Prediction:", prediction[0])
