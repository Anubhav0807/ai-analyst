# Problem Statement: Collect data on daily water intake and classify people
# as Healthy or Needs Improvement. Predict the category for a new person.

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data2.csv")
print(df)

le = LabelEncoder()
df["Health_Status"] = le.fit_transform(df["Health_Status"])

x = df[["Water_Intake"]]
y = df["Health_Status"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)

exercise_hours = float(input("Enter today's water intake in liters: "))
prediction = model.predict(pd.DataFrame({"Water_Intake": [exercise_hours]}))

result = le.inverse_transform(prediction)
print("Predicted health status:", result[0])
