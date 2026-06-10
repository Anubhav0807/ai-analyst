# Problem Statement: Collect data on hours spent exercising per week
# and categorize people as Fit or Not Fit. Predict the category for a new person.

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data1.csv")
print(df)

le = LabelEncoder()
df["Fitness_Status"] = le.fit_transform(df["Fitness_Status"])

x = df[["Exercise_Hours"]]
y = df["Fitness_Status"]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)

exercise_hours = float(input("Enter exercise hours: "))
prediction = model.predict(pd.DataFrame({"Exercise_Hours": [exercise_hours]}))

result = le.inverse_transform(prediction)
print("Predicted fitness status:", result[0])
