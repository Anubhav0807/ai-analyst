# Problem Statement: Collect data on mobile usage hours and screen time from 15 people.
# Predict daily screen time based on mobile usage.

import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data2.csv")
print(df)

x = df[["Mobile_Usage_Hours"]]
y = df["Screen_Time_Hours"]

model = LinearRegression()
model.fit(x, y)

mobile_usage_hours = float(input("Enter the mobile usage hours: "))
prediction = model.predict(pd.DataFrame({"Mobile_Usage_Hours": [mobile_usage_hours]}))
print("Screen time hours predicted:", prediction[0])
