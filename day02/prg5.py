import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
data = {
  "Hours": [1, 2, 3, 4, 5],
  "Marks": [20, 40, 50, 70, 90]
}

df = pd.DataFrame(data)

# Input and Output
x = df[["Hours"]]
y = df["Marks"]

# Create model
model = LinearRegression()

# Train model
model.fit(x, y)

# Predict
prediction = model.predict(pd.DataFrame([[6]], columns=["Hours"]))

print("Predicted Marks:", prediction)

# Graph
plt.scatter(df["Hours"], df["Marks"])
plt.plot(df["Hours"], model.predict(x))
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.show()
