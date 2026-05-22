import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# data = {
#   "Hours": [1, 2, 3, 4, 5, 6],
#   "Result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]
# }

data = pd.read_csv("data.csv")

df = pd.DataFrame(data)
print(df)
print()

le = LabelEncoder()
df["Result"] = le.fit_transform(df["Result"])
print(df)
print()

x = df[["Hours"]] # Input
y = df["Result"]  # Output

model = LinearRegression()
model.fit(x, y)

hours = float(input("Enter the no. of hours studied: "))
result = model.predict(pd.DataFrame([[hours]], columns=["Hours"]))
print("Prediction:", result)
