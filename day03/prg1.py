import pandas as pd
from sklearn.linear_model import LogisticRegression

# data = {
#   "Hours": [1, 2, 3, 4, 5, 6],
#   "Result": ["Fail", "Fail", "Fail", "Pass", "Pass", "Pass"]
# }

data = pd.read_csv("data.csv")

df = pd.DataFrame(data)
print(df)

x = df[["Hours"]] # Input
y = df["Result"]  # Output

model = LogisticRegression()
model.fit(x, y)

hours = float(input("Enter the no. of hours studied: "))
result = model.predict(pd.DataFrame([[hours]], columns=["Hours"]))
print("Prediction:", result)
