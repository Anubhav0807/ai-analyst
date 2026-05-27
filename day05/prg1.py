import pandas as pd
from sklearn.svm import SVC

data = {
  "Hours": [1, 2, 3, 4, 5, 6],
  "Result": [0, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
print(df)

x = df[["Hours"]]
y = df["Result"]

model = SVC(kernel="linear")
model.fit(x, y)

prediction = model.predict(pd.DataFrame([[3.5]], columns=["Hours"]))
result = "Pass" if prediction[0] else "Fail"
print("Prediction:", result)