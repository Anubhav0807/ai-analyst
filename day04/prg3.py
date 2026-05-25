import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data3.csv")
print(df)

le_class = LabelEncoder()
df["Class"] = le_class.fit_transform(df["Class"])
# print(df)

x = df[["x1", "x2"]]
y = df[["Class"]]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)

x1 = input("Enter value of x1: ")
x2 = input("Enter value of x2: ")
prediction = model.predict(pd.DataFrame({"x1": [x1], "x2": [x2]}))
output = le_class.inverse_transform(prediction)
print("Class:", output[0])
