import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("fruits.csv")
print(df)

le_fruit = LabelEncoder()
le_color = LabelEncoder()
le_size = LabelEncoder()
le_taste = LabelEncoder()

df["Fruit"] = le_fruit.fit_transform(df["Fruit"])
df["Color"] = le_color.fit_transform(df["Color"])
df["Size"] = le_size.fit_transform(df["Size"])
df["Taste"] = le_taste.fit_transform(df["Taste"])
# print(df)

x = df[["Color", "Size", "Taste"]]
y = df["Fruit"]

model = GaussianNB()
model.fit(x, y)

color = input("Enter the color of fruit: ")
size = input("Enter the size of fruit: ")
taste = input("Enter the taste of fruit: ")

sample = pd.DataFrame({
  "Color": [le_color.transform([color])[0]],
  "Size": [le_size.transform([size])[0]],
  "Taste": [le_taste.transform([taste])[0]]
})

prediction = model.predict(sample)
fruit = le_fruit.inverse_transform(prediction)
print("Predicted fruit is:", fruit[0])