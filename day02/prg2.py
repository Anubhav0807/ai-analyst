import pandas as pd

data = {
  "Name": ["Arun", "Priya", "Rahul"],
  "Marks": [85, 90, 78]
}

df = pd.DataFrame(data)
# df = pd.read_csv("file_name.csv")
# df = pd.read_excel("file_name.xlxs")
print(df)
