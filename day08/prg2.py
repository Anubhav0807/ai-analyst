import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

data = {
  "Age": [20, 22, 23, 35, 40, 42, 43, 45, 70],
  "Income": [15, 18, 20, 22, 50, 52, 55, 58, 100]
}

df = pd.DataFrame(data)
model = DBSCAN(eps=8, min_samples=3)
df["Cluster"] = model.fit_predict(df)
print(df)

plt.scatter(
  df["Age"],
  df["Income"],
  c=df["Cluster"]
)

plt.xlabel("Age")
plt.ylabel("Income")
plt.title("DBSCAN Clustering")
plt.show()
