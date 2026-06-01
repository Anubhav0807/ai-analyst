import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
  "Age": [18, 20, 22, 25, 30, 35, 40, 45, 50, 55],
  "Spending_Score": [90, 85, 88, 80, 60, 58, 40, 35, 20, 15]
}

df = pd.DataFrame(data)
# print(df.head())

x = df[["Age", "Spending_Score"]]
model = KMeans(n_clusters=3, random_state=42)
model.fit(x)
df["Cluster"] = model.labels_
print(df)
plt.scatter(df["Age"], df["Spending_Score"], df["Cluster"])
plt.scatter(
  model.cluster_centers_[:,0],
  model.cluster_centers_[:,1],
  s=200, # size of cluster point
  color="red",
  marker='x'
)
plt.xlabel("Age")
plt.ylabel("Spending_Score")
plt.show()
