import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("data1.csv")
model = AgglomerativeClustering(n_clusters=2, linkage="ward")
df["Cluster"] = model.fit_predict(df)
print(df)

plt.scatter(
  df["Age"],
  df["Income"],
  c=df["Cluster"]
)

plt.xlabel("Age")
plt.ylabel("Income")
plt.title("Agglomerative Clustering")
plt.show()
