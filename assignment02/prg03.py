import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("people.csv")

X = df[['Age', 'Income']]

Z = linkage(X, method='ward')

plt.figure(figsize=(8, 5))
dendrogram(Z, labels=df['PersonID'].values)
plt.title("Dendrogram (Ward Linkage)")
plt.xlabel("Person ID")
plt.ylabel("Distance")
plt.show()
