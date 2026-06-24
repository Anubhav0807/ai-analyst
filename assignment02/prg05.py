import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

df = pd.read_csv("people.csv")

X = df[['Age', 'Income']]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
df['KMeans'] = kmeans.fit_predict(X)

hierarchical = AgglomerativeClustering(n_clusters=2)
df['Hierarchical'] = hierarchical.fit_predict(X)

dbscan = DBSCAN(eps=10000, min_samples=2)
df['DBSCAN'] = dbscan.fit_predict(X)

print(df)
