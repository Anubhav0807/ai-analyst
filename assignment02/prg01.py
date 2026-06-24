import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("customers.csv")
x = df[['Age', 'SpendingScore']]

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)

df['Cluster'] = kmeans.fit_predict(x)
print(df[['CustomerID', 'Age', 'SpendingScore', 'Cluster']])
