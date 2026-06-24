import pandas as pd
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("people.csv")
x = df[['Age', 'Income']]

hc = AgglomerativeClustering(n_clusters=2)

df['Cluster'] = hc.fit_predict(x)
print(df[['PersonID', 'Age', 'Income', 'Cluster']])
