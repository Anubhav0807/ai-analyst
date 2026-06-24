import pandas as pd
from sklearn.cluster import DBSCAN

df = pd.DataFrame({
    'X': [1, 1, 2, 2, 8, 8, 9, 9, 50, 55],
    'Y': [1, 2, 1, 2, 1, 2, 1, 2, 100, 100]
})

dbscan = DBSCAN(eps=3, min_samples=2)

df['Cluster'] = dbscan.fit_predict(df[['X', 'Y']])

print(df)

outliers = df[df['Cluster'] == -1]

print("\nOutliers:")
print(outliers)
