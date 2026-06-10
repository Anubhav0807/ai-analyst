import pandas as pd
from sklearn.decomposition import PCA

data = {
    "Age": [25, 30, 35, 40, 45],
    "Salary": [30000, 40000, 50000, 60000, 70000],
    "Experience": [2, 5, 8, 10, 12]
}

df = pd.DataFrame(data)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df)
print(reduced_data)
