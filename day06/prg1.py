from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
print(iris.feature_names)
print(iris.target_names)

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
print(df)
