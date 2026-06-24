import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("employee.csv")

X = df[
    [
        "Age",
        "Salary",
        "Experience",
        "TrainingHours",
        "PerformanceScore"
    ]
]

print("Original Shape:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

total_variance = pca.explained_variance_ratio_.sum()

print("\nTotal Variance Retained:")
print(total_variance)

reduced_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

print("\nReduced Dataset:")
print(reduced_df)
