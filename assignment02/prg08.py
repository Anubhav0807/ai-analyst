import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("students.csv")

X = df[
    [
        "StudyHours",
        "Attendance",
        "AssignmentsCompleted",
        "InternalMarks",
        "FinalExamMarks"
    ]
]

print("Original Shape:", X.shape)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("Reduced Shape:", X_pca.shape)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PrincipalComponent1", "PrincipalComponent2"]
)

print("\nPCA Transformed Dataset:")
print(pca_df)