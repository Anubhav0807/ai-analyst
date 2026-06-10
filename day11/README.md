# Dimensionality Reduction
A dimension is a feature (column) in a dataset.
Dimensionality Reduction converts many features into fewer features WITHOUT losing imp info.

## The curse of dimensionality
As the number of features (dimensions) grows, the amount of data needed to train a good model grows exponentially.

## Type of Dimensionality Reduction
- Feature Selection - **Select** the most useful existing features and **remove** the rest.
- Feature Extraction - **Create** new features by **combining** or **compressing** the original ones.

## PCA - Principal Component Analysis
1. Standardise Data
    - Convert all feature values to the same scale (mean=0, std=1)
    - Without this, feature with larger values dominate
2. Find Relationships
    - Compute the covariance matrix
    - Highly coorelated features carry redundant info
3. Find Principal Components
    - Compute eigenvectors and eigenvalues of the covariance matrix
    - PC1 captures the mos tvariance, PC2 the next and so on
4. Keep Imp Components
    - Select the top N components that explain most of the variance
    - Typically keep enough to explain 95% of total variance

### Advantages
- Faster Training
- Less Memory Usage
- Removes Noise
- Reduces Overfiting
- Better Visualisation

### Disadvantages
- Info Loss
- Harder to Interpret
- Difficult Interpretation
- Assumes Linearity
- Scaling Required