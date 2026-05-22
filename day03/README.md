# Contents
- Classification
- Naive Bayes
- Decision Tree
- K-Nearest Neighbours
- Regression vs Classification
- Hands-On Lab

## Classification
- Binary Classification
  - Examples:
    - Spam or Not Spam
    - Sick or Healthy
  - Algorithms:
    - Logistic Regression
    - SVM
    - Decision Tree
- Multi-class Classification
  - Examples:
    - Apple/Orange/Bannana
    - Engligh/Hindi/Tamil
  - Algorithms:
    - Naive Bayes
    - Random Forest
    - KNN

### Classification vs Regression
- Key rule:
  - Discrete output -> Classification
  - Continous output -> Regression

## Naive Bayes Classifier
`P(A|B) = P(B|A) * P(A) / P(B)`

- Posterior Probability
  - `P(A|B)`
  - Probability of A being true given that B is true
- Likelihood
  - `P(B|A)`
  - Probability of seeing B given that A is true
- Prior Probability
  - `P(A)`
  - Initial probability of A before any evidence
- Marginal Likelihood
  - `P(B)`
  - Total probability of B across all casses