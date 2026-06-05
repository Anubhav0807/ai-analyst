import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = {
  "Milk": [1, 1, 0, 1, 1],
  "Bread": [1, 1, 1, 0, 1],
  "Butter": [0, 1, 1, 1, 1],
  "Egg": [1, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

frequent_items = apriori(df, min_support=0.4, use_colnames=True)
rules = association_rules(frequent_items, metric="confidence", min_threshold=0.7)

print(frequent_items)
print(rules[["antecedents", "support", "confidence", "lift"]])
