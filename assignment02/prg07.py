import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

df = pd.read_csv("transactions.csv")

transactions = []

for items in df["Items"]:
    transactions.append(items.split(","))

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(te_array, columns=te.columns_)

frequent_patterns = fpgrowth(
    basket,
    min_support=0.3,
    use_colnames=True
)

print("Frequent Patterns:")
print(frequent_patterns)

largest_pattern = frequent_patterns.loc[
    frequent_patterns['support'].idxmax()
]

print("\nMost Common Item Combination:")
print(largest_pattern)
