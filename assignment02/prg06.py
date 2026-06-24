import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("transactions.csv")

transactions = []

for items in df["Items"]:
    transactions.append(items.split(","))

te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

basket = pd.DataFrame(te_array, columns=te.columns_)

frequent_itemsets = apriori(
    basket,
    min_support=0.3,
    use_colnames=True
)

print("Frequent Itemsets:")
print(frequent_itemsets)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.1
)

print("\nAssociation Rules:")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

best_rule = rules.loc[rules["confidence"].idxmax()]

print("\nRule with Highest Confidence:")
print(best_rule[["antecedents", "consequents", "support", "confidence", "lift"]])
