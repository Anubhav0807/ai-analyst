import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv("orders.csv")

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

print("Frequent Itemsets")
print(frequent_itemsets)

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.7
)

print("\nAssociation Rules")
print(rules[["antecedents", "consequents", "support", "confidence", "lift"]])

selected_rules = rules[
    (rules["support"] > 0.3) &
    (rules["confidence"] > 0.7)
]

print("\nRules with Support > 0.3 and Confidence > 0.7")
print(selected_rules[["antecedents", "consequents",
                      "support", "confidence", "lift"]])
