import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

transaction = [
  ["Milk", "Bread", "Butter"],
  ["Bread", "Butter"],
  ["Milk", "Butter"],
  ["Milk", "Bread"],
  ["Milk", "Bread", "Butter"]
]

te = TransactionEncoder()
te_array = te.fit(transaction).transform(transaction)

df = pd.DataFrame(te_array, columns=te.columns_)

frequent_items = fpgrowth(df, min_support=0.4, use_colnames=True)
print(frequent_items)
