import pandas as pd
from mlxtend.frequent_patterns import apriori
df= pd.read_csv('Week 11/Week11_basket_analysis.csv')   
print(df.info())

min_support = 0.2
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
print(frequent_itemsets)