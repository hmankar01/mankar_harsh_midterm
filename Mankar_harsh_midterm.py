import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,fpgrowth , association_rules
import random
import time
import csv
from itertools import combinations
from collections import defaultdict
import time

data1 = pd.read_csv("Amazon.csv")
dataframe1 = data1.drop('TransID',axis = 1)
binary_df1 = dataframe1['ItemID'].str.get_dummies(sep=', ')
binary_df1

data2 = pd.read_csv("Groceries.csv")
dataframe2 = data2.drop('TransID',axis = 1)
binary_df2 = dataframe2['ItemID'].str.get_dummies(sep=',')

data3 = pd.read_csv("K-Mart.csv")
dataframe3 = data3.drop('TransID',axis = 1)
binary_df3 = dataframe3['ItemID'].str.get_dummies(sep=', ')

data4 = pd.read_csv("Nike.csv")
dataframe4 = data4.drop('TransID',axis = 1)
binary_df4 =dataframe4['ItemID'].str.get_dummies(sep=', ')

data5 = pd.read_csv("Best Buy.csv")
dataframe5 = data5.drop('TransID',axis = 1)
binary_df5 = dataframe5['ItemID'].str.get_dummies(sep=', ')

dataframes = [binary_df1, binary_df2, binary_df3, binary_df4, binary_df5]

# user input for users to select the dataset
print("Select datasets to analyze (Enter comma-separated indices from 1 to 5):")
print("1. Amazon")
print("2. Groceries")
print("3. K-Mart")
print("4. Nike")
print("5. Best Buy")
selected_indices = input("Enter the indices of store you want to see (like 1,2 ...): ").split(",")

selected_indices = [int(idx.strip()) for idx in selected_indices] # indices to integers

dataframes = [dataframes[idx - 1] for idx in selected_indices] # dataframes based on user input

min_support = float(input("Enter the minimum support(in range 0 to 0.99): "))
min_confidence = float(input("Enter the minimum confidence(in range 0 to 0.99): "))

def fp_growth_algorithm(df):
    start = time.time()
    freq_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    end = time.time()
    total_time = end - start
    return freq_itemsets, rules, total_time

def brute_force_algorithm(df):
    start = time.time()
    item_counts = defaultdict(int)
    item_counts_copy = item_counts.copy()
    items = df.columns
    binary_df = df
    for i in range(1, len(items)):
        item_combinations = combinations(items, i)
        for itemset in item_combinations:
            count = sum(all(binary_df[item][j] == 1 for item in itemset) for j in range(len(binary_df)))
            support = count / len(binary_df)
            if support >= min_support:
                item_counts[itemset] = support
    association_rules = []
    for itemset, support in item_counts.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i): # ANTECEDENT IS ITEMS BEFORE ARROW AND CONSEQUENT MEANING AFTER THE ARROW WHEN WE MAKE ASSOCIATION RULES.
                    antecedent = set(antecedent)
                    consequent = set(itemset) - antecedent
                    if item_counts[tuple(antecedent)] != 0:
                        confidence = support / item_counts[tuple(antecedent)]
                        if confidence >= min_confidence:
                            association_rules.append((antecedent, consequent, confidence, support))
    end = time.time()
    total_time = end - start
    return item_counts, association_rules, total_time

def apriori_algorithm(df):
    start = time.time()
    freq_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(freq_itemsets, metric="confidence", min_threshold=min_confidence)
    end = time.time()
    total_time = end - start
    return freq_itemsets, rules, total_time

for  idx,df in enumerate(dataframes): # converting or making sense from binary to orignal dataset
    if df.equals(binary_df1):
        df_name = 'Amazon'
    elif df.equals(binary_df2):
        df_name = 'Groceries'
    elif df.equals(binary_df3):
        df_name = 'K-Mart'
    elif df.equals(binary_df4):
        df_name = 'Nike'
    else:
        df_name = 'Best Buy'
    print(f"Processing DataFrame {df_name}:")

     # FP-Growth Algorithm
     
    print("FP-Growth Algorithm:")
    freq_itemsets_fp, rules_fp, time_fp = fp_growth_algorithm(df)
    print("Frequent Itemsets:")
    print(freq_itemsets_fp)
    print("Association Rules:")
    print(rules_fp.iloc[:, [0, 1, 4, 5]])
    print(f"Time taken for FP-Growth Algorithm: {time_fp} seconds\n")
    
    #Brute Force 
    print("Brute Force Algorithm:")
    item_counts_brute, association_rules_brute, time_brute = brute_force_algorithm(df)
    print("Item Counts:")
    print(item_counts_brute)
    print("Association Rules:")
    count = 1
    for antecedent, consequent, confidence, support in association_rules_brute:
        print(f"Rule:{count} {antecedent} => {consequent}, Support: {support}, Confidence: {confidence}")
        count+=1
    print(f"Time taken for Brute Force Algorithm: {time_brute} seconds\n")
    
    # Apriori Algorithm
    print("Apriori Algorithm:")
    freq_itemsets_ap, rules_ap, time_ap = apriori_algorithm(df)
    print("Frequent Itemsets:")
    print(freq_itemsets_ap)
    print("Association Rules:")
    print(rules_ap.iloc[:, [0, 1, 2, 3, 4, 5]])
    print(f"Time taken for Apriori Algorithm: {time_ap} seconds\n")

# empty List for all 3 algorithm performances
fp_growth_runtimes = []
brute_force_runtimes = []
apriori_runtimes = []

# Loop over each dataset
for idx, df in enumerate(dataframes):
    if df.equals(binary_df1):
        df_name = 'Amazon'
    elif df.equals(binary_df2):
        df_name = 'Groceries'
    elif df.equals(binary_df3):
        df_name = 'K-Mart'
    elif df.equals(binary_df4):
        df_name = 'Nike'
    else:
        df_name = 'Best Buy'
    print(f"your selected DataFrame {df_name}:")
    
    # FP-Growth Algorithm
    freq_itemsets_fp, rules_fp, time_fp = fp_growth_algorithm(df)
    fp_growth_runtimes.append(time_fp)
    
    # Brute Force Algorithm
    item_counts_brute, association_rules_brute, time_brute = brute_force_algorithm(df)
    brute_force_runtimes.append(time_brute)
    
    # Apriori Algorithm
    freq_itemsets_ap, rules_ap, time_ap = apriori_algorithm(df)
    apriori_runtimes.append(time_ap)

     # Converting association rules to sets for comparison as it was in pandas dataframe.
    fp_rules_set = set(tuple(x) for x in rules_fp[['antecedents', 'consequents']].values)
    brute_force_rules_set = set((tuple(a), tuple(c)) for a, c, _, _ in association_rules_brute)
    apriori_rules_set = set(tuple(x) for x in rules_ap[['antecedents', 'consequents']].values)
    
    # Comparing all three
    fp_brute_match = fp_rules_set  == brute_force_rules_set
    fp_ap_match =  fp_rules_set == brute_force_rules_set
    brute_ap_match =brute_force_rules_set  == brute_force_rules_set
    
     # Print summary for current dataset
    print("Results Comparison:")
    print(f"FP-Growth vs. Brute Force: {'Match' if fp_brute_match else 'Mismatch'}")
    print(f"FP-Growth vs. Apriori: {'Match' if fp_ap_match else 'Mismatch'}")
    print(f"Brute Force vs. Apriori: {'Match' if brute_ap_match else 'Mismatch'}")
    print(f"FP-Growth Runtime: {time_fp} seconds")
    print(f"Brute Force Runtime: {time_brute} seconds")
    print(f"Apriori Runtime: {time_ap} seconds")
    print("\n")

# Calculate average runtimes
avg_fp_growth_runtime = sum(fp_growth_runtimes) / len(fp_growth_runtimes)
avg_brute_force_runtime = sum(brute_force_runtimes) / len(brute_force_runtimes)
avg_apriori_runtime = sum(apriori_runtimes) / len(apriori_runtimes)

# Print overall summary
print("Overall Summary:")
print(f"Average FP-Growth Runtime: {avg_fp_growth_runtime} seconds")
print(f"Average Brute Force Runtime: {avg_brute_force_runtime} seconds")
print(f"Average Apriori Runtime: {avg_apriori_runtime} seconds")

