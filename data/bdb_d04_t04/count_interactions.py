import pandas as pd

am = pd.read_csv("affinity_matrix.csv", header=0, index_col=0, sep=",")

print(am.shape)
print("Good interactions:")
print((am.shape[0]*am.shape[1])-(am.isna().sum().sum()))
print("Nan interactions:")
print(am.isna().sum().sum())


