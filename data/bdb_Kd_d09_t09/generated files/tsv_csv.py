
import pandas as pd
  
tsv = pd.read_csv("affinity_matrix.csv", header=0, index_col=0, sep='\t')
tsv.to_csv("affinity_matrix.csv", sep=',')

