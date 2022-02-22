# the interactions file holds Kd values, this file changes them into pkd

import csv
import numpy as np
import pandas as pd
import sys

alpha = pd.read_csv(sys.argv[1])
beta = alpha.iloc[:,:1]
alpha = alpha.drop(["cids"],axis=1)
alpha = alpha.apply(lambda x : -np.log10(x/1e9))
alpha = pd.concat([beta,alpha],axis=1)

alpha.to_csv("affinity_matrix.csv",index=False,header=True)



