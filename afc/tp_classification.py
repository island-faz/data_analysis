#!/usr/bin/env python3

import pandas
import matplotlib.pyplot as plt

autos = pandas.read_excel("Data_Methodes_Factorielles.xlsx", sheet_name="AUTOS_MDS_SOURCE", index_col=0)
print(autos)

from pandas.plotting import scatter_matrix
#scatter_matrix(autos, figsize=(5,5))
#plt.show()

from sklearn import preprocessing
autos_cr = preprocessing.scale(autos)

print(autos_cr)

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Z = linkage(autos_cr, method='ward', metric='euclidean')

#plt.title("CAH")

dendrogram(Z, labels=autos.index, orientation='top', color_threshold=0, leaf_rotation=90)
plt.show()
