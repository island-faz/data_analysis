#!/usr/bin/env python3

import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


data = pandas.read_excel("autos-acp2.xlsx", sheet_name=0, header=0, index_col=0)

sc = StandardScaler()
scaled_data = sc.fit_transform(data)

print(data)
#print(scaled_data)

pca = PCA(n_components=2)
pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

print(scaled_data.shape)
print(x_pca)

"""
plt.figure(figsize=(6, 6))
plt.scatter(x_pca[:,0], x_pca[:,1])
plt.show()
"""

fig, axes = plt.subplots(figsize=(12,12))
axes.set_xlim(-6,6) #même limites en abscisse
axes.set_ylim(-6,6) #et en ordonnée

n = len(data)


for i in range(n):
    plt.annotate(data.index[i], (x_pca[i,0], x_pca[i,1]))

#ajouter les axes
plt.plot([-6,6],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)
#affichage
plt.show()
