#!/usr/bin/env python3

import pandas

D = pandas.read_excel("Data_Methodes_Factorielles.xlsx", sheet_name="AFC_ETUDES", index_col=0)
print(D)


#librairie

import numpy

#calcul des totaux en ligne

tot_lig =numpy.sum(D.values,axis=1)

print(tot_lig)


#[ 302  575 1825  467  615]

#calcul des totaux en colonne

tot_col = numpy.sum(D.values,axis=0)
print(tot_col)


#profils lignes

prof_lig = numpy.apply_along_axis(arr = D.values, axis=1, func1d=lambda x:x/numpy.sum(x))

print(prof_lig)


#représentation graphique

import matplotlib.pyplot as plt

somme = numpy.zeros(shape=(prof_lig.shape[0]))

for i in range(prof_lig.shape[1]):
    plt.barh(range(prof_lig.shape[0]),prof_lig[:,i],left=somme)
    somme =somme + prof_lig[:,i]

plt.yticks(range(prof_lig.shape[0]),D.index)
plt.show()



#calul du profil marginal corresp.–ligne grisée de Figure 100

prof_marg_lig = tot_col/numpy.sum(tot_col)
print(prof_marg_lig)


#distance du KHI-2 entre cadre(2) et ouvrier(4)

print(numpy.sum((prof_lig[2,:]-prof_lig[4,:])**2/prof_marg_lig))
#distance du KHI-2 entre cadre(2) et patron(1)

print(numpy.sum((prof_lig[2,:]-prof_lig[1,:])**2/prof_marg_lig))

#distance entre paires de modalités lignes
distPairesLig =numpy.zeros(shape=(prof_lig.shape[0],prof_lig.shape[0]))


#double boucle
for i in range(prof_lig.shape[0]-1):
    for j in range(i+1,prof_lig.shape[0]):
        distPairesLig[i,j] = numpy.sum((prof_lig[i,:]-prof_lig[j,:])**2/prof_marg_lig)
    #distPairesLig[j,i] = distPairesLig[i,j]

#affichage
print(pandas.DataFrame(distPairesLig,index=D.index,columns=D.index))

#affichage sous forme de heatmap
import seaborn as sns

sns.heatmap(distPairesLig,vmin=0,vmax=numpy.max(distPairesLig),linewidth=0.1,cmap='Blues',xticklabels=D.index,yticklabels=D.index)
