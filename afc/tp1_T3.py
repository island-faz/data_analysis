#!/usr/bin/env python3

import pandas
import numpy

#importation de la librairie
from fanalysis.ca import CA

D = pandas.read_excel("Data_Methodes_Factorielles.xlsx", sheet_name="AFC_ETUDES", index_col=0)

print(D)

#nombre de modalités ligne
K =D.shape[0]
#nombre de modalités colonnes
L =D.shape[1]

#effectif total
n =numpy.sum(D.values)
print(n)

Hmax = numpy.min([K-1,L-1])
print(Hmax)

#lancer les calculs
afc =CA(row_labels=D.index,col_labels=D.columns)
afc.fit(D.values)

#propriétés de l'objet
# print(dir(afc))

#affichage des valeurs propres
print("\n------Valeurs propres-------")
print(afc.eig_)
print("----------------------------\n")

#seuil -- moyenne des valeurs propres
meanValPropre =numpy.mean(afc.eig_[0])
print("Moy de valeurs propres: " + str(meanValPropre))

#ou seuil sur les pourcentages
print("Seuil sur les pourcentages: " + str(1/Hmax* 100) )

#afc.plot_eigenvalues()
#plt.show()

print("\n------ Infos sur les modalitées lignes -------")
print(afc.row_topandas())
print("----------------------------------------------\n")


#profil marginal des modalités lignes
profMargLig =numpy.sum(D.values,axis=1)/n
print("profile marginal des modalités lignes" + str(profMargLig))


#contributions
contribLig = (numpy.reshape(profMargLig, (5, 1)) * afc.row_coord_**2) / afc.eig_[0] * 100
print("\n------ Contributions -------")
print(contribLig)
print("\n----------------------------")


#distance à l'origine -distance du KHI2
print("----------------- distance à l'origine - distance du KHI2 ------------------")
distoLig = numpy.sum(afc. row_coord_**2, axis=1)
print(distoLig)
print("----------------------------------------------------------------------------")

#cos2 des lignes
cos2Lig = afc.row_coord_**2/numpy.reshape(distoLig, (5,1)) * 100
print("----------------- Cos2 des Lignes ------------------")
print(cos2Lig)
print("----------------------------------------------------")

#statistiques pour les points colonnes
print("----------------- statistiques pour les points colonnes ------------------")
print(afc.col_topandas())
print("--------------------------------------------------------------------------")

#représentation simultanée
afc.mapping(num_x_axis=1, num_y_axis=2, figsize=(7,7))

plt.show()

"""
import matplotlib.pyplot as plt

fig, ax =plt.subplots(figsize=(10,10))
ax.axis([-0.7,+0.7,-0.7,+0.7])
ax.plot([-0.7,+0.7],[0,0],color='silver',linestyle='--')
ax.plot([0,0],[-0.7,+0.7],color='silver',linestyle='--')
ax.set_xlabel("Dim.1")
ax.set_ylabel("Dim.2")

plt.title("Modalité ligne supplémentaire")

for i inrange(D.shape[0]):
    ax.text(afc.row_coord_[i,0], afc.row_coord_[i,1], D.index[i], color='red')

for i inrange(D.shape[1]):
    ax.text(afc.col_coord_[i,0], afc.col_coord_[i,1], D.columns[i],color='blue')

ax.text(coordBourges[0][0], coordBourges[0][1], "Bourgeois", color='green')
plt.show()
"""
