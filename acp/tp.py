#!/usr/bin/env python3

import pandas

D = pandas.read_excel("Data_Methodes_Factorielles.xlsx", sheet_name="DATA_ACP_ACTIF", index_col=0)
print(D.info())
print("---------------------------------------------------------")
print(D)
print("---------------------------------------------------------")
#importer la librairie graphique
import matplotlib.pyplot as plt


#préparer le graphique
"""
fig, ax =plt.subplots(figsize=(10,10))
ax.plot(D.CYL,D.PUISS,"wo")
ax.axis([1000,3000,50,140])
ax.set_xlabel("CYL")
ax.set_ylabel("PUISS")
#ajouter les labels des véhicules
for v in D.index:
    ax.text(D.CYL[v], D.PUISS[v], v)
"""

#faire afficher
#plt.show()

#afficher les deux variables en triant selon CYL
Dbis = D.sort_values(by="CYL",ascending=True)[['CYL','PUISS']]
print(Dbis)


#librairie graphique
import seaborn as sns

#pairplot
#sns.pairplot(D)

#plt.show()


#librairie numpy pour les manipulations matricielles
import numpy

#distance entre l'Audi et la Fiat
print("distance entre l'Audi et la Fiat: " + str(numpy.sqrt(numpy.sum((Dbis.loc['Audi 100'] - Dbis.loc['Fiat 132']) ** 2))) )

#distance entre l'Audi et la Mazda

print("distance entre l'Audi et la Mazda: " + str(numpy.sqrt(numpy.sum((Dbis.loc['Audi 100']-Dbis.loc['Mazda 9295'])**2))))

#matrice X numpy –plus facile à manipuler
X = Dbis.values

print(X)

#écarts-type des variables CYL et PUISS
sigmas =numpy.std(X,axis=0,ddof=0)
print("Ecarts-type(Sigma): " + str(sigmas))

#distance pondérée entre l'Audi et la Fiat
distPonderee = numpy.sqrt(numpy.sum((1/sigmas**2)*(Dbis.loc['Audi 100']-Dbis.loc['Fiat 132'])**2))

print("Distance Pondérée entre Audi et Fiat: " + str(distPonderee))

#distance entre l'Audi et la Mazda
dist_audi_mazda = numpy.sqrt(numpy.sum((1/sigmas**2)*(Dbis.loc['Audi 100']-Dbis.loc['Mazda 9295'])**2))
print("distance entre l'Audi et la Mazda: " + str(dist_audi_mazda))

#nombre d'observations
n =X.shape[0] #18
#inertie version 1 --distancesentre paires d'individus
Ip_v1 = 0
#double boucle(i1, i2)
for i1 in range(n):
    for i2 in range(n):
        Ip_v1 = Ip_v1 +numpy.sum((X[i1,:]-X[i2,:])**2)

#moyenne des écarts entre paires d'individus
Ip_v1 =(1/(2*n**2)) * Ip_v1
print("Inertie, Approche 1 = %.2f"%(Ip_v1))


#moyennes des variables -- Coordonnées de G (vecteur des moyennes)
moyennes = numpy.mean(X, axis=0)
print("moyennes des variables -- Coordonnées de G: " + str(moyennes))

#graphique des points avec le barycentre G
fig, ax =plt.subplots(figsize=(10,10))
ax.plot(D.CYL,D.PUISS,color='xkcd:light blue',marker='o',linestyle='None')
ax.axis([1000,3000,50,140])
ax.set_xlabel("CYL")
ax.set_ylabel("PUISS")

#ajouter des traits pointillés entre les points et le barycentre
for i in range(D.shape[0]):
    ax.plot([moyennes[0], D.CYL[i]], [moyennes[1], D.PUISS[i]], color='silver', linestyle='dashed')

#barycentre G
ax.text(moyennes[0],moyennes[1],"G",fontsize=14)

#plt.show()

#inertie version 2 -écarts au barycentre
Ip_v2 = numpy.mean(numpy.apply_along_axis(func1d=lambda x: numpy.sum((x-moyennes)**2), axis=1, arr=X))
print("Inertie, Approche 2 = %.2f"%(Ip_v2))

#vérification --somme des variances
print("Somme des variances = %.2f"%(numpy.sum(sigmas**2)))

#données centrées et réduites
Z=(X-moyennes)/sigmas
print(pandas.DataFrame(Z,index=Dbis.index))

#vérification moyennes-nulles
print(numpy.mean(Z,axis=0))

#vérification écarts-type–égaux à 1
print(numpy.std(Z,axis=0,ddof=0))

#outil pour l'ACP
from sklearn.decomposition import PCA
acp =PCA()

#coordonnées factorielle
coord =acp.fit_transform(Z)

#afficher les nouvelles coordonnées
print(pandas.DataFrame(coord,index=Dbis.index))

#position des véhicules dans le repère factoriel

"""
fig, ax =plt.subplots(figsize=(10,10))
ax.plot(coord[:,0],coord[:,1],"wo")
ax.axis([-4,+4,-4,+4])
ax.plot([-4,+4],[0,0],color='silver',linestyle='--')
ax.plot([0,0],[-4,+4],color='silver',linestyle='--')
ax.set_xlabel("Comp.1 (89.83%)")
ax.set_ylabel("Comp.2 (10.17%)")

#ajouter les labels des véhicules
for i in range(n):
    ax.text(coord[i,0], coord[i,1], Dbis.index[i])
"""

#faire afficher
#plt.show()

#former la matrice X avec (p=6) variables maintenant
X =D.values
#calculer la matrice de covariance
#rowvar = False pour dire que les variables sont organisées en colonnes
#ddof = 0 pour utiliser (1/n)
V =numpy.cov(X,ddof=0,rowvar=False)
numpy.set_printoptions(precision=2,suppress=True)
print(V)


#calculer la trace de la matrice V
trace =V.trace()
print(trace)


#vecteur moyenne pour (p = 6) variables
moyennes =numpy.mean(X,axis=0)
#inertie par l'écart au barycentre
print(numpy.mean(numpy.apply_along_axis(func1d=lambda x: numpy.sum((x-moyennes)**2),axis=1,arr=X)))


#matrice Z des variables centrées
Z =X -moyennes
print(pandas.DataFrame(Z,index=D.index))

#matrice des corrélations
R=numpy.corrcoef(X,rowvar=False)
print(R)

#vecteur ecart-type pour (p = 6) variables
sigmas =numpy.std(X,axis=0)

#centrage et réduction
Z =(X -moyennes)/sigmas

#correspondance : produit matriciel : (1/n) (Z'Z)
print(numpy.dot(numpy.transpose(Z),Z)/n)

#heat map pour identifier visuellement les corrélations fortes

"""
sns.heatmap(R,xticklabels=D.columns,yticklabels=D.columns,vmin=-1,vmax=+1,center=0,cmap="RdBu",linewidths=0.5)
plt.show()
"""

acp =PCA()

#coordonnées factorielles
coord =acp.fit_transform(Z)

#afficher les nouvelles coordonnées des premiers véhicules
print(pandas.DataFrame(coord,index=Dbis.index).head())

#corrélation des facteurs avec les variables
#Mlambda =numpy.corrcoef(x=coord, y=Z, rowvar=False)[:p,p:]

#affichage des corrélations : lignes = facteurs, colonnes = variables
#print(Mlambda)

#slambda =numpy.sum(Mlambda**2,axis=1)
#print(slambda)


#dont la somme totale (des lambda)= inertie = p puisque ACP normée
#print(numpy.sum(slambda))

acp.correlation_circle(num_x_axis=1, num_y_axis=2)
plt.show()
