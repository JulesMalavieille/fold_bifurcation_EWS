"""
Created on Thu Jun  6 10:26:09 2024

@author: Jules Malavieille
"""

# Résolution du modèle : dX = rX(1-X/K) - (X**2/(1+X**2))dW

import numpy as np
import matplotlib.pyplot as plt 
import csv

r = 1
K = 10
b = 0.15

nbval = 500000
N = 2**8
dt = 1/N
tmax = int(dt*nbval)
coef = 1


""" Génération du bruit """
def bruit():
    B = []
    for i in range(nbval):
        epsi = np.sqrt(dt)*np.random.normal(0,1)
        B.append(epsi)
    return B
    

""" Schéma de Milstein """
def Milstein(B):
    X0 = 4
    R = 1
    Xtemp = [X0]
    Dt = R*dt
    for j in range(nbval-1):
        Winc = 0
        for i in range(R):
            Winc += B[j*R+i]
        X = Xtemp[j] + Dt*(r*Xtemp[j]*(1-Xtemp[j]/K) - Xtemp[j]**2/(1+Xtemp[j]**2)) + b*Xtemp[j]*Winc + 0.5*b**2*Xtemp[j]*(Winc**2 - Dt)
        Xtemp.append(X)
    return Xtemp


""" Schéma de Milstein deterministe """
def Milstein_det(B):
    X0 = 4
    R = 1
    Xtemp = [X0]
    Dt = R*dt
    for j in range(nbval-1):
        X = Xtemp[j] + Dt*(r*Xtemp[j]*(1-Xtemp[j]/K) - Xtemp[j]**2/(1+Xtemp[j]**2)) 
        Xtemp.append(X)
    return Xtemp


""" Schéma de Milstein avec variation de paramètre"""
def Milstein_param(B):
    X0 = 8
    R = 1
    Xtemp = [X0]
    Dt = R*dt
    r = 1
    rL = [r]
    for j in range(nbval-1):
        Winc = 0
        for i in range(R):
            Winc += B[j*R+i]
        X = Xtemp[j] + Dt*(rL[j]*Xtemp[j]*(1-Xtemp[j]/K) - Xtemp[j]**2/(1+Xtemp[j]**2)) + b*Xtemp[j]*Winc + 0.5*b**2*Xtemp[j]*(Winc**2 - Dt)
        Xtemp.append(X)
        r -= 0.000002
        if r < 0:
            r = 0
        rL.append(r)
    return Xtemp, rL
    

""" Schéma de milstein pour un modèle deterministe"""
def Milstein_param_deterministe(B):
    X0 = 8
    R = 1
    Xtemp = [X0]
    Dt = R*dt
    r = 1
    rL = [r]
    for j in range(nbval-1):
        X = Xtemp[j] + Dt*(rL[j]*Xtemp[j]*(1-Xtemp[j]/K) - Xtemp[j]**2/(1+Xtemp[j]**2)) 
        Xtemp.append(X)
        r -= 0.000002
        if r < 0:
            r = 0
        rL.append(r)
    return Xtemp, rL


""" Tronquer une liste"""
def cut_L(L, x):
    L_cut = []
    i = 0
    for l in L:
        if i == 0:
            L_cut.append(l)
        i += 1
        if i == x:
            i = 0
    return L_cut
        
    

""" Partie du modèle f et g """
def f(X):
    Y = []
    for x in X:
        y = r*(1-x/K)
        Y.append(y)
    return Y


def g(X):
    Y = []
    for x in X:
        y = x/(1+x**2)
        Y.append(y)
    return Y


""" calcul de r et K """
def r_calc(X):
    Y = []
    for x in X:
        r = (-2*x**3*(-1+x**2))/((1-x**2)*((1+x**2)**2))
        Y.append(r)
    return Y


def K_calc(X):
    Y = []
    for x in X:
        K = (2*x**3)/(x**2-1)
        Y.append(K)
    return Y


#B = bruit()
#E = Milstein(B)
# E_r, rr = Milstein_param(B)
# E_rd, rrd = Milstein_param_deterministe(B)
#E_det = Milstein_det(B)

t = np.linspace(0, tmax, nbval)
t_r = np.linspace(0, tmax, int(nbval))
Xp = np.linspace(0, 10, 1000)
X = np.linspace(1, 25*coef, 1000*coef)

f = f(Xp)
g = g(Xp)
rL = r_calc(X)
KL = K_calc(X)


""" Modèle en fonction du temps"""
# plt.figure(1)
# plt.plot(t, E, label="Modèle détérministe") 
# plt.plot(t, E_det, label="Modèle stochastique")
# plt.title("Modèle du ver d'épicéa", fontsize=18)
# plt.xlabel("Temps", fontsize=15)
# plt.ylabel("X", fontsize=15)
# plt.grid()
# plt.legend()


""" Portrait de phase de f et g """
# plt.figure(2)
# plt.plot(Xp, f, label="f(x)")
# plt.plot(Xp, g, label="g(x)")
# plt.title("Portrait de phase de f et g", fontsize=18)
# plt.ylabel("f(x) et g(x)", fontsize=15)
# plt.xlabel("x", fontsize=15)
# plt.ylim(0)
# plt.grid()
# plt.legend()


""" Diagramme de r en fonction de K """
# plt.figure(3)
# plt.plot(KL, rL)
# plt.title("Diagramme de r en fonction de K", fontsize=18)
# plt.xlabel("K", fontsize=15)
# plt.ylabel("r", fontsize=15)
# plt.grid()

# t = 0
# for i in range(10, len(KL)):
#     if KL[i] > K:
#         if t == 0:
#             print()
#             print("Pour K =",K, "la valeur de r pour le passage à l'état basse densité est",round(rL[i], 3))
#             print()
#             t += 1
        
# t = 0
# for i in range(0, len(KL)):
#     if KL[i] < K:
#         if t == 0:
#             print()
#             print("Pour K =",K, "la valeur de r pour le passage à l'état haute densité est",round(rL[i], 2))
#             print()
#             t += 1
        
     
""" Modélisation régime shift """
for i in range(1):
    B = bruit()
    E = Milstein(B)
    E_r, rr = Milstein_param(B)
    t_r = np.linspace(0, tmax, int(nbval))
    """ Modèle avec r variable"""
    #plt.figure(4)
    E_r_cut = cut_L(E_r, int(nbval/10000))
    #plt.plot(t_r ,E_r, color="black")
    plt.plot(E_r_cut)
    
# #plt.plot(t_r, E_rd, label="Modèle deterministe", linewidth=2)
# #plt.plot(t_r, rr, label="Driver : valeur de r")
# plt.xlabel("Temps", fontsize=15)
# plt.ylabel("X", fontsize=15)
# plt.title("Modèle en fonction du temps pour un r variable", fontsize=18)
# plt.legend()
# plt.grid()
    

""" X réplicats du modèle avec r variable """  # Attention demande beaucoup de puissance de calcul ou de temps
# div = 1000  # Facteur par lequel on tronque la serie 
# rep = 1000 # Pour nbval = 500 000; N = 2**8; b = 0.15; r = 1 - 0.000002
# it = 0
# M = np.zeros([int(nbval/div), rep])
# for i in range(rep):
#     B = bruit()
#     E_r, rr = Milstein_param(B)
#     E_rr = cut_L(E_r, div)
#     M[:, i] = E_rr
#     it += 1
#     print(it)
    
# nom = "matrice.csv"

# with open(nom, mode='w', newline='') as fichier_csv:
#     writer = csv.writer(fichier_csv)
#     for ligne in M:
#         writer.writerow(ligne)





