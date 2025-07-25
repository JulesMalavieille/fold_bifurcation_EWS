"""
Created on Mon Jun  3 13:53:24 2024

@author: Jules Malavieille
"""

# Solving stochastical differential equation by Euler-Maruyama and Milsetin scheme 
# Equation : dX = a*X*dt + b*X*dW

import numpy as np 
import matplotlib.pyplot as plt 

""" Parameters """
a = 2       
b = 0.15      

tmax = 100
nbval = 1000
R = 4       # number of time steps for Wiener process  
L = nbval/R
N = 2**9
dt = 1/N


""" Wiener process """
def bruit():
    B = []
    for i in range(nbval):
        epsi = np.sqrt(dt)*np.random.normal(0,1)
        B.append(epsi)
    return B
    

""" Analytic solution """ 
def func(W):
    X =[]
    X0 = 1
    for i in range(nbval):
        x = X0 * np.exp(((a-b**2/2)*(i+1)*dt) + b*W[i])
        X.append(x)
    return X
        

""" Euler-Maruyama Scheme """
def EulerMaruyama(B):
    R = 4
    Dt = R*dt
    E = []
    e = 1
    E.append(e)
    for j in range(int(L)):
        Winc = 0
        for i in range(R):
            Winc += B[j*R+i] 
        e = E[j] + a*E[j]*Dt + b*E[j]*Winc 
        E.append(e)
    return E


""" Milstein Scheme """
def Milstein(B):
    X0 = 1
    R = 1
    Xtemp = [X0]
    Dt = R*dt
    Winc = 0
    for j in range(nbval):
        Winc = 0
        for i in range(R):
            Winc += B[j*R+i]
        X = Xtemp[j] + Dt*a*Xtemp[j] + b*Xtemp[j]*Winc + 0.5*b**2*Xtemp[j]*(Winc**2 - Dt)
        Xtemp.append(X)
    return Xtemp
            
            
B = bruit()
W = np.cumsum(B)

X = func(W)
E = EulerMaruyama(B)
M = Milstein(B)

t = np.linspace(0, tmax, nbval)
tt = np.linspace(0, tmax, nbval+1)
T = np.linspace(0, tmax, len(E))

plt.plot(t, X, label="Solution analytique")
plt.plot(T, E, label="Schéma d'Euler-Maruyama")
plt.plot(tt, M, label="Schéma de Milstein")
plt.title("Résolution d'une équation différentielle stochastique par le schéma de Milstein", fontsize=18)
plt.xlabel("Temps", fontsize=15)
plt.ylabel("X", fontsize=15)
plt.grid()
plt.legend()




