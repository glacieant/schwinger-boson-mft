#!/usr/bin/env python
import os
import multiprocessing as mp
import itertools
import time
import datetime
import sys

#from mpi4py import MPI
#from mpi4py.futures import MPIPoolExecutor

#mode = "cluster"
mode = "pc"

# OMP threads 
#if mode == 'pc':
#    os.environ["OMP_NUM_THREADS"] = "10"

import numpy as np
from numpy import linalg

from scipy import optimize
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.figure as figure
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FormatStrFormatter

def compilevar(mu,Q,Z):

    X = np.hstack((mu,Q.real,Q.imag,Z.real,Z.imag))

    return X

def Zravel(Z):
    
    z1 = np.array([Z[0],Z[3].conj()])
    z2 = np.array([Z[1],Z[4].conj()])
    z3 = np.array([Z[2],Z[5].conj()])

    return z1, z2, z3

def expandvar(X):
    
    mu = np.abs(X[:3])
    Q = X[3:6]+1j*X[6:9]
    Z = X[9:15]+1j*X[15:21]

    return mu, Q, Z

def Ham(X,NK,L):

    mu, Q, Z = expandvar(X)
    
    n1 = int(NK/L)
    n2 = NK%L

    k1 = 2*n1*np.pi/L
    k2 = 2*n2*np.pi/L
    k3 = -k1-k2

    PHASEab = 1+np.exp(-1j*k3)+np.exp(1j*k2)
    PHASEbc = np.exp(1j*k3)+1+np.exp(-1j*k1)
    PHASEca = np.exp(-1j*k2)+np.exp(1j*k1)+1
    
    B12 = -Q[0]*(PHASEab)/2
    B13 = Q[2]*(PHASEca.conjugate())/2
    B21 = Q[0]*(PHASEab.conjugate())/2
    B23 = -Q[1]*(PHASEbc)/2
    B31 = -Q[2]*(PHASEca)/2
    B32 = Q[1]*(PHASEbc.conjugate())/2

    B = np.array([[0,B12,B13],
        [B21,0,B23],
        [B31,B32,0]]
        )
    
    A11 = mu[0]
    A22 = mu[1]
    A33 = mu[2]
    
    A = np.array([[A11,0,0],
        [0,A22,0],
        [0,0,A33]])
    
    D = np.block([[A,B],[B.conj().transpose(),A]])

    return D
 
def ZeemanHam(X,h,S,NK,L):

    return Ham(X,NK,L) - (h*S/2)*np.diag([1,1,1,-1,-1,-1])

def eig(X,h,S,NK,L):
    
    D = Ham(X,NK,L)
    Sigma = np.diag([1,1,1,-1,-1,-1])

    l, V = np.linalg.eig(Sigma.dot(D))
    LL = (V.conj().transpose()).dot(Sigma).dot(V)
    r, U = np.linalg.eigh(LL)
    rax = np.argsort(-r)
    rhf = np.diag(1/np.sqrt(np.abs(r[rax])))
    U[:] = U[:,rax]
    T = V.dot(U).dot(rhf)
    TD = T.conj().transpose()

    ENX = TD.dot(D).dot(T)
    EQX = np.real(ENX-D)
    
    return np.trace(EQX)/2

def eigSum(X,h,S,L):
    
    YS = 0
    
    for NK in range(1,L*L):

        YS += eig(X,h,S,NK,L)

    return YS/(L**2)
    

def clen(X,h,S,L):

    mu, Q, Z = expandvar(X)

    return ((3/2)*np.linalg.norm(Q)**2
            + np.vdot(Z,np.dot(ZeemanHam(X,h,S,0,L),Z)).real
            - np.sum(mu)*(2*S)
            )

def rotSpinor(theta,phi,chi):

    return np.array([[np.cos(theta/2)*np.exp(1.0j*phi/2)*np.exp(1.0j*chi/2),
        np.sin(theta/2)*np.exp(-1.0j*phi/2)*np.exp(1.0j*chi/2)],
        [-np.sin(theta/2)*np.exp(1.0j*phi/2)*np.exp(-1.0j*chi/2),
            np.cos(theta/2)*np.exp(-1.0j*phi/2)*np.exp(-1.0j*chi/2)
            ]])

def FreeEN(X,h,S,L):
    
    EQ = eigSum(X,h,S,L)
    
    ECL = clen(X,h,S,L)

    return ECL + EQ + (9/4)*((2*S)**2)

def paramMAKE(h,S,Z):

    z1, z2, z3 = Zravel(Z)

    Q12 = np.dot(z1,np.dot(leviC,z2))
    Q23 = np.dot(z2,np.dot(leviC,z3))
    Q31 = np.dot(z3,np.dot(leviC,z1))
        
    if np.abs(z1[0]) > gTol:
        mu1 = np.abs(-((-h*(S/2)*pauli[2,0,0]*z1[0]+(1/2)*(-3*Q12*leviC[0,1]*z2[1].conjugate()
                                                    -3*Q31*leviC[1,0]*z3[1].conjugate()
                                                    ))/z1[0]))
    else:
        mu1 = np.abs(-((-h*(S/2)*pauli[2,1,1]*z1[1]+(1/2)*(-3*Q12*leviC[1,0]*z2[0].conjugate()
                                                    -3*Q31*leviC[0,1]*z3[0].conjugate()
                                                    ))/z1[1]))
    if np.abs(z2[0]) > gTol:
        mu2 = np.abs(-((-h*(S/2)*pauli[2,0,0]*z2[0]+(1/2)*(-3*Q23*leviC[0,1]*z3[1].conjugate()
                                                    -3*Q12*leviC[1,0]*z1[1].conjugate()
                                                    ))/z2[0]))
    else:
        mu2 = np.abs(-((-h*(S/2)*pauli[2,1,1]*z2[1]+(1/2)*(-3*Q23*leviC[1,0]*z3[0].conjugate()
                                                    -3*Q12*leviC[0,1]*z1[0].conjugate()
                                                    ))/z2[1]))
    if np.abs(z3[0]) > gTol:
        mu3 = np.abs(-((-h*(S/2)*pauli[2,0,0]*z3[0]+(1/2)*(-3*Q31*leviC[0,1]*z1[1].conjugate()
                                                    -3*Q23*leviC[1,0]*z2[1].conjugate()
                                                    ))/z3[0]))
    else:
        mu3 = np.abs(-((-h*(S/2)*pauli[2,1,1]*z3[1]+(1/2)*(-3*Q31*leviC[1,0]*z1[0].conjugate()
                                                    -3*Q23*leviC[0,1]*z2[0].conjugate()
                                                    ))/z3[1]))

    return np.array([mu1, mu2, mu3]), np.array([Q12,Q23,Q31])



def ZMAKE(VA,kappa):
    
    thetaARR = VA[:3]
    chiARR = VA[3:6]
    phiARR = VA[6:9]

    z1 = np.dot(rotSpinor(thetaARR[0],phiARR[0],chiARR[0]),np.array([np.sqrt(kappa),0]))
    z2 = np.dot(rotSpinor(thetaARR[1],phiARR[1],chiARR[1]),np.array([np.sqrt(kappa),0]))
    z3 = np.dot(rotSpinor(thetaARR[2],phiARR[2],chiARR[2]),np.array([np.sqrt(kappa),0]))

    Z = np.array([z1[0],z2[0],z3[0],z1[1].conj(),z2[1].conj(),z3[1].conj()])
    
    return Z

def SolveCLQU(iS,ich,ih,S,char,h,mfTol,L,MaxIter,fstring,rundata):
    
    kappa = 2*S
    thetaARR = theta(char,h)
    chiARR = chi(char,h)
    phiARR = phi(char,h)
    VA = np.ravel([thetaARR,chiARR,phiARR])
    Z = ZMAKE(VA,kappa)

    mu, Q = paramMAKE(h,S,Z)

    X = compilevar(mu,Q,Z) 
    
    En = FreeEN(X,h,S,L)/3

    z1, z2, z3 = Zravel(Z)
    
    S1 = np.array([np.vdot(z1,np.dot(pauli[0],z1)),np.vdot(z1,np.dot(pauli[1],z1)),np.vdot(z1,np.dot(pauli[2],z1))])/2
    S2 = np.array([np.vdot(z2,np.dot(pauli[0],z2)),np.vdot(z2,np.dot(pauli[1],z2)),np.vdot(z2,np.dot(pauli[2],z2))])/2
    S3 = np.array([np.vdot(z3,np.dot(pauli[0],z3)),np.vdot(z3,np.dot(pauli[1],z3)),np.vdot(z3,np.dot(pauli[2],z3))])/2
    Sz = np.abs(np.sum(S1+S2+S3))/3
    
    m = kappa/2

    print(char,h,S,Sz,m,char)
    sys.stdout.flush()

    if rundata == "on":

        with open(fstring,"a") as file:
            file.write(f"{iS} {ich} {ih} {Sz} {En} {m} {mu[0]} {mu[1]} {mu[2]} {Q[0]} {Q[1]} {Q[2]} {Z[0]} {Z[1]} {Z[2]} {Z[3]} {Z[4]} {Z[5]}\n")

    return iS, ich, ih, Sz, En, mu, Q, Z, m, char

def theta(string,h):

    if string == 'umbrella' or string == 'umbrellaSL':

        if h <= 9:
        
            ARR = np.array([np.arccos(h/9),np.arccos(h/9),np.arccos(h/9)])

        else:
        
            ARR = np.array([1,1,1])

    
    elif string == 'yv' or string == 'yvSL':

        if h <= 3:

            ARR = np.array([np.arccos((3+h)/6),-np.arccos((3+h)/6),np.pi])
    
        elif h > 3 and h <= 9:

            ARR = np.array([-np.arccos((-27+h**2)/(6*h)),np.arccos((27+h**2)/(12*h)),np.arccos((27+h**2)/(12*h))])

        else:

            ARR = np.array([1,1,1])

    elif string == 'uud' or string == 'uudSL':

        ARR = np.array([0,0,np.pi])
    
    return ARR

def chi(string,h):

    if string == 'umbrella' or string == 'umbrellaSL':

        ARR = np.array([0,2*np.pi/3,4*np.pi/3])
    
    elif string == 'yv' or string == 'yvSL':
 
        ARR = np.array([0,0,0])
     
    elif string == 'uud' or string == 'uudSL':
 
        ARR = np.array([0,0,0])
    
    return ARR

def phi(string,h):

    if string == 'umbrella' or string == 'umbrellaSL':
        
        ARR = np.array([0,0,0])
    
    elif string == 'yv' or string == 'yvSL':
        
        ARR = np.array([0,0,0])
     
    elif string == 'uud' or string == 'uudSL':
 
        ARR = np.array([0,0,0])
 
    return ARR

## External Parameters

pauli = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
leviC = np.array([[0,1],[-1,0]])
dt = 10**(-4)
gTol = 10**(-16)
mfTol = 10**(-3)
L = 6
MaxIter = 1000
SlqMxRatio = 0.5

rundata = "off"

SARR = np.linspace(0.05,0.15,endpoint=False,num=16)
SARR = np.append(SARR,np.linspace(0.15,0.19,endpoint=False,num=24))
SARR = np.append(SARR,np.linspace(0.19,0.25,endpoint=False,num=16))
SARR = np.append(SARR,[0.25,0.5,0.75,1.0])
SARR = np.append(SARR,np.linspace(2,100,endpoint=False,num=16))

SARR = np.array([1/2,1])

CONFIG = np.array(['yv','uud','umbrella'])

hARR = np.linspace(0,3,endpoint=False,num=16)
hARR= np.append(hARR,np.linspace(3,5,endpoint=False,num=16))
hARR= np.append(hARR,np.linspace(5,6,endpoint=False,num=16))
hARR= np.append(hARR,np.linspace(6,7,endpoint=False,num=16))
hARR= np.append(hARR,np.linspace(7,8,endpoint=False,num=16))
hARR= np.append(hARR,np.linspace(8,9,num=16))

muARR = np.zeros((SARR.size,CONFIG.size,hARR.size,3),dtype=float) 
QARR = np.zeros((SARR.size,CONFIG.size,hARR.size,3),dtype=complex) 
ZARR = np.zeros((SARR.size,CONFIG.size,hARR.size,6),dtype=complex) 
freeARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float) 
SzARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float) 
mARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float) 

if __name__=='__main__':
    
    dpath = 'run_data'
    TSTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H-%M-%S')
    fstring = dpath+"/runningData_"+TSTAMP

    if rundata == "on":

        # Data directory
        if not os.path.exists(dpath):
            os.makedirs(dpath)

        f = open(fstring+".dat","w")

        np.savez_compressed(fstring+".npz",
                        L = L,
                        SARR = SARR,
                        hARR = hARR,
                        CONFIG = CONFIG
                        )

    ex_arr = []

    for iS in range(SARR.size):
    
        S = SARR[iS]
     
        for ich in range(CONFIG.size):

            char = CONFIG[ich]
    
            for ih in range(hARR.size):
    
                h = hARR[ih]
            
                ex_arr.append((iS,ich,ih,S,char,h,mfTol,L,MaxIter,fstring+".dat",rundata))
     
    if mode == "cluster":

  
        with MPIPoolExecutor() as executor:
            res = list(executor.starmap(SolveCLQU,ex_arr))
     
    elif mode == "pc":
    
        with mp.Pool() as pool:
            res = pool.starmap(SolveCLQU,ex_arr)
        #res = [SolveCLQU(*exr) for exr in ex_arr]
        
    for ires in range(len(res)):
        
        iS, ich, ih, Sz, En, mu, Q, Z, m, char = res[ires]
            
        muARR[iS,ich,ih] = mu
        QARR[iS,ich,ih] = Q
        freeARR[iS,ich,ih] = En
        ZARR[iS,ich,ih] = Z
        SzARR[iS,ich,ih] = Sz
        mARR[iS,ich,ih] = m
        
    TSTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H-%M-%S')
    np.savez_compressed('L_'+str(L)+'_TS_'+TSTAMP+'_'+'largeS_data.npz',
                        SARR=SARR,
                        hARR=hARR,
                        CONFIG=CONFIG,
                        muARR=muARR,
                        QARR=QARR,
                        ZARR=ZARR,
                        freeARR=freeARR,
                        SzARR=SzARR,
                        mARR=mARR,
                        )
