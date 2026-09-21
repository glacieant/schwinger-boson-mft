#!/usr/bin/env python
import os
import sys
import multiprocessing as mp
import itertools
import time

import numpy as np
from numpy import linalg

import cython_en1ch

#from mpi4py import MPI

#mode = "cluster"
mode = "pc"

def compilevar(mu,Q):

    X = mu

    for q in Q:

        X = np.append(X,np.full(3,q))

    for q in Q:

        X = np.append(X,np.full(3,np.conj(q)))

    return X

def expandvar(X):

    mu = X[:3]
    
    Q12 = X[3:6]
    Q23 = X[6:9]
    Q31 = X[9:12]
    
    QC12 = X[12:15]
    QC23 = X[15:18]
    QC31 = X[18:21]
    
    return mu, Q12, Q23, Q31, QC12, QC23, QC31

def Ham(X,NK,L):

    mu, Q12, Q23, Q31, QC12, QC23, QC31 = expandvar(X)
    
    n1 = int(NK/L)
    n2 = NK%L

    k1 = 2*n1*np.pi/L
    k2 = 2*n2*np.pi/L
    k3 = -k1-k2
    
    PHASEab = np.exp([0,-1j*k3,1j*k2])
    PHASEbc = np.exp([1j*k3,0,-1j*k1])
    PHASEca = np.exp([-1j*k2,1j*k1,0])
    
    B12 = -Q12.dot(PHASEab)/2
    B13 = Q31.dot(PHASEca.conjugate())/2
    B21 = Q12.dot(PHASEab.conjugate())/2
    B23 = -Q23.dot(PHASEbc)/2
    B31 = -Q31.dot(PHASEca)/2
    B32 = Q23.dot(PHASEbc.conjugate())/2

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

def bogolSolve(X,h,S,Z,L):
    
    Sigma = np.diag([1,1,1,-1,-1,-1])
    MK = np.zeros((L*L,6,6),dtype=complex)
    EK = np.zeros((L*L,6),dtype=float)

    for NK in range(L*L):
    
        #D = ZeemanHam(X,h,S,NK,L)
        D = Ham(X,NK,L)
        
        ## Colpa method ##

        try:

            #H = cholesky(D)
            H = np.linalg.cholesky(D).conj().T
            l, U = np.linalg.eigh(H.dot(Sigma).dot(H.conj().transpose()))
            lax = np.argsort(-l)
            U[:] = U[:,lax]
            l = np.diag(l[lax])
            W = Sigma.dot(l)
            Hinv = np.linalg.inv(H)
            T = Hinv.dot(U).dot(np.sqrt(W))
            TD = T.conj().transpose()
        
        ## Wessel-Milat method ##

        except:
            
            l, V = np.linalg.eig(Sigma.dot(D))
            LL = (V.conj().transpose()).dot(Sigma).dot(V)
            r, U = np.linalg.eigh(LL)
            rax = np.argsort(-r)
            rhf = np.diag(1/np.sqrt(np.abs(r[rax])))
            U[:] = U[:,rax]
            T = V.dot(U).dot(rhf)
            TD = T.conj().transpose()

        #EQX = np.diag(TD@D@T).real
        
        EQX = np.diag(np.abs(TD@D@T))
        upordr = np.argsort(EQX[:3])
        dwnordr = np.argsort(EQX[3:])
        ordr = np.hstack([upordr,dwnordr]
                         )
        
        EQX = EQX[ordr]
        T  = T[:,ordr]
        
        MK[NK] = T
        EK[NK] = EQX
    
    #i0 = np.argmin(EK[0,:3])
    #E0 = EK[0,i0]
    #EK[:] -= E0
    MK[0,:] = 0

    return MK, EK

def dEN(X,h,S,Z,MK,EK,NQ,L,NSYS):

    Sigma = np.array([1, 1, 1, -1, -1, -1],dtype=float)

    iDim = X.size
    
    irange = np.arange(3, iDim)
    iqx = irange.size//2
    flucDim = 2*iqx

    #z1, z2, z3  = Zravel(Z)

    #kappa = np.array([np.linalg.norm(z1)**2,np.linalg.norm(z2)**2,np.linalg.norm(z3)**2])

    #SE = cython_en1ch.loopEval(X,h,S,Z,L,flucDim,irange,MK,EK,Sigma,NQ,NSYS).real
    SE = cython_en1ch.loopEval(X,h,S,L,flucDim,irange,MK,EK,Sigma,NQ,NSYS).real

    return SE


def eig(X,h,S,NK,L):
    
    D = Ham(X,NK,L)
    #D = ZeemanHam(X,h,S,NK,L)
    Sigma = np.diag([1,1,1,-1,-1,-1])
    
    l, V = np.linalg.eig(Sigma@D)

    EQX = np.trace(np.abs(l)-np.real(D))/2

    return EQX

def eigSum(X,EK,h,S,L):
    
    YS = 0
   
    for NK in range(1,L*L):

        YS += eig(X,h,S,NK,L)
        #YS += np.sum(EK[NK])/2

    return YS/(L**2)
    
def clen(X,h,S,Z,L):

    mu, Q12, Q23, Q31, QC12, QC23, QC31 = expandvar(X)

    Q = np.array([Q12[0],Q23[0],Q31[0]])

    return ((3/2)*np.linalg.norm(Q)**2
            + np.vdot(Z,np.dot(ZeemanHam(X,h,S,0,L),Z)).real
            - np.sum(mu)*(2*S)
            ).real

def FreeEN(X,EK,h,S,Z,L):
    
    EQ = eigSum(X,EK,h,S,L)
    
    ECL = clen(X,h,S,Z,L)

    return ECL + EQ + (9/4)*((2*S)**2)

def polEigSum2(X,MK,EK,h,S,Z,L):

    #LT = L//2
    LT = L

    NSYS = LT * LT

    #NRANGE = np.arange(NSYS*NSYS)
    NRANGE = np.arange(NSYS)


    SE = 0

    #SE += cndEigSum2(X,h,S,Z,MK,EK,L,NSYS)

    #MK[0,:] = 0

    kARR = zip(itertools.repeat(X),itertools.repeat(h),itertools.repeat(S),
               itertools.repeat(Z),itertools.repeat(MK),
               itertools.repeat(EK),
               NRANGE,itertools.repeat(L),
               itertools.repeat(NSYS),
               )

    with mp.Pool() as pool:
        res = pool.starmap(dEN,kARR)
    
    #SE += 2*sum(res)/(L**2)
    SE += sum(res)/(L**2)

    return SE

def FlucEN(iS, ich, ih, mu, Q, h, S, Z, nc, LARR):

    X = compilevar(mu, Q)

    EnARR0 = []
    EnARR1 = []

    for L in LARR:

        MK, EK = bogolSolve(X,h,S,Z,L)
        
        if fit == "1/L":
            En0 = (1/3)*FreeEN(X,EK,h,S,Z,L)
        else:
            En0 = freeARR[iS,ich,ih]
        En1 = (1/3)*polEigSum2(X,MK,EK,h,S,Z,L)

        EnARR0.append(En0)
        EnARR1.append(En1)
    
    if fit == "1/L":
        try:
            coef0 = np.polynomial.polynomial.polyfit(1/LARR,EnARR0,deg=1)
        except:
            coef0 = [EnARR0[-1]]
        try:
            coef1 = np.polynomial.polynomial.polyfit(1/LARR,EnARR1,deg=1)
        except:
            coef1 = [EnARR1[-1]]
    else:
        coef0 = [EnARR0[-1]]
        coef1 = [EnARR1[-1]]

    print(SARR[iS],CONFIG[ich],hARR[ih],coef0[0],coef1[0],coef0[0]+coef1[0],errARR[iS,ich,ih])
    #print(SARR[iS],CONFIG[ich],hARR[ih],freeARR[iS,ich,ih],coef1[0],freeARR[iS,ich,ih]+coef1[0],errARR[iS,ich,ih])
    sys.stdout.flush()

    return iS, ich, ih, coef0[0], coef0[0] + coef1[0]

## External Parameters
    
pauli = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
leviC = np.array([[0,1],[-1,0]])
#LARR = np.array([96])
#LARR = np.array([24,30,36])
#LARR = np.array([120])
#LARR = np.array([144])
LARR = np.array([36])
#fit = "1/L"
fit = "last"

## Simulation Parameters
    
params = np.load('var1ch_data.npz')
SARR = params['SARR']
CONFIG = params['CONFIG']
hARR=params['hARR']
QARR=params['QARR']
ZARR=params['ZARR']
SzARR=params['SzARR']
muARR=params['muARR']
ncARR=params['ncARR']
freeARR=params['freeARR']
errARR=params['errARR']
freeARR2=np.zeros(freeARR.shape,dtype=float)

#indxSARR = np.array([np.argwhere(SARR==0.25),np.argwhere(SARR==0.5),np.argwhere(SARR==0.75),np.argwhere(SARR==1.0)]).flatten()
#indxSARR = np.array([np.argwhere(SARR==0.25),np.argwhere(SARR==0.5)]).flatten()
#indxSARR = np.array([np.argwhere(SARR==0.5)]).flatten()
indxSARR = range(SARR.size)


if __name__=='__main__':

    if mode == "cluster":
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    ex_arr = []

    #for iS in range(SARR.size):
    for iS in indxSARR:

        #for ih in range(hARR.size//6,hARR.size):
        for ih in range(hARR.size):
            
            for ich in range(CONFIG.size):
            
            
                mu = muARR[iS, ich, ih]
                Q = QARR[iS, ich, ih]
                Z = ZARR[iS, ich, ih]
                nc = ncARR[iS, ich, ih]
                
                h = hARR[ih]
                S = SARR[iS]
                
                ex_arr.append((iS, ich, ih, mu, Q, h, S, Z, nc, LARR))

    if mode == "cluster":

        num_tuples = len(ex_arr)
        chunk_size = num_tuples//size

        start = rank * chunk_size
        end = start + chunk_size

        if rank == size - 1:
            end = num_tuples

        results = []
        for i in range(start, end):

            result = FlucEN(*ex_arr[i])
            results.append(result)

        all_results = comm.gather(results, root=0)

        if rank != 0:
            exit()

        res = []
        for result in all_results:
            res.extend(result)

    elif mode == "pc":

        #with mp.Pool() as pool:
        #    res = pool.starmap(FlucEN,ex_arr)

        res = [FlucEN(*exr) for exr in ex_arr]


    for ires in range(len(res)):
        
        iS, ich, ih, En0, En = res[ires]
        
        freeARR[iS, ich, ih] = En0
        freeARR2[iS, ich, ih] = En


    np.savez_compressed('sb_fluctdata_contour.npz',
                            SARR=SARR,
                            CONFIG=CONFIG,
                            hARR=hARR,
                            muARR=muARR,
                            QARR=QARR,
                            ZARR=ZARR,
                            ncARR=ncARR,
                            freeARR=freeARR,
                            freeARR2=freeARR2,
                            indxSARR = indxSARR,
                            LARR = LARR
                            )
