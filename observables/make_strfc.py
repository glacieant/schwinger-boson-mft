 #!/usr/bin/env python
import os
#import multiprocessing as mp
import itertools
import time

# MKL threads 
os.environ["MKL_DYNAMIC"] = "FALSE"
#os.environ["MKL_NUM_THREADS"] = "1" 
#os.environ["NUMEXPR_NUM_THREADS"] = "16" 
#os.environ["OMP_NUM_THREADS"] = "8" 

path = 'figs'
if not os.path.exists(path):
    os.makedirs(path)

import numpy as np
from numpy import linalg

#from scipy.linalg import eigvalsh
#from scipy.linalg import eigvals
#from scipy import optimize
#from scipy.linalg import null_space
#from scipy import integrate
#from scipy.linalg import cholesky
#from scipy.linalg import ldl
#from scipy.linalg import block_diag
#from scipy.misc import derivative as ddx
from scipy import ndimage
from scipy.linalg import pinv

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.figure as figure
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FormatStrFormatter

def compilevar(Q,mu):

    Q12ARR = np.full(3,Q[0])
    Q23ARR = np.full(3,Q[1])
    Q31ARR = np.full(3,Q[2])
    
    Y = np.hstack((Q12ARR,Q23ARR,Q31ARR))

    X = np.hstack((mu,Y,Y.conjugate()))

    return X

def NK_expand(NK,L):

    return int(NK/L), NK%L

def Ang(NK,L):
    
    n1, n2 = NK_expand(NK,L)

    k1 = 2*n1*np.pi/L
    k2 = 2*n2*np.pi/L
    k3 = -k1-k2
    
    PHASEab = np.exp([0,-1j*k3,1j*k2])
    PHASEbc = np.exp([1j*k3,0,-1j*k1])
    PHASEca = np.exp([-1j*k2,1j*k1,0])
    
    return np.array([PHASEab,PHASEbc,PHASEca])

def Ham(X,h,S,NKQ,NK,L):

    angK = Ang(NK,L)
    angKQ = Ang(NKQ,L)
    
    muARR = X[:3]

    Q12ARR = X[3:6]
    Q23ARR = X[6:9]
    Q31ARR = X[9:12]
    
    Q12ARRC = X[12:15]
    Q23ARRC = X[15:18]
    Q31ARRC = X[18:21]
    
    def A(angKQ,angK):

        A11 = muARR[0]
        A12 = 0
        A13 = 0
        A21 = 0
        A22 = muARR[1]
        A23 = 0
        A31 = 0
        A32 = 0
        A33 = muARR[2]
         
        return np.array([[A11,A12,A13],
            [A21,A22,A23],
            [A31,A32,A33]])
 
    def B(angKQ,angK):
        
        B12 = -Q12ARR.dot(angK[0])/2
        B13 = Q31ARR.dot(angKQ[2].conjugate())/2
        B21 = Q12ARR.dot(angKQ[0].conjugate())/2
        B23 = -Q23ARR.dot(angK[1])/2
        B31 = -Q31ARR.dot(angK[2])/2
        B32 = Q23ARR.dot(angKQ[1].conjugate())/2

        return np.array([[0,B12,B13],
            [B21,0,B23],
            [B31,B32,0]]
            )
    
    def AT(angKQ,angK):

        AC11 = muARR[0]
        AC12 = 0 
        AC13 = 0
        AC21 = 0
        AC22 = muARR[1]
        AC23 = 0
        AC31 = 0
        AC32 = 0
        AC33 = muARR[2]
         
        return np.array([[AC11,AC12,AC13],
            [AC21,AC22,AC23],
            [AC31,AC32,AC33]])

   
    def BT(angKQ,angK):

        BC12 = Q12ARRC.dot(angKQ[0])/2
        BC13 = -Q31ARRC.dot(angK[2].conjugate())/2
        BC21 = -Q12ARRC.dot(angK[0].conjugate())/2
        BC23 = Q23ARRC.dot(angKQ[1])/2
        BC31 = Q31ARRC.dot(angKQ[2])/2
        BC32 = -Q23ARRC.dot(angK[1].conjugate())/2

        return np.array([[0,BC12,BC13],
            [BC21,0,BC23],
            [BC31,BC32,0]]
            )
    
    
    o = np.zeros((3,3),dtype=float)
    ZeemanTerm = -(h*S/2)*np.identity(3)
    
    H = np.block([[A(angKQ,angK)+ZeemanTerm,o,o,B(angKQ,angK)],
                  [o,AT(angK.conjugate(),angKQ.conjugate()).T-ZeemanTerm,B(angK.conjugate(),angKQ.conjugate()).T,o],
                  [o,BT(angK.conjugate(),angKQ.conjugate()).T,A(angK.conjugate(),angKQ.conjugate()).T+ZeemanTerm,o],
                  [BT(angKQ,angK),o,o,A(angKQ,angK)-ZeemanTerm]])
    
    return H

def Vert(i,X,NKQ,NK,L):
    
    XM = np.zeros(X.size,dtype=complex)
    XM[i] = 1

    angK = Ang(NK,L)
    angKQ = Ang(NKQ,L)

    muARR = XM[:3]

    Q12ARR = XM[3:6]
    Q23ARR = XM[6:9]
    Q31ARR = XM[9:12]
    
    Q12ARRC = XM[12:15]
    Q23ARRC = XM[15:18]
    Q31ARRC = XM[18:21]
 
    def A(angKQ,angK):

        A11 = muARR[0]
        A12 = 0
        A13 = 0
        A21 = 0
        A22 = muARR[1]
        A23 = 0
        A31 = 0
        A32 = 0
        A33 = muARR[2]
         
        return np.array([[A11,A12,A13],
            [A21,A22,A23],
            [A31,A32,A33]])
 
    def B(angKQ,angK):
        
        B12 = -Q12ARR.dot(angK[0])/2
        B13 = Q31ARR.dot(angKQ[2].conjugate())/2
        B21 = Q12ARR.dot(angKQ[0].conjugate())/2
        B23 = -Q23ARR.dot(angK[1])/2
        B31 = -Q31ARR.dot(angK[2])/2
        B32 = Q23ARR.dot(angKQ[1].conjugate())/2

        return np.array([[0,B12,B13],
            [B21,0,B23],
            [B31,B32,0]]
            )
    
    def AT(angKQ,angK):

        AC11 = muARR[0]
        AC12 = 0 
        AC13 = 0
        AC21 = 0
        AC22 = muARR[1]
        AC23 = 0
        AC31 = 0
        AC32 = 0
        AC33 = muARR[2]
         
        return np.array([[AC11,AC12,AC13],
            [AC21,AC22,AC23],
            [AC31,AC32,AC33]])

   
    def BT(angKQ,angK):

        BC12 = Q12ARRC.dot(angKQ[0])/2
        BC13 = -Q31ARRC.dot(angK[2].conjugate())/2
        BC21 = -Q12ARRC.dot(angK[0].conjugate())/2
        BC23 = Q23ARRC.dot(angKQ[1])/2
        BC31 = Q31ARRC.dot(angKQ[2])/2
        BC32 = -Q23ARRC.dot(angK[1].conjugate())/2

        return np.array([[0,BC12,BC13],
            [BC21,0,BC23],
            [BC31,BC32,0]]
            )
    
    o = np.zeros((3,3),dtype=float)
    ZeemanTerm = o
    
    H = np.block([[A(angKQ,angK)+ZeemanTerm,o,o,B(angKQ,angK)],
                  [o,AT(angK.conjugate(),angKQ.conjugate()).T-ZeemanTerm,B(angK.conjugate(),angKQ.conjugate()).T,o],
                  [o,BT(angK.conjugate(),angKQ.conjugate()).T,A(angK.conjugate(),angKQ.conjugate()).T+ZeemanTerm,o],
                  [BT(angKQ,angK),o,o,A(angKQ,angK)-ZeemanTerm]])
    
    return H
 
def bogolSolve(X,Z,h,S,L):

    dim = int(Ham(X,h,S,0,0,L)[0].size/2)
    MK = np.zeros((L*L,2*dim,2*dim),dtype=complex)
    EK = np.zeros((L*L,2*dim),dtype=float)
    Sigma = np.diag(np.hstack((np.ones(dim),-np.ones(dim))))

    for NK in range(L*L):
        
        D = Ham(X,h,S,NK,NK,L)
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

        MK[NK] = T
        #EK[NK] = np.real(np.diag(TD.dot(D).dot(T)))
        EK[NK] = np.real(np.diag(TD.dot(D).dot(T)))
    
    i0 = np.argmin(EK[0,:dim])
    EK -= EK[0,i0]
    nc = 2*(L*np.linalg.norm(Z)/np.linalg.norm(MK[0,:,i0]))**2
    MK[0,:,i0] *= np.sqrt(1+nc)
    #MK[0,:,i0] = L*np.hstack((Z[:3],Z[3:6].conj(),Z[:3].conj(),Z[3:6]))
    MK[0,:,0:i0] = 0
    MK[0,:,i0+1:] = 0
   
    return MK, EK


def StructFactor(iS,ich,ih,SARR,CONFIG,hARR,QARR,muARR,ZARR,ncARR,L,LT,T,FREQ):

    N = L*L
    ETA = 4/L
    #ETA = 0.01
    
    S = SARR[iS]
    char = CONFIG[ich]
    h = hARR[ih]
    
    Z = ZARR[iS,ich,ih]

    mu = muARR[iS,ich,ih]
    Q = QARR[iS,ich,ih]
    nc = ncARR[iS,ich,ih]

    X = compilevar(Q,mu)
    
    iDim = X.size
    
    irange = np.arange(3, iDim)
    iqx = irange.size//2
    flucDim = iqx
    #D0inv = np.block([[np.zeros((iqx,iqx)),np.eye(iqx)],[np.eye(iqx),np.zeros((iqx,iqx))]])/2
    #D0inv = np.block([[np.zeros((iqx,iqx)),np.eye(iqx)],[np.eye(iqx),np.zeros((iqx,iqx))]])
    D0inv = np.eye(flucDim)/2

    dim = int(Ham(X,h,S,0,0,L)[0].size/2)
    #MK, EK, nb = bogolSolve(X,Z,h,S,nc,L)
    MK, EK = bogolSolve(X,Z,h,S,L)
    Sigma = np.hstack((np.ones(dim),-np.ones(dim)))

    eA = np.array([0,0])
    eB = np.array([1,0])
    eC = np.array([1/2,np.sqrt(3)/2])

    b1 = (2*np.pi/LT)*np.array([1,-1/np.sqrt(3)])
    b2 = (2*np.pi/LT)*np.array([0,2/np.sqrt(3)])
    
    u1 = np.sqrt(3)*np.array([0,1])
    u2 = np.sqrt(3)*np.array([-np.sqrt(3)/2,-1/2])
    
    KDIM = int(LT/2+LT/6+LT/3)
    MOM = np.zeros((KDIM),dtype=int) 
    
    SzQ = np.zeros((KDIM,2*dim,2*dim),dtype=complex)
    SyQ = np.zeros((KDIM,2*dim,2*dim),dtype=complex)
    SxQ = np.zeros((KDIM,2*dim,2*dim),dtype=complex)
    
    o = np.array([[0,0,0],
                  [0,0,0],
                  [0,0,0]])
    

    def spinVert(kA, kB, kC):
        

        preFact = np.diag([np.exp(1j*kA),np.exp(1j*kB),np.exp(1j*kC)])/(2*np.sqrt(3))
        
        JX = np.block([[o,preFact,o,o],
                             [preFact,o,o,o],
                             [o,o,o,preFact],
                             [o,o,preFact,o]
                             ])

        JY = np.block([[o,-1j*preFact,o,o],
                             [1j*preFact,o,o,o],
                             [o,o,o,-1j*preFact],
                             [o,o,1j*preFact,o]
                             ])

        JZ = np.block([[preFact,o,o,o],
                             [o,-preFact,o,o],
                             [o,o,preFact,o],
                             [o,o,o,-preFact]
                             ])


        return JX, JY, JZ

 
    
    for i in range(int(LT/2)):
        
        n1 = i
        n2 = 0
        
        k = n1*b1 +n2*b2
        
        kA = k.dot(eA)
        kB = k.dot(eB)
        kC = k.dot(eC)

        SxQ[i], SyQ[i], SzQ[i] = spinVert(kA,kB,kC) 
 
        k1 = k.dot(u1)
        k2 = k.dot(u2)
        m1 = int((L*k1)/(2*np.pi))
        m2 = int((L*k2)/(2*np.pi))
        NQ  = m1*L + m2
    
        MOM[i] = (NQ+16*N)%N
        
    for i in range(int(LT/6)):
        
        n1 = int(LT/2)+i
        n2 = 2*i
         
        k = n1*b1 +n2*b2
         
        kA = k.dot(eA)
        kB = k.dot(eB)
        kC = k.dot(eC)

        ip = int(LT/2)+i
        
        SxQ[ip], SyQ[ip], SzQ[ip] = spinVert(kA,kB,kC)

        k1 = k.dot(u1)
        k2 = k.dot(u2)
        m1 = int((L*k1)/(2*np.pi))
        m2 = int((L*k2)/(2*np.pi))
        NQ  = m1*L + m2
    
        MOM[int(LT/2)+i] = (NQ+16*N)%N
    
    for i in range(int(LT/3)):
        
        n1 = int(LT/2+LT/6)-2*i
        n2 = int(LT/3)-i
                 
        k = n1*b1 +n2*b2
         
        kA = k.dot(eA)
        kB = k.dot(eB)
        kC = k.dot(eC)
        
        ip = int(LT/2+LT/6)+i

        SxQ[ip], SyQ[ip], SzQ[ip] = spinVert(kA,kB,kC)

        k1 = k.dot(u1)
        k2 = k.dot(u2)
        m1 = int((L*k1)/(2*np.pi))
        m2 = int((L*k2)/(2*np.pi))
        NQ  = m1*L + m2

        MOM[int(LT/2+LT/6)+i] = (NQ+16*N)%N

    FNUM = FREQ.size
    ZRQ = np.zeros(FNUM,dtype=float)
    idFREQ = np.ones(FNUM,dtype=float)
 
    SFX = np.zeros((KDIM,FNUM),dtype=complex)
    SFX3 = np.zeros((KDIM,FNUM),dtype=complex)
        
    AMAT = np.block([[np.zeros((dim,dim)),np.ones((dim,dim))],
                     [np.ones((dim,dim)),np.zeros((dim,dim))]])
    
    CMAT = np.block([[np.zeros((dim,dim)),np.ones((dim,dim))],
                     [-np.ones((dim,dim)),np.zeros((dim,dim))]])
    
    def bose(E):

        #return 1/(np.exp(E/T)-1)
        return -np.heaviside(-E,1)

    for ik in range(KDIM):
           
        NQ = MOM[ik]
    
        #NQM = (N-NQ)%N

        #NKarr = np.unique(((np.array([-1,0,1,NQM-1,NQM,NQM+1])+2*N)%N))
        NKarr = np.unique([0])
        
        PI = np.zeros((FNUM,flucDim,flucDim), dtype=complex)
    
        LAMB_XL = np.zeros((FNUM,flucDim), dtype=complex)
        LAMB_XR = np.zeros((FNUM,flucDim), dtype=complex)
        LAMB_YL = np.zeros((FNUM,flucDim), dtype=complex)
        LAMB_YR = np.zeros((FNUM,flucDim), dtype=complex)
        LAMB_ZL = np.zeros((FNUM,flucDim), dtype=complex)
        LAMB_ZR = np.zeros((FNUM,flucDim), dtype=complex)

        #for NK in range(N):
        for NK in NKarr:
            
            GAMMA_KQK = np.zeros((flucDim,2*dim,2*dim),dtype=complex)
            GAMMA_KKQ = np.zeros((flucDim,2*dim,2*dim),dtype=complex)
       
            NKQ = int((NQ + NK)%N)

            for a in range(flucDim):

                i = irange[a]
                j = irange[a+flucDim]
        
                GAMMA_KKQ[a] = Vert(i,X,NK,NKQ,L)
                GAMMA_KQK[a] = Vert(j,X,NKQ,NK,L)
                                                                                                    
            AX = -np.outer(Sigma,Sigma)*np.subtract.outer(bose(Sigma*(EK[NKQ])),
                                                         bose(Sigma*(EK[NK])))
            BX = np.subtract.outer(Sigma*EK[NKQ],Sigma*EK[NK])
            Y = np.add.outer(ZRQ,AX)/np.add.outer(-(FREQ+1j*ETA),BX)

            #EX = np.add.outer(EK[NKQ],EK[NK])
            #Y = np.multiply.outer(idFREQ,AMAT)/(1e-20+np.multiply.outer(idFREQ,EX)+np.multiply.outer(-(FREQ+1j*ETA),CMAT))
          
            # Polarization function
        
            gamma_KKQ = np.tensordot(np.tensordot(GAMMA_KKQ,MK[NK].conj(),axes=(1,0)),MK[NKQ],axes=(1,0))
            gamma_KQK = np.tensordot(np.tensordot(GAMMA_KQK,MK[NKQ].conj(),axes=(1,0)),MK[NK],axes=(1,0))

            VERT_KKQ_KQK = np.einsum('aij,bji->abji',gamma_KKQ,gamma_KQK,optimize=True)

            PI += (1/(N))*np.tensordot(Y,VERT_KKQ_KQK,axes=([1,2],[2,3]))

            # Spin vertex
        
            JzQ = (MK[NKQ].conj().transpose()).dot(SzQ[ik]).dot(MK[NK])
            JxQ = (MK[NKQ].conj().transpose()).dot(SxQ[ik]).dot(MK[NK])
            JyQ = (MK[NKQ].conj().transpose()).dot(SyQ[ik]).dot(MK[NK])
            
            #  Spin-Polarization loop

            PSI_Z = Y*JzQ 
            LAMB_ZL += (1/N)*np.tensordot(PSI_Z,gamma_KKQ,axes=([1,2],[2,1]))
            PSI_ZC = Y*(JzQ.conj()) 
            LAMB_ZR += (1/N)*np.tensordot(PSI_ZC,gamma_KQK,axes=([1,2],[1,2]))

            PSI_X = Y*JzQ 
            LAMB_XL += (1/N)*np.tensordot(PSI_X,gamma_KKQ,axes=([1,2],[2,1]))
            PSI_XC = Y*(JzQ.conj()) 
            LAMB_XR += (1/N)*np.tensordot(PSI_XC,gamma_KQK,axes=([1,2],[1,2]))

            PSI_Y = Y*JzQ 
            LAMB_YL += (1/N)*np.tensordot(PSI_Y,gamma_KKQ,axes=([1,2],[2,1]))
            PSI_YC = Y*(JzQ.conj()) 
            LAMB_YR += (1/N)*np.tensordot(PSI_YC,gamma_KQK,axes=([1,2],[1,2]))

            # Large N structure factor

            SFX[ik,:] += (1/(2*N))*np.tensordot(Y,np.abs(JxQ)**2,axes=([1,2],[0,1]))
            SFX[ik,:] += (1/(2*N))*np.tensordot(Y,np.abs(JyQ)**2,axes=([1,2],[0,1]))
            SFX[ik,:] += (1/(2*N))*np.tensordot(Y,np.abs(JzQ)**2,axes=([1,2],[0,1]))

        # RPA fluctuation propagator
        
        DRPA = np.add.outer(ZRQ,D0inv) - PI
        GreensRPA = np.linalg.pinv(DRPA,rcond=ETA**2,hermitian=False)
        #GreensRPA = np.linalg.inv(DRPA)

        # 1/N Structure factor correction

        LAMB_XLR = np.einsum('wa,wb->wab',LAMB_XL,LAMB_XR,optimize=True)
        SFX3[ik,:] += -(1/2)*np.sum(LAMB_XLR*GreensRPA,axis=(1,2))
        LAMB_YLR = np.einsum('wa,wb->wab',LAMB_YL,LAMB_YR,optimize=True)
        SFX3[ik,:] += -(1/2)*np.sum(LAMB_YLR*GreensRPA,axis=(1,2))
        LAMB_ZLR = np.einsum('wa,wb->wab',LAMB_ZL,LAMB_ZR,optimize=True)
        SFX3[ik,:] += -(1/2)*np.sum(LAMB_ZLR*GreensRPA,axis=(1,2))


    return SFX, SFX3
  

## External Parameters
    
T = 0.01
pauli = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
leviC = np.array([[0,1],[-1,0]])
dt = 10**(-20)
#L = 384
L = 96
FMAX = 3

LT = int(np.sqrt(3*L*L))
#LT = L
if (LT % 6) != 0:
    LT += 6 - (LT % 6)

LE = int(LT/2+LT/6+LT/3)
FDIM = 1200
FREQ = np.linspace(0,FMAX,num=FDIM)

params = np.load('var1ch_data.npz')

SARR = params['SARR']
CONFIG = params['CONFIG']
hARR=params['hARR']
QARR=params['QARR']
ZARR=params['ZARR']
muARR=params['muARR']
freeARR=params['freeARR']
SzARR=params['SzARR']
ncARR=params['ncARR']

sfcARR=np.zeros((SARR.size,CONFIG.size,hARR.size,LE,FREQ.size),dtype=complex)
sfcARR3=np.zeros((SARR.size,CONFIG.size,hARR.size,LE,FREQ.size),dtype=complex)

#indxSARR = np.array([np.argwhere(SARR==0.25),np.argwhere(SARR==0.5),np.argwhere(SARR==0.75),np.argwhere(SARR==1.0)]).flatten()

if __name__=='__main__':

    for iS in range(SARR.size):
    #for iS in [1]:
        
        for ich in range(CONFIG.size):

            for ih in range(hARR.size):
                
                cfig, cx = plt.subplots()

                SFX, SFX3 = StructFactor(iS,ich,ih,SARR,CONFIG,hARR,QARR,muARR,ZARR,ncARR,L,LT,T,FREQ)
                sfcARR[iS,ich,ih,:] = SFX
                sfcARR3[iS,ich,ih,:] = SFX3



np.savez_compressed('sb_sfcdata.npz',
                    SARR=SARR,
                    CONFIG=CONFIG,
                    hARR=hARR,
                    sfcARR=sfcARR,
                    sfcARR3=sfcARR3,
                    L=L,
                    T=T,
                    FREQ=FREQ
                    )
