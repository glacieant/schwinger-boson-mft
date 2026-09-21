#!/usr/bin/env python
import os
import sys
import multiprocessing as mp
import itertools
import time
#import psutil
#import gc
import functools as func

# MKL threads 
#os.environ["MKL_DYNAMIC"] = "FALSE"
#os.environ["MKL_NUM_THREADS"] = "1"
#os.environ["NUMEXPR_NUM_THREADS"] = "1" 
#os.environ["OMP_NUM_THREADS"] = "1" 

# numpy and torch modules
import numpy as np
import torch

# scipy modules
from scipy.linalg import cholesky, eig, eigh
from scipy import optimize

#from mpi4py.futures import MPIPoolExecutor


def compilevar(Q,mu):

    Q12ARR = np.full(3,Q[0])
    Q23ARR = np.full(3,Q[1])
    Q31ARR = np.full(3,Q[2])
    
    Y = np.hstack((Q12ARR,Q23ARR,Q31ARR))

    X = np.hstack((mu,Y,Y.conjugate()))

    return X

def Ang(K,L):
    
    device = K.device

    # Reduced Brillouin zone momenta
    u1 = -np.sqrt(3)*torch.tensor([0.,1.],dtype=torch.float64,device=device)
    u2 = -np.sqrt(3)*torch.tensor([-np.sqrt(3)/2,-1/2],dtype=torch.float64,device=device)

    k1 = torch.matmul(K,u1)
    k2 = torch.matmul(K,u2)
    k3 = -k1-k2
    
    PHASEab = torch.exp(torch.tensor([0,-1j*k3,1j*k2],dtype=torch.complex128,device=device))
    PHASEbc = torch.exp(torch.tensor([1j*k3,0,-1j*k1],dtype=torch.complex128,device=device))
    PHASEca = torch.exp(torch.tensor([-1j*k2,1j*k1,0],dtype=torch.complex128,device=device))
    
    return torch.vstack((PHASEab,PHASEbc,PHASEca))

def Ham(X,h,S,KQ,K,L):

    device = X.device
    XM = X.type(torch.complex128)
    
    angK = Ang(K,L)
    angKQ = Ang(KQ,L)
    
    muARR = XM[:3]

    Q12ARR = XM[3:6]
    Q23ARR = XM[6:9]
    Q31ARR = XM[9:12]
    
    Q12ARRC = XM[12:15]
    Q23ARRC = XM[15:18]
    Q31ARRC = XM[18:21]
    
    def A(angKQ,angK):

        return torch.diag(muARR)

    def B(angKQ,angK):
        
        B12 = -torch.dot(Q12ARR,angK[0])/2
        B13 = torch.dot(Q31ARR,torch.conj(angKQ[2]))/2
        B21 = torch.dot(Q12ARR,torch.conj(angKQ[0]))/2
        B23 = -torch.dot(Q23ARR,angK[1])/2
        B31 = -torch.dot(Q31ARR,angK[2])/2
        B32 = torch.dot(Q23ARR,torch.conj(angKQ[1]))/2

        return torch.tensor([[0,B12,B13],
            [B21,0,B23],
            [B31,B32,0]],
            device=device
            )
    
    def AT(angKQ,angK):

        return torch.diag(muARR)
   
    def BT(angKQ,angK):

        BC12 = torch.dot(Q12ARRC,angKQ[0])/2
        BC13 = -torch.dot(Q31ARRC,torch.conj(angK[2]))/2
        BC21 = -torch.dot(Q12ARRC,torch.conj(angK[0]))/2
        BC23 = torch.dot(Q23ARRC,angKQ[1])/2
        BC31 = torch.dot(Q31ARRC,angKQ[2])/2
        BC32 = -torch.dot(Q23ARRC,torch.conj(angK[1]))/2

        return torch.tensor([[0,BC12,BC13],
            [BC21,0,BC23],
            [BC31,BC32,0]],
            device=device
            )
    
    
    o = torch.zeros((3,3),device=device)
    ZeemanTerm = -(h*S/2)*torch.eye(3,device=device)
    
    H = torch.vstack((torch.hstack((A(angKQ,angK)+ZeemanTerm,o,o,B(angKQ,angK))),
                  torch.hstack((o,AT(torch.conj(angK),torch.conj(angKQ)).T-ZeemanTerm,B(torch.conj(angK),torch.conj(angKQ)).T,o)),
                  torch.hstack((o,BT(torch.conj(angK),torch.conj(angKQ)).T,A(torch.conj(angK),torch.conj(angKQ)).T+ZeemanTerm,o)),
                  torch.hstack((BT(angKQ,angK),o,o,A(angKQ,angK)-ZeemanTerm))))
    
    return H

def Vert(i,XDIM,P,K,L,device):

    XM = torch.zeros(XDIM,device=device)
    XM[i] = 1

    return Ham(XM,0,0,P,K,L)

def BogEig(D):

    device = D.device
    dim = D.shape[0]//2
    Sigma = torch.diag(torch.hstack((torch.ones(dim,device=device),-torch.ones(dim,device=device)))).type(torch.complex128)
    
    # Smit et al. Bogoliubov solver (https://doi.org/10.1103/PhysRevB.101.054424)
    
    l, v = torch.linalg.eig(torch.matmul(Sigma,D))
    ls = torch.argsort(-torch.real(l))
    l = l[ls]
    v = v[:,ls]
    ls2 = torch.argsort(torch.real(l)[dim:])
    l = torch.hstack((l[:dim],l[dim:][ls2]))
    v = torch.hstack((v[:,:dim],v[:,dim:][:,ls2]))
    vnorm = torch.diag(torch.matmul(torch.t(torch.conj(v)),torch.matmul(Sigma,v)))
    T = v/torch.sqrt(torch.abs(vnorm))

    TD = torch.t(torch.conj(T))
    W = torch.real(torch.diag(torch.matmul(torch.matmul(TD,D),T)))

    return W, T

def computeSFC(iQ,device,X,h,S,L,
               aRange,astRange,D0inv,MOM,BZ,FREQ,ETA,T,
               SQ,BzChoice):

    start = time.time()

    X = torch.tensor(X,device=device)
    XDIM = X.shape[0]

    NS = L*L
    
    Q = torch.tensor(MOM[iQ],dtype=torch.float64,device=device)
    Q0 = torch.tensor([0,0],dtype=torch.float64,device=device)

    dim = Ham(X,h,S,Q,Q,L).shape[-1]//2
    Sigma = torch.hstack((torch.ones(dim,device=device),-torch.ones(dim,device=device)))

    def bose(E):

        return (1/(torch.exp(E/T)-1))
    
    SQX = torch.tensor(SQ[iQ],device=device)
    SQCX = torch.swapaxes(torch.conj(SQX),1,2)
   
    
    BZ = torch.tensor(BZ,device=device)
    FREQ = torch.tensor(FREQ,device=device)

    FNUM = FREQ.shape[0]
    D0inv = torch.tensor(D0inv,device=device)
    flucDim = aRange.shape[0]
    PI = torch.zeros((FNUM,flucDim,flucDim), dtype=torch.complex128,device=device)
    LAMB_L = torch.zeros((3,FNUM,flucDim), dtype=torch.complex128,device=device)
    LAMB_R = torch.zeros((flucDim,FNUM,3), dtype=torch.complex128,device=device)

    SFX = torch.zeros((2,FNUM),dtype=torch.complex128,device=device)
    
    if -Q not in BZ:
        # Brillouin zone cut
        if BzChoice == "momBEC":
            BZX = torch.vstack((Q0,-Q)) # BEC contribution
        else:
            BZX = torch.cat((BZ,-Q[None,...]),dim=0) # Brillouin zone
    else:
        if BzChoice == "momBEC":
            BZX = torch.tensor([[0,0]],dtype=torch.float64,device=device)
        else:
            BZX = BZ

    for K in BZX:

        #start = time.time()

        GAMMA_KQK = torch.zeros((flucDim,2*dim,2*dim),dtype=torch.complex128,device=device)
        GAMMA_KKQ = torch.zeros((flucDim,2*dim,2*dim),dtype=torch.complex128,device=device)

        KQ = K + Q

        for a in range(flucDim):

            i = astRange[a]
            j = aRange[a]
    
            GAMMA_KKQ[a] = Vert(i,XDIM,K,KQ,L,device)
            GAMMA_KQK[a] = Vert(j,XDIM,KQ,K,L,device)
    
        
        D = Ham(X,h,S,K,K,L)
        EK, MK = BogEig(D)

        D = Ham(X,h,S,KQ,KQ,L)
        EKQ, MKQ = BogEig(D)

        XKQ = (Sigma*EKQ)[:,None]
        XK = (Sigma*EK)[None,:]

        AX = - torch.outer(Sigma,Sigma)*(bose(XKQ)-bose(XK))
        BX = (XKQ-XK)[None,...]
        FX = -(FREQ+1j*ETA)[:,None,None]
        eFX = torch.exp((1/L)*FX*torch.diag(Sigma))
        FX = FX*eFX
        gg_KQK = AX/(FX+BX)
        
        # Polarization function

        gamma_KKQ = torch.einsum("ca,icb,bd->iad",torch.conj(MK),GAMMA_KKQ,MKQ)
        gamma_KQK = torch.einsum("ca,icb,bd->iad",torch.conj(MKQ),GAMMA_KQK,MK)

        PI += (1/NS)*torch.einsum("iab,wab,jba->wij",gamma_KQK,gg_KQK,gamma_KKQ)

        # Spin vertices
        
        JQ = torch.einsum('ca,xcb,bd->xad',torch.conj(MKQ),SQX,MK)
        JQM = torch.einsum('ca,xcb,bd->xad',torch.conj(MK),SQCX,MKQ)

        SFX[0] += (1/NS)*torch.einsum('xab,wab,xba->w',JQ,gg_KQK,JQM)

        LAMB_L += (1/NS)*torch.einsum("xab,wab,iba->xwi",JQ,gg_KQK,gamma_KKQ)
        LAMB_R += (1/NS)*torch.einsum("iab,wab,xba->iwx",gamma_KQK,gg_KQK,JQM)

        #end = time.time()
        #print(K,"done",end-start)
        #sys.stdout.flush()
        
       
    # RPA fluctuation propagator

    DRPA = D0inv - PI
    GreensRPA =  torch.linalg.inv(DRPA)

    # 1/N Structure factor correction

    SFX[1] = torch.einsum('xwi,wij,jwx->w',LAMB_L,GreensRPA,LAMB_R)

    end = time.time()
    
    print(iQ,"complete!",end-start)
    sys.stdout.flush()

    if device == 'cpu':

        TFX = SFX.numpy()

    else:

        TFX = SFX.cpu().numpy()

    return TFX/4

def BZcutVert(L,LT):
    
    # Reciprocal lattice vectors
    b1 = (2*np.pi/LT)*np.array([1,-1/np.sqrt(3)])
    b2 = (2*np.pi/LT)*np.array([0,2/np.sqrt(3)])
    
    # Sublattice displacement vectors
    eA = np.array([0,0])
    eB = np.array([1,0])
    eC = np.array([1/2,np.sqrt(3)/2])
   
    KDIM = int(LT/2+LT/6+LT/3)
    MOM = np.zeros((KDIM,2),dtype=float) 
    
    J = np.zeros((KDIM,3,12,12),dtype=complex)
    
    o = np.array([[0,0,0],
                  [0,0,0],
                  [0,0,0]])

    # High symmetry points
    
    SYM = {
    "GAMMA": np.array([0,0]),
     "M": np.array([0,2*np.pi/np.sqrt(3)]),
     "K": np.array([2*np.pi/3,2*np.pi/np.sqrt(3)]),
     "KP": np.array([np.pi/3,np.pi/np.sqrt(3)])
    }
    
    PATH = ["GAMMA","M","K","GAMMA"] 

    ip = 0

    for i in range(LT//2):
        
        n1 = 0
        n2 = i
        
        k = n1*b1 +n2*b2
        
        kA = k.dot(eA)
        kB = k.dot(eB)
        kC = k.dot(eC)
       
        preFact = np.diag([np.exp(1j*kA),np.exp(1j*kB),np.exp(1j*kC)])/(2*np.sqrt(3))
    
        J[ip,0] = np.block([[o,preFact,o,o],
                             [preFact,o,o,o],
                             [o,o,o,preFact],
                             [o,o,preFact,o]
                             ])

        J[ip,1] = np.block([[o,-1j*preFact,o,o],
                             [1j*preFact,o,o,o],
                             [o,o,o,-1j*preFact],
                             [o,o,1j*preFact,o]
                             ])

        J[ip,2] = np.block([[preFact,o,o,o],
                             [o,-preFact,o,o],
                             [o,o,preFact,o],
                             [o,o,o,-preFact]
                             ])
           
        
        MOM[ip] = k
        ip += 1


    for i in range(LT//6):
        
        n1 = 2*i
        n2 = LT//2 + i
         
        k = n1*b1 +n2*b2
         
        kA = k.dot(eA)
        kB = k.dot(eB)
        kC = k.dot(eC)
       
        preFact = np.diag([np.exp(1j*kA),np.exp(1j*kB),np.exp(1j*kC)])/(2*np.sqrt(3))
    
        J[ip,0] = np.block([[o,preFact,o,o],
                             [preFact,o,o,o],
                             [o,o,o,preFact],
                             [o,o,preFact,o]
                             ])

        J[ip,1] = np.block([[o,-1j*preFact,o,o],
                             [1j*preFact,o,o,o],
                             [o,o,o,-1j*preFact],
                             [o,o,1j*preFact,o]
                             ])

        J[ip,2] = np.block([[preFact,o,o,o],
                             [o,-preFact,o,o],
                             [o,o,preFact,o],
                             [o,o,o,-preFact]
                             ])
       
        MOM[ip] = k
        ip += 1

    for i in range(LT//3):
        
        n1 = LT//3-i
        n2 = LT//2+LT//6-2*i
                 
        k = n1*b1 +n2*b2
         
        kA = k.dot(eA)
        kB = k.dot(eB)
        kC = k.dot(eC)
       
        preFact = np.diag([np.exp(1j*kA),np.exp(1j*kB),np.exp(1j*kC)])/(2*np.sqrt(3))
    
        J[ip,0] = np.block([[o,preFact,o,o],
                             [preFact,o,o,o],
                             [o,o,o,preFact],
                             [o,o,preFact,o]
                             ])

        J[ip,1] = np.block([[o,-1j*preFact,o,o],
                             [1j*preFact,o,o,o],
                             [o,o,o,-1j*preFact],
                             [o,o,1j*preFact,o]
                             ])

        J[ip,2] = np.block([[preFact,o,o,o],
                             [o,-preFact,o,o],
                             [o,o,preFact,o],
                             [o,o,o,-preFact]
                             ])

        MOM[ip] = k
        ip += 1

    return MOM, J

def StructFactor(iS,ich,ih,S,h,Q,mu,Z,nc,L,T,MOM,BZ,FREQ,ETA,SQ,mode,BzChoice):

    print(SARR[iS],CONFIG[ich],hARR[ih],"begins!")
    sys.stdout.flush()
    
    NS = L*L

    # Variational solution
    X = compilevar(Q,mu)
    XDIM = X.size
    
    # Symmetry breaking field

    def f(hx):

        K0 = torch.tensor([0.,0.],dtype=torch.float64,device='cpu')
        XP = torch.tensor(X,device='cpu')
        hx = torch.tensor(hx,device='cpu')
        M = Ham(XP,h-hx,S,K0,K0,L)
        dim = M.shape[0]//2
        Ones = torch.ones(dim,dtype=torch.float64,device='cpu')
        Sigma = torch.diag(torch.hstack((Ones,-Ones)))

        W, V = BogEig(M)

        return np.abs(torch.min(W).item())

    if nc > 0:
        hsym = (optimize.minimize(f,0).x)[0] + 1/NS # Zero mode regulator 
    else:
        hsym = 0 # spin liquid

    # Fluctuation propagator #
    irange = np.arange(XDIM)
    D0inv = np.zeros((XDIM,XDIM),dtype=float)
    
    flucDim = irange.shape[0]
    iqx = (flucDim-3)//2
    D0inv[3:,3:] = np.vstack((np.hstack((np.zeros((iqx,iqx)),
                                  np.eye(iqx))),
                       np.hstack((np.eye(iqx),
                                  np.zeros((iqx,iqx))))))/2
    aRange = irange
    astRange = irange
    
    KNUM = MOM.shape[0]
    FNUM = FREQ.shape[0]

    iMOM = np.arange(KNUM)

    if mode == "gpu":
        nGPU = torch.cuda.device_count()
        devices = list([f'cuda:{id}' for id in range(nGPU)])
    elif mode == "mps":
        devices = list(['mps'])
    else:
        devices = list(['cpu'])


    exr = zip(iMOM,
            itertools.cycle(devices),
            itertools.repeat(X),
            itertools.repeat(h-hsym),# Symmetry breaking field
            itertools.repeat(S),
            itertools.repeat(L),
            itertools.repeat(aRange),
            itertools.repeat(astRange),
            itertools.repeat(D0inv),
            itertools.repeat(MOM),
            itertools.repeat(BZ),
            itertools.repeat(FREQ),
            itertools.repeat(ETA),
            itertools.repeat(T),
            itertools.repeat(SQ),
            itertools.repeat(BzChoice),
            )

    maxProc = 8*len(devices)
    if mode == "cluster":
        with MPIPoolExecutor() as executor:
            res = np.array(list(executor.starmap(computeSFC,exr)))
    else:
        with mp.Pool(processes=maxProc) as pool:
            res = np.array(pool.starmap(computeSFC,exr,chunksize=1))
    #res = np.array([computeSFC(*ex) for ex in exr])
    

    SFX = res[:,0,:]
    SFX3 = res[:,1,:]

    return SFX, SFX3
  

## External Parameters
    
pauli = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
leviC = np.array([[0,1],[-1,0]])
dt = 10**(-20)
#L = 384
L = 60
T = 0.02
FMAX = 3
FDIM = 3000

#mode = "cluster"
#mode = "cpu"
mode = "gpu"
#mode = "mps"

# momenta grid
#LT = L
#LT = 8*L
LT = 384

# Spinon broadening factor
ETA = 4/LT # smoother
#ETA = T  

# momBEC = condensate momenta, fullBZ = full Brillouin zone
#BzChoice = "momBEC"
BzChoice = "fullBZ"

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

#indxSARR = np.array([np.argwhere(SARR==0.25),np.argwhere(SARR==0.5),np.argwhere(SARR==0.75),np.argwhere(SARR==1.0)]).flatten()

if __name__=='__main__':
    
    MOM, SQ = BZcutVert(L,LT)
    
    # Reduced Brillouin zone
    u1 = -np.sqrt(3)*np.array([0,1])
    u2 = -np.sqrt(3)*np.array([-np.sqrt(3)/2,-1/2])
   
    b1 = (1/L)*np.array([2*np.pi/3,-2*np.pi/np.sqrt(3)])
    b2 = (1/L)*np.array([4*torch.pi/3,0])
    BZ = []
    LHF = L//2
    for n1 in range(-LHF,LHF):
        for n2 in range(-LHF,LHF):
            BZ.append(n1*b1+n2*b2)

    BZ = np.array(BZ)

    FREQ = np.linspace(0,FMAX,num=FDIM)
    
    sfcARR=np.zeros((SARR.size,CONFIG.size,hARR.size,MOM.shape[0],FREQ.size),dtype=complex)
    sfcARR3=np.zeros((SARR.size,CONFIG.size,hARR.size,MOM.shape[0],FREQ.size),dtype=complex)
  
    for iS in range(SARR.size):
        
        for ich in range(CONFIG.size):

            for ih in range(hARR.size):
                
                S = SARR[iS]
                char = CONFIG[ich]
                h = hARR[ih]
    
                Z = ZARR[iS,ich,ih]

                mu = muARR[iS,ich,ih]
                Q = QARR[iS,ich,ih]
                nc = ncARR[iS,ich,ih]

                SFX, SFX3 = StructFactor(iS,ich,ih,S,h,Q,mu,Z,nc,L,T,MOM,BZ,FREQ,ETA,SQ,mode,BzChoice)
                
                sfcARR[iS,ich,ih,:] = SFX
                sfcARR3[iS,ich,ih,:] = SFX3

    np.savez_compressed('sb_sfcdata.npz',
                        SARR=SARR,
                        CONFIG=CONFIG,
                        hARR=hARR,
                        sfcARR=sfcARR,
                        sfcARR3=sfcARR3,
                        L=L,
                        LT=LT,
                        T=T,
                        FREQ=FREQ,
                        ETA=ETA
                        )
