#!/usr/bin/env python
import os
import multiprocessing as mp
import itertools
import time
import datetime
import sys

# mpi4py modules
#from mpi4py import MPI
#from mpi4py.futures import MPIPoolExecutor

#mode = "cluster" # for CPU clusters
mode = "cpu" # cpu machines
#mode = "gpu" # for machines with nvidia cuda cores

# numpy and torch modules
import numpy as np
import torch
from torch.autograd import grad
from torch.autograd import gradcheck
import torch.optim as optim

# scipy modules
from scipy import optimize
from scipy import integrate
from scipy.linalg import cholesky, solve_triangular

def compilevar(mu,Q,Z,iy):

    X = np.hstack((mu,Q.real,Q.imag,Z.real,Z.imag,iy))

    return X

def Zravel(Z):
    
    z1 = np.array([Z[0],Z[3].conj()])
    z2 = np.array([Z[1],Z[4].conj()])
    z3 = np.array([Z[2],Z[5].conj()])

    return z1, z2, z3

def expandvar(X):
   
    mu = X[:3]
    Q = X[3:6] + 1j*X[6:9]
    Z = X[9:15] + 1j*X[15:21]
    iy = X[21]

    return mu, Q, Z, iy

def Ham(X, NKarr, L, device):

    mu, Q, Z, iy = expandvar(X)
    
    n1 = NKarr // L
    n2 = NKarr % L
    
    k1 = 2 * n1 * torch.pi / L
    k2 = 2 * n2 * torch.pi / L
    k3 = -k1 - k2

    PHASEab = 1 + torch.exp(-1j * k3) + torch.exp(1j * k2)
    PHASEbc = torch.exp(1j * k3) + 1 + torch.exp(-1j * k1)
    PHASEca = torch.exp(-1j * k2) + torch.exp(1j * k1) + 1

    J = torch.zeros((NKarr.shape[0],3,3,3),dtype=torch.complex128,device=device)
    
    J[:,0,0,1] = PHASEab
    J[:,0,1,0] = -torch.conj(PHASEab)

    J[:,1,1,2] = PHASEbc
    J[:,1,2,1] = -torch.conj(PHASEbc)

    J[:,2,0,2] = -torch.conj(PHASEca)
    J[:,2,2,0] = PHASEca


    B = - torch.tensordot(Q,J,dims=([0],[1]))/2
    BHC = torch.transpose(torch.conj(B),1,2)
    A = torch.zeros_like(B,dtype=torch.complex128,device=device)
    A[:] = torch.diag(mu)
    C1 = torch.cat((A,B),dim=2)
    C2 = torch.cat((BHC,A),dim=2)
    
    D = torch.cat((C1,C2),dim=1)

    return D


def ZeemanHam(X, NKarr, h, S, L,device):

    Sigma = torch.diag(torch.tensor([1, 1, 1, -1, -1, -1],dtype=torch.complex128,device=device))
    
    return Ham(X, NKarr, L, device) - (h * S / 2) *Sigma 

def BogEig(X, NKarr, h, S, L,device):
    
    MK = ZeemanHam(X, NKarr, h, S, L,device)
    dim = MK.shape[-1] // 2

    Sigma = torch.diag(
            torch.hstack((torch.ones(dim,dtype=torch.complex128,device=device),
            -torch.ones(dim,dtype=torch.complex128,device=device)))
            )

    SMK = torch.swapaxes(torch.tensordot(Sigma,MK,dims=([1],[1])),0,1)
    LK, VK = torch.linalg.eig(torch.matmul(Sigma, MK))
    LAX = torch.argsort(-torch.real(LK),dim=1)
    LK = torch.take_along_dim(LK,LAX,dim=1)
    WK = torch.matmul(LK,Sigma) # Sigma^T = Sigma

    return WK


def enQBZ(X, NKarr, h, S, L,device,solver):

    WK = BogEig(X, NKarr, h, S, L,device)
    HK = torch.diagonal(Ham(X, NKarr, L, device),dim1=1,dim2=2)
    
    EQ = torch.sum(torch.real(WK-HK)) / 2

    return EQ

def enQ(X, h, S, L, device, solver):

    NS = L * L

    # Brillouin zone momentum array
    BZ = torch.arange(1, NS,device=device)
    YS = enQBZ(X,BZ,h,S,L,device,solver)/NS

    return YS

def denQ(X, h, S, L, device, solver):

    NS = L * L
    
    fEQ = enQ(X,h,S,L,device,solver)
    dEQ = grad(fEQ,X)[0]
    

    return dEQ

def enCL(X, h, S, L,device):

    NK0 = torch.tensor([0],device=device)
    W0 = torch.flatten(BogEig(X, NK0, h, S, L,device))

    D = ZeemanHam(X, NK0, h, S, L,device)[0]

    mu, Q, Z, iy = expandvar(X)

    # Classical energy
    ECL = torch.real((3/2) * torch.norm(Q)**2 
                + torch.matmul(torch.conj(Z),torch.matmul(D, Z)) 
                - torch.sum(mu) * (2 * S))
    
    return ECL


def denCL(X, h, S, L,device):
    
    fECL = enCL(X,h,S,L,device)
    dECL = grad(fECL,X)[0]

    return dECL

def FreeEN(X,h,S,L,T,device,solver):

    XGPU = torch.tensor(X,dtype=torch.float64,device=device)
    ECL = enCL(XGPU,h,S,L,device)
    EQ = enQ(XGPU,h,S,L,device,solver)
    #ETHRM = enTHRM(XGPU,h,S,L,T,device)

    ETOT = ECL + EQ + (9/4)*((2*S)**2)
    #ETOT = ECL + EQ + ETHRM + (9/4)*((2*S)**2)

    if device == 'cpu':
        XE = ETOT.numpy()
    else:
        XE = ETOT.cpu().numpy()

    return XE

def stationary(X,h,S,L,T,device,solver):
    
    ### self-consistent equations ###

    XGPU = torch.tensor(X,dtype=torch.float64,device=device,requires_grad=True)
    #dE = denQ(XGPU,h,S,L,device) + denTHRM(XGPU,h,S,L,T,device) + denCL(XGPU,h,S,L,device)
    dE = denQ(XGPU,h,S,L,device,solver) + denCL(XGPU,h,S,L,device)

    if device == 'cpu':
        XE = dE.numpy()
    else:
        XE = dE.cpu().numpy()

    return XE

def rotSpinor(theta,phi,chi):

    return np.array([[np.cos(theta/2)*np.exp(1.0j*phi/2)*np.exp(1.0j*chi/2),
        np.sin(theta/2)*np.exp(-1.0j*phi/2)*np.exp(1.0j*chi/2)],
        [-np.sin(theta/2)*np.exp(1.0j*phi/2)*np.exp(-1.0j*chi/2),
            np.cos(theta/2)*np.exp(-1.0j*phi/2)*np.exp(-1.0j*chi/2)
            ]])

def SolveMFSL(S):

    def gamma(k1,k2):
        
        return (np.sin(k1)+np.sin(k2)+np.sin(-(k1+k2)))

    def Omega(k1,k2,r):

        return np.sqrt(1-(r*gamma(k1,k2))**2)
    
    def integ1(k1,k2,r):

        return (gamma(k1,k2)**2)/Omega(k1,k2,r)
 
    def integ2(k1,k2,r):

        return 1/Omega(k1,k2,r)

    def f(r):
            
        X = integrate.dblquad(integ2,0,2*np.pi,0,2*np.pi,args=[r])[0]/(2*np.pi)**2 
        Y = -2*S - 1 + X

        return Y

    try:

        r = optimize.bisect(f,0,2/(3*np.sqrt(3)))
        
        X = integrate.dblquad(integ1,0,2*np.pi,0,2*np.pi,args=[r])[0]/(2*np.pi)**2 
        mu = X/3
        Q = mu*r
        gap = np.sqrt(mu**2-27*(Q**2)/4) 
        EN = (mu*integrate.dblquad(Omega,0,2*np.pi,0,2*np.pi,args=[r])[0]/(2*np.pi)**2 
                +3*(np.abs(Q)**2)/2
              -mu*(1+2*S)
                +(3/4)*((2*S)**2)
              )

    except:

        mu = np.nan
        Q = np.nan
        gap = 0
        EN = np.nan

    return mu, Q, gap, EN

def critS(h,L):

    X = np.array([1,0])

    def gamma(k1,k2):
        
        return (np.sin(k1)+np.sin(k2)+np.sin(-(k1+k2)))

    def omega(k1,k2,mu,Q):

        return np.sqrt((mu**2)-((Q*gamma(k1,k2))**2))
    
    def integ1(k1,k2,mu,Q):

        return 1/omega(k1,k2,mu,Q)
    
    def integ2(k1,k2,mu,Q):

        return mu/omega(k1,k2,mu,Q)

    def momsum(f,*args):

        Y = 0

        for NK in range(L*L):

            k1 = (NK/L)*(2*np.pi/L)
            k2 = (NK%L)*(2*np.pi/L)

            Y += f(k1,k2,*args)

        return Y/(L*L)


    def sol(X):

        Q = X[0]
        S = X[1]

        mu = np.sqrt((h*S)**2+27*(Q**2))/2

        eq1 =  3 - momsum(integ1,mu,Q)
        eq2 =  -(2*S+1) + momsum(integ2,mu,Q)

        return np.array([eq1,eq2])


    X = np.array([0.1,0.13])
    X = optimize.root(sol,X,method='hybr').x

    ERR = (np.linalg.norm(sol(X))**2)/np.sqrt(2)

    QC = X[0]
    SC = X[1]
    muC = np.sqrt((h*SC)**2+27*(QC**2))/2

    return SC, muC, QC, ERR


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
    
    mu = np.array([mu1, mu2, mu3])
    
    Q = np.array([Q12,Q23,Q31])

    return mu, Q 

def ZMAKE(VA,kappa):
    
    thetaARR = VA[:3]
    chiARR = VA[3:6]
    phiARR = VA[6:9]

    z1 = np.dot(rotSpinor(thetaARR[0],phiARR[0],chiARR[0]),np.array([np.sqrt(kappa),0]))
    z2 = np.dot(rotSpinor(thetaARR[1],phiARR[1],chiARR[1]),np.array([np.sqrt(kappa),0]))
    z3 = np.dot(rotSpinor(thetaARR[2],phiARR[2],chiARR[2]),np.array([np.sqrt(kappa),0]))

    Z = np.array([z1[0],z2[0],z3[0],z1[1].conj(),z2[1].conj(),z3[1].conj()])
    
    return Z

def uniMag(Z):

    z1, z2, z3 = Zravel(Z)
    
    S1 = np.array([np.vdot(z1,np.dot(pauli[0],z1)),np.vdot(z1,np.dot(pauli[1],z1)),np.vdot(z1,np.dot(pauli[2],z1))])/2
    S2 = np.array([np.vdot(z2,np.dot(pauli[0],z2)),np.vdot(z2,np.dot(pauli[1],z2)),np.vdot(z2,np.dot(pauli[2],z2))])/2
    S3 = np.array([np.vdot(z3,np.dot(pauli[0],z3)),np.vdot(z3,np.dot(pauli[1],z3)),np.vdot(z3,np.dot(pauli[2],z3))])/2
    SzTot = np.abs(np.sum(S1+S2+S3))

    return SzTot

def paramMAKEavg(h,S,Z):

    z1, z2, z3 = Zravel(Z)

    Q12 = np.dot(z1,np.dot(leviC,z2))
    Q23 = np.dot(z2,np.dot(leviC,z3))
    Q31 = np.dot(z3,np.dot(leviC,z1))

    mu = np.sqrt((h*S/2)**2+(9/4)*(np.abs(Q12)**2+np.abs(Q23)**2+np.abs(Q31)**2))

    return np.array([mu, mu, mu]), np.array([Q12,Q23,Q31])

def solveMF(mu,Q,Z,h,S,L,mfTol,MaxIter,device,solver):

    NK0 = torch.tensor([0],device=device)

    Y = np.zeros(9)

    ERR = 1e10
    ERR1 = 1e10
    ERR2 = 1e10

    r = 0.3

    for i in range(MaxIter):

        if ERR > mfTol:

            mu = np.abs(mu)
            X = compilevar(mu,Q,Z,0)
            Y[:] = X[:9]

            XGPU = torch.tensor(X,dtype=torch.float64,device=device,requires_grad=True)
            dEig = denQ(XGPU,h,S,L,device,solver)
            D = ZeemanHam(XGPU,NK0,h,S,L,device)[0]

            if device == 'cpu':
                dEig = dEig.numpy()[:9]
                D = D.detach().numpy()
            else:
                dEig = dEig.cpu().numpy()[:9]
                D = D.detach().cpu().numpy()

            kappaTot = (6*S-np.sum(dEig[:3])+0j)

            w, v = np.linalg.eigh(D)
            i0 = np.argmin(w)

            mu = mu - r*w[i0]
            Z = np.sqrt(kappaTot)*v[:,i0]

            dQ = (dEig[3:6] + 1j*dEig[6:9])/2

            z1, z2, z3 = Zravel(Z)

            Q12 = np.dot(z1,np.dot(leviC,z2)) - (2/3)*dQ[0]
            Q23 = np.dot(z2,np.dot(leviC,z3)) - (2/3)*dQ[1]
            Q31 = np.dot(z3,np.dot(leviC,z1)) - (2/3)*dQ[2]

            Q = np.array([Q12,Q23,Q31])

            ERR1 = np.abs(w[i0])
            ERR2 = np.linalg.norm(compilevar(mu,Q,Z,0)[:9]-Y)

            ERR = np.max([ERR1,ERR2])

        else:
            break

    return mu, Q, Z, ERR

def uudLatticeSolve(iS,L):

    kappa = 2*SARR[iS]

    def gamma(k1,k2):

        return (np.cos(k1)+np.cos(k2)+np.cos(k1+k2))

    def Omega(k1,k2,r):

        return np.sqrt(3 + r**2 - gamma(k1,k2))

    def integ1(k1,k2,r):

        return (3+2*gamma(k1,k2))/(6*Omega(k1,k2,r))

    def integ2(k1,k2,r):

        return 2*np.sqrt(4*r**2+18)/Omega(k1,k2,r)

    def rsolve(r):

        X = 0

        for NK in range(L*L):

            i, j = np.unravel_index(NK,(L,L))
            k1 = 2*np.pi*i/L
            k2 = 2*np.pi*j/L

            X += integ2(k1,k2,r)

        Y = 9*kappa -r*np.sqrt(18+4*r**2)*kappa + 2*(r**2)*(-4-5*kappa+X/(L*L))

        return Y

    def QC(r):

        X = 0

        for NK in range(L*L):

            i, j = np.unravel_index(NK,(L,L))
            k1 = 2*np.pi*i/L
            k2 = 2*np.pi*j/L

            X += integ1(k1,k2,r)

        return kappa*(-6*r+3*np.sqrt(18+4*(r**2)))/(16*(r**2)) + X/(L*L)

    def enUUD(Q,mu):

        X = 0

        for NK in range(L*L):

            i, j = np.unravel_index(NK,(L,L))
            k1 = 2*np.pi*i/L
            k2 = 2*np.pi*j/L

            X += (1/3)*np.sqrt(4*mu**2-6*(Q**2)-4*(Q**2)*gamma(k1,k2))

        return Q**2 + mu*(-(2/3)-kappa) + (1/3)*np.sqrt(mu**2-(9/2)*Q**2)*kappa + X/(L*L)

    r = optimize.newton(rsolve,0.01)
    Q = QC(r)
    mu = Q*np.sqrt(r**2+9/2)
    En = enUUD(Q,mu) + (3/4)*(kappa**2)

    return iS, mu, Q, En

def SolveCLQU(iS,ich,ih,S,char,h,mfTol,L,T,MaxIter,SpinSL,SpinCrit,fstring,device,rundata,solver,constraint,uudSOL):
    
    start = time.time()

    muSL, QSL, gapSL, ENSL = SpinSL

    hc = 2*gapSL/S

    heff_list = np.unique([0.1,1,2.5,6,min(h,7),h])

    hnum = len(heff_list)
    hmax = np.amax(heff_list)
    
    if char == 'uud' and uud == "exact":

        iSX, muX, QX, EnX = uudSOL
        B = h*S
        delta = (B/2)-np.sqrt(muX**2-(9/2)*QX**2)
        En = -B*S/3 + EnX
        Q = np.array([0,QX,-QX])
        mu = np.array([muX+delta,muX+delta,muX-delta])
        Z = np.sqrt(2*S)*np.array([1,1,0,0,0,1])
        iy = 0
        X = compilevar(mu,Q,Z,iy)
        ERR = 0

    elif h >= hc and constraint == "averaged":

        hans = 0
        kappa = 2*S
        thetaARR = theta(char,hans)
        chiARR = chi(char,hans)
        phiARR = phi(char,hans)
        VA = np.ravel([thetaARR,chiARR,phiARR])
        Z = ZMAKE(VA,kappa)
        mu, Q = paramMAKEavg(h,S,Z)

        mu, Q, Z, ERR = solveMF(mu,Q,Z,h,S,L,mfTol,MaxIter,device,solver)

        iy = 0
        X = compilevar(mu,Q,Z,iy)
        En = FreeEN(X,h,S,L,T,device,solver)/3

    elif h >= hc:
        
        X_ARR = []
        En_ARR = []
        ERR_ARR = []
        
        for hn in range(hnum):
        
            heff = heff_list[hn]
        
            ### large S ansatz ###
        
            kappa = 2*S
            
            thetaARR = theta(char,heff)
            chiARR = chi(char,heff)
            phiARR = phi(char,heff)
            VA = np.ravel([thetaARR,chiARR,phiARR])
            
            Z = ZMAKE(VA,kappa)

            mu, Q = paramMAKE(heff,S,Z)

            iy = 0 

            Y = compilevar(mu,Q,Z,iy)
            
            ### self-consistent equations ###
            
            Y = optimize.root(stationary,Y,
                              args=(h,S,L,T,device,solver),
                              method='hybr',
                              ).x

            ERRY = np.amax(np.abs(stationary(Y,h,S,L,T,device,solver)))

            if ERRY < mfTol:
            
                X_ARR.append(Y)
                En_ARR.append(FreeEN(Y,h,S,L,T,device,solver)/3)
                ERR_ARR.append(ERRY)

        try:
            ihmin = np.argmin(En_ARR)
            X = X_ARR[ihmin]
            En = En_ARR[ihmin]
            ERR = ERR_ARR[ihmin]

        except:

            kappa = 2*S
            
            thetaARR = theta(char,h)
            chiARR = chi(char,h)
            phiARR = phi(char,h)
            VA = np.ravel([thetaARR,chiARR,phiARR])
            
            Z = ZMAKE(VA,kappa)

            mu, Q = paramMAKE(h,S,Z)

            iy = 0

            X = compilevar(mu,Q,Z,iy)
            
            ### self-consistent equations ###
            
            X = optimize.root(stationary,X,
                              args=(h,S,L,T,device,solver),
                              method='lm',
                              ).x
            
            En = FreeEN(X,h,S,L,T,device,solver)/3
            ERR = np.amax(np.abs(stationary(X,h,S,L,T,device,solver)))

    else:
        
        mu = np.full(3,muSL,dtype=float)
        Q = np.full(3,QSL,dtype=complex)
        Z = np.zeros(6,dtype=complex)
        nc = 0
        iy = 0

        X = compilevar(mu,Q,Z,iy)
        En = ENSL
        ERR = 0

    
    mu, Q, Z, iy = expandvar(X)

    SzTot = uniMag(Z)
    Sz = SzTot/3
    nc = (np.linalg.norm(Z)**2)/3

    end = time.time()
    print("type","h","S","Sz/S","nc","En","ERR","Time")
    print(char,h,S,Sz/S,nc,En,ERR,end-start)
    sys.stdout.flush()

    if rundata == "on":
    
        with open(fstring,"a") as file:
            file.write(f"{iS} {ich} {ih} {Sz} {En} {nc} {ERR} {mu[0]} {mu[1]} {mu[2]} {Q[0]} {Q[1]} {Q[2]} {Z[0]} {Z[1]} {Z[2]} {Z[3]} {Z[4]} {Z[5]}\n")

    else:

        pass

    return iS, ich, ih, Sz, En, mu, Q, Z, nc, ERR

def theta(string,h):

    if string == 'umbrella':

        if h <= 9:
        
            ARR = np.array([np.arccos(h/9),np.arccos(h/9),np.arccos(h/9)])

        else:
        
            ARR = np.array([0,0,0])
 
    
    elif string == 'yv':

        if h <= 3:

            ARR = np.array([np.arccos((3+h)/6),-np.arccos((3+h)/6),np.pi])
    
        elif h > 3 and h <= 9:

            ARR = np.array([-np.arccos((-27+h**2)/(6*h)),np.arccos((27+h**2)/(12*h)),np.arccos((27+h**2)/(12*h))])
 
        else:
        
            ARR = np.array([0,0,0])
   
    elif string == 'uud':

        ARR = np.array([0,0,np.pi])
    
    return ARR

def chi(string,h):

    if string == 'umbrella':
        
        ARR = np.array([0,2*np.pi/3,4*np.pi/3])
    
    elif string == 'yv':
 
        ARR = np.array([0,0,0])
     
    elif string == 'uud':
 
        ARR = np.array([0,0,0])
    
    return ARR

def phi(string,h):

    if string == 'umbrella':
        
        ARR = np.array([0,0,0])
    
    elif string == 'yv':
        
        ARR = np.array([0,0,0])
     
    elif string == 'uud':
 
        ARR = np.array([0,0,0])
 
    return ARR

## External Parameters

pauli = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
leviC = np.array([[0,1],[-1,0]])
dt = 10**(-4)
gTol = 10**(-16)
mfTol = 10**(-6)
L = 36
T = 1/(L*L) # finite temperature
MaxIter = 1000
SlqMxRatio = 0.5

rundata = "off"  # run data file switch

solver = "smit" # Bogoliubov solver "wessel-milat"/"colpa"/"smit"

constraint = "onsite" # n = 2S on each sublattice
#constraint = "averaged" # n_1+n_2+n_3 = 6S over three sublattices

uud = "exact" # uudLatticeSolve
#uud = "variational" # solver

SARR = np.array([0.05,0.06,0.07,0.09,0.10,0.12,0.14,0.16,0.18,0.20,0.22,0.25,0.3,0.4,0.5,1,2,4])
#SARR = np.array([0.5])

CONFIG = np.array(['yv','uud','umbrella'])
#CONFIG = np.array(['yv','uud'])

hARR = np.linspace(0,3,endpoint=False,num=16)
hARR= np.append(hARR,np.linspace(3,5,endpoint=False,num=16))
hARR= np.append(hARR,np.linspace(5,6,endpoint=False,num=16))
hARR= np.append(hARR,np.linspace(6,7,endpoint=False,num=16))
hARR= np.append(hARR,np.linspace(7,8,endpoint=False,num=16))
hARR= np.append(hARR,np.linspace(8,9,num=16))
hARR= np.append(hARR,np.linspace(9,12,num=16))
#hARR = np.array([0.5,4.3,6.0])

muARR = np.zeros((SARR.size,CONFIG.size,hARR.size,3),dtype=float) 
QARR = np.zeros((SARR.size,CONFIG.size,hARR.size,3),dtype=complex) 
ZARR = np.zeros((SARR.size,CONFIG.size,hARR.size,6),dtype=complex) 
freeARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float) 
SzARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float) 
ncARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float) 
errARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float) 


if __name__=='__main__':
    
    
    if mode == "cluster":

        with MPIPoolExecutor() as executor:
            SpinSL = np.array(list(executor.starmap(SolveMFSL,[(S,) for S in SARR])))
 
        with MPIPoolExecutor() as executor:
            SpinCrit = np.array(list(executor.starmap(critS,[(h,L,) for h in hARR])))

        with MPIPoolExecutor() as executor:
            uudSOL = list(executor.starmap(uudLatticeSolve,[(iS,L,) for iS in range(SARR.size)]))

   
    elif mode == "cpu" or mode == "gpu":

        with mp.Pool() as pool:
            SpinSL = np.array(pool.starmap(SolveMFSL,[(S,) for S in SARR]))
    
        with mp.Pool() as pool:
            SpinCrit = np.array(pool.starmap(critS,[(h,L,) for h in hARR]))

        uudSOL = [uudLatticeSolve(iS,L,) for iS in range(SARR.size)]


    # Data file
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
                        T = T,
                        SARR = SARR,
                        hARR = hARR,
                        CONFIG = CONFIG,
                        SpinSL = SpinSL,
                        SpinCrit = SpinCrit,
                        mfTol = mfTol,
                        MaxIter = MaxIter
                        )

    else:

        pass

    if mode == "gpu":
        nGPU = torch.cuda.device_count()
        devices = itertools.cycle(list([f'cuda:{id}' for id in range(nGPU)]))
    elif mode == "cluster" or mode == "cpu":
        devices = itertools.cycle(list(['cpu']))

    ex_arr = []

    for iS in range(SARR.size):
    
        S = SARR[iS]
    
        for ich in range(CONFIG.size):

            char = CONFIG[ich]
        
            for ih in range(hARR.size):
    
                h = hARR[ih]

                ex_arr.append((iS,ich,ih,S,char,h,mfTol,L,T,
                    MaxIter,SpinSL[iS],SpinCrit[ih],fstring+".dat",next(devices),rundata,solver,constraint,uudSOL[iS]))

    if mode == "cluster":
  
        with MPIPoolExecutor() as executor:
            res = list(executor.starmap(SolveCLQU,ex_arr))
     
    elif mode == "cpu" or mode == "gpu":
    
        with mp.Pool() as pool:
            res = pool.starmap(SolveCLQU,ex_arr,chunksize=1)
        #res = [SolveCLQU(*exr) for exr in ex_arr]
        
    for ires in range(len(res)):
        
        iS, ich, ih, Sz, En, mu, Q, Z, nc, ERR = res[ires]
            
        muARR[iS,ich,ih] = mu
        QARR[iS,ich,ih] = Q
        freeARR[iS,ich,ih] = En
        ZARR[iS,ich,ih] = Z
        SzARR[iS,ich,ih] = Sz
        ncARR[iS,ich,ih] = nc
        errARR[iS,ich,ih] = ERR
        
    TSTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H-%M-%S')
    np.savez_compressed('L_'+str(L)+'_TS_'+TSTAMP+'_'+'var1ch_data.npz',
                        SARR=SARR,
                        CONFIG=CONFIG,
                        hARR=hARR,
                        SpinSL=SpinSL,
                        SpinCrit=SpinCrit,
                        muARR=muARR,
                        QARR=QARR,
                        ZARR=ZARR,
                        freeARR=freeARR,
                        SzARR=SzARR,
                        ncARR=ncARR,
                        errARR=errARR,
                        L=L,
                        T=T,
                        mfTol=mfTol,
                        MaxIter=MaxIter,
                        )
