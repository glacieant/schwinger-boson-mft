#!/usr/bin/env python
import os
import multiprocessing as mp
import itertools
import time

# MKL threads 
os.putenv("MKL_DYNAMIC","FALSE")

path = 'figs'
if not os.path.exists(path):
    os.makedirs(path)

import numpy as np
from numpy import linalg

from scipy.linalg import eigvalsh
from scipy.linalg import eigvals
from scipy import optimize
from scipy.linalg import null_space
from scipy import integrate
from scipy.linalg import cholesky
from scipy.linalg import ldl
from scipy import ndimage

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.figure as figure
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FormatStrFormatter
from matplotlib import colors
import matplotlib.ticker as tick
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colormaps
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes
import matplotlib.colorbar as colorbar

## External Parameters

pauli = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])
leviC = np.array([[0,1],[-1,0]])
dt = 10**(-3)
gTol = 10**(-16)
mfTol = 10**(-6)
L = 48
MaxIter = 1000
SlqMxRatio = 0.5

params = np.load('var1ch_data.npz')

SARR = params['SARR']
SpinSL = params['SpinSL']
CONFIG = params['CONFIG']
hARR=params['hARR']
QARR=params['QARR']
muARR=params['muARR']
ZARR=params['ZARR']
freeARR=params['freeARR']
SzARR=params['SzARR']
try:
    ncARR=params['ncARR']
except:
    ncARR=params['mARR']
errARR=params['errARR']
try:
    L = params['L']
except:
    pass
try:
    mfTol = params['mfTol']
except:
    pass

bads = np.nonzero(errARR > 2*mfTol)

freeARR[bads] = np.nan
ncARR[bads] = np.nan
SzARR[bads] = np.nan

LABEL = [0]*len(CONFIG)

for ich in range(CONFIG.size):

    if CONFIG[ich] == 'yv':
        LABEL[ich] = 'coplanar'
    elif CONFIG[ich] == 'umbrella':
        LABEL[ich] = 'noncoplanar'
    elif CONFIG[ich] == 'uud':
        LABEL[ich] = 'UUD'

## plot routines

# palette
plt.style.use('tableau-colorblind10')


## mathtext style
plt.rcParams['mathtext.fontset']='cm'
plt.rc('font',family='serif')

def compilevar(mu,Q,Z):

    X = np.hstack((mu,Q.real,Q.imag,Z.real,Z.imag))

    return X

def expandvar(X):
    
    mu = X[:3]
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
    
    Sigma = np.diag([1,1,1,-1,-1,-1])
    D = ZeemanHam(X,h,S,NK,L)
    dim = D.shape[0]//2
    
    # Smit et al. Bogoliubov solver (https://doi.org/10.1103/PhysRevB.101.054424)
    
    l, v = np.linalg.eig(np.matmul(Sigma,D))
    ls = np.argsort(-np.real(l))
    l = l[ls]
    v = v[:,ls]
    ls2 = np.argsort(np.real(l)[dim:])
    l = np.hstack((l[:dim],l[dim:][ls2]))
    v = np.hstack((v[:,:dim],v[:,dim:][:,ls2]))
    vnorm = np.diag(np.matmul(np.transpose(np.conj(v)),np.matmul(Sigma,v)))
    T = v/np.sqrt(np.abs(vnorm))

    TD = np.transpose(np.conj(T))
    ENX = np.real(np.diag(np.matmul(np.matmul(TD,D),T)))
   
    return ENX
 
def critS(h):

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

        L = 72

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

    Qc = X[0]
    Sc = X[1]
    muc = np.sqrt((h*Sc)**2+27*(Qc**2))/2

    return [Sc,muc,Qc,ERR]


if __name__=='__main__':
    
    
    #with MPIPoolExecutor() as executor:
    #    ScARR = np.array(list(executor.starmap(critS,[(h,) for h in hARR])))
    with mp.Pool() as pool:
        ScARR = np.array(pool.starmap(critS,[(h,) for h in hARR]))
    
    gapDat = SpinSL[:,2]

    hAXIS = [0,2,4,6,8,9]

    ## plot routines

    # palette
    plt.style.use('tableau-colorblind10')

    # marker strings

    markers = itertools.cycle(['o','^','s','D','X','p','H','P',
                               "*","|","1"])

    #markers = ['o','s','^','D','X','p','H','P',
    #    "*","|","1","+"]


    ## mathtext style
    plt.rcParams['mathtext.fontset']='cm'
    plt.rc('font',family='serif')
        
    frame = [0.2,0.2,0.8,0.8]
    w,h = figure.figaspect(1.0)

    afig = plt.figure(figsize=(w,h))
    ax = afig.add_axes(frame)


    for ich in range(CONFIG.size):
        
        char = CONFIG[ich]

        if char != "uud":

            satH = np.zeros(SARR.size,dtype=float)

            for iS in range(SARR.size):

                try:
                    ihc = np.amin(np.argwhere(SzARR[iS,ich,:]/SARR[iS]>1-mfTol))
                    satH[iS] = hARR[ihc]
                except:
                    satH[iS] = np.nan

            
            line = ax.plot(1/(2*SARR),
                           satH,
                           ms=10,
                           marker='o',
                           label=LABEL[ich],
                           lw=4,
                           linestyle='--',
                           zorder=2,
                           )
            
            ax.tick_params(which='major',width=2,
                           labelsize=22,direction='in',
                           length=10,
                           #pad=15,
                           bottom=True,top=False,
                           left=True,right=False)
 
    legend = ax.legend(
            loc='best',
            #loc='upper left',
            #bbox_to_anchor=(1.45,0.85),
            #bbox_transform=afig.transFigure,
            fontsize=22,
            #markerscale=0.75,
            facecolor='w',
            edgecolor='k',
            framealpha=1,
            ncol=1,
            borderpad=0.5,
            handletextpad=0.5,
            handlelength=0.4,
            #labelspacing=0.4,
            #frameon=False
            )

               
    ax.set_title(r'Saturation Field',fontsize=22)
    ax.set_xlabel(r'$1/\kappa$',fontsize=22)
    ax.set_ylabel(r'$h_{\rm sat}/J$',fontsize=22)
    ax.set_xlim(left=0,right=10)
    ax.set_ylim(bottom=0)
    
    afig.savefig("figs/saturation_field.pdf",
                 bbox_inches='tight',transparent=True,
                 )

    plt.close('all')
    
    frame = [0.2,0.2,0.8,0.8]
    w,h = figure.figaspect(1.0)
    
    afig = plt.figure(figsize=(w,h))
    ax = afig.add_axes(frame)
    
    line = ax.plot(1/(2*SARR),
                   gapDat,
                   ms=10,
                   marker='o',
                   lw=4,
                   linestyle='--',
                   zorder=2,
                   )
    
    ax.tick_params(which='major',width=2,
                   labelsize=22,direction='in',
                   length=10,
                   #pad=15,
                   bottom=True,top=False,
                   left=True,right=False)
        
    #ax.set_title(r'Spinon Gap',fontsize=22)
    ax.set_xlabel(r'$1/\kappa$',fontsize=22)
    ax.set_ylabel(r'$\Delta/J$',fontsize=22)
    ax.set_xlim(left=0,right=10)
    ax.set_ylim(bottom=0)
    
    afig.savefig("figs/spinon_gap.pdf",
                 bbox_inches='tight',transparent=True,
                 )
    
    plt.close('all')
    
    
    for iS in range(SARR.size):

        kappa = 2*SARR[iS]

        # marker strings

        markers = itertools.cycle(['o','s','s','D','X','p','H','P',
                                   "*","|","1"])

        markers2 = ['o','s','s','D','X','p','H','P',
            "*","|","1","+"]


        frame = [0.2,0.2,0.8,0.8]
        w,h = figure.figaspect(1.0)
        afig = plt.figure(figsize=(w,h))
        ax = afig.add_axes(frame)
        bfig = plt.figure(figsize=(w,h))
        bx = bfig.add_axes(frame)
        cfig = plt.figure(figsize=(w,h))
        cx = cfig.add_axes(frame)
        dfig = plt.figure(figsize=(w,h))
        dx = dfig.add_axes(frame)
        
        mfig = plt.figure(figsize=(w,h))
        mx = mfig.add_axes(frame)

        YARR = ((-hARR)*SARR[iS]/2)+gapDat[iS]
    
        line = mx.plot(hARR,
                       YARR*np.heaviside(YARR,0),
                       ms=10,
                       marker=next(markers),
                       lw=4,
                       linestyle='--',
                       zorder=2,
                       )


        mx.tick_params(which='major',width=2,
                   labelsize=22,direction='in',
                   length=10,
                   #pad=15,
                   bottom=True,top=False,
                   left=True,right=False)
        
        mx.set_title(r'$\kappa = $ '+str("%.2f" % kappa),fontsize=22)
        mx.set_xlabel(r'$h/J$',fontsize=22)
        mx.set_ylabel(r'$\Delta/J$',fontsize=22)
        mx.set_xticks(hAXIS)
        mx.set_xlim(left=0)
        mx.set_ylim(bottom=0)

        mfig.savefig("figs/spinon_gap_kappa-"+str("%.2f" % kappa)+".pdf",
                 bbox_inches='tight',transparent=True,
                 )
            
        hc = 2*gapDat[iS]/SARR[iS]
        hARG0 = np.argwhere(hARR<hc)
       
        if hARG0.size > 0:

            En = freeARR[iS,0]
            Sz = SzARR[iS,0]/SARR[iS]
            nc = ncARR[iS,0]
            
            line = ax.plot(hARR[hARG0],
                           En[hARG0],
                           #ms=10,
                           #marker=next(markers),
                           #marker=markers2[4],
                           #fillstyle='none',
                           label=r'$\mathbb{Z}_2$ SL',
                           lw=4,
                           linestyle=':',
                           #zorder=2-ich,
                           alpha=1
                           )

            
            line = bx.plot(hARR[hARG0],
                           Sz[hARG0],
                           #ms=10,
                           #marker=next(markers),
                           #marker=markers2[4],
                           #fillstyle='none',
                           label=r'$\mathbb{Z}_2$ SL',
                           lw=4,
                           linestyle=':',
                           #zorder=2-ich,
                           alpha=1
                           )

            line = cx.plot(hARR[hARG0],
                           nc[hARG0],
                           #ms=10,
                           #marker=next(markers),
                           #marker=markers2[4],
                           #fillstyle='none',
                           label=r'$\mathbb{Z}_2$ SL',
                           lw=4,
                           linestyle=':',
                           #zorder=2-ich,
                           alpha=1
                           )



        for ich in range(CONFIG.size):
            
            for ih in range(hARR.size):
                
                S = SARR[iS]
                h = hARR[ih]
                X = compilevar(muARR[iS,ich,ih],QARR[iS,ich,ih],ZARR[iS,ich,ih])
                LK = 48
                MOM = np.arange(-LK,LK,1)*(2*np.pi/LK)
                SpecUp = np.zeros((2*LK,3),dtype=float)
                SpecDown = np.zeros((2*LK,3),dtype=float)

                for NK in range(2*LK):
                    NKP = (NK - LK)

                    band = eig(X,h,S,NKP,2*LK)

                    SpecUp[NK] =  band[:3]
                    SpecDown[NK] =  band[3:]

                c1, c2 = figure.figaspect(1.0)
                ffig = plt.figure(figsize=(c1,c2))
                fx = ffig.add_axes(frame)

                # marker
                line = fx.plot(MOM,
                               SpecUp[:,0],
                               color='tab:blue',
                               #ms=10,
                               #marker='x',
                               #marker=next(markers),
                               label=r'$\uparrow$',
                               lw=4,
                               linestyle='dotted',
                               zorder=2,
                               )
                line = fx.plot(MOM,
                               SpecUp[:,1:],
                               color='tab:blue',
                               #ms=10,
                               #marker='x',
                               #marker=next(markers),
                               lw=4,
                               linestyle='dotted',
                               zorder=2,
                               )
                
                # marker
                line = fx.plot(MOM,
                               SpecDown[:,0],
                               color='tab:orange',
                               #ms=10,
                               #marker='o',
                               #marker=next(markers),
                               label=r'$\downarrow$',
                               lw=4,
                               linestyle='dashed',
                               zorder=1,
                               )
                
                line = fx.plot(MOM,
                               SpecDown[:,1:],
                               color='tab:orange',
                               #ms=10,
                               #marker='o',
                               #marker=next(markers),
                               lw=4,
                               linestyle='dashed',
                               zorder=1,
                               )

                kappa = 2*S
                invkappa = 1/kappa

                #fx.set_title('SB Spectrum for '+CONFIG[ich]+r': $h$ = '+str("%.2f" % hARR[ih])+r', $1/\kappa$  = '+str("%.2f" % invkappa),fontsize=22)
                #fx.set_xlabel(r'$k_1$',fontsize=22)
                fx.set_ylabel(r'$\omega_{n\pm}(\bf\it k )$',fontsize=22)
                fx.set_ylim(bottom=0)
                fx.set_xlim(left=-4*np.pi/3,right=4*np.pi/3)
                fx.set_xticks([-4*np.pi/3,0,4*np.pi/3])
                fx.set_xticklabels([r'$K$',r'$\Gamma$',r'$K$'])
                
                fx.tick_params(which='major',width=2,
                               labelsize=22,direction='in',
                               length=10,
                               #pad=15,
                               bottom=True,top=False,
                               left=True,right=False)
                legend = fx.legend(
                        loc='best',
                        #loc='upper left',
                        #bbox_to_anchor=(1.45,0.85),
                        #bbox_transform=afig.transFigure,
                        fontsize=22,
                        #markerscale=0.75,
                        facecolor='w',
                        edgecolor='k',
                        framealpha=1,
                        ncol=1,
                        borderpad=0.5,
                        #handletextpad=0.5,
                        handlelength=1,
                        #labelspacing=0.4,
                        #frameon=False
                        )



                results_dir = "figs/spec/"+CONFIG[ich]+"_kappa_"+str("%.2f" % kappa)
                if not os.path.isdir(results_dir):
                    os.makedirs(results_dir)
                ffig.savefig(results_dir+"/spectrum_kappa_"+str("%.2f" % kappa)+"_h_"+str("%.2f" % hARR[ih])+".pdf",
                             bbox_inches='tight',transparent=True,)
                plt.close()
                


            lstyle = ['solid','solid','dashed']
            alpha = [1,0.8,0.8]

            #if CONFIG[ich] == "uud":
                
            hc = 2*gapDat[iS]/SARR[iS]
            hARG2 = np.argwhere(hARR>=hc)

            if hARG2.size > 0:

                try:
                    hARG = np.concatenate(([hARG0[-1]],hARG2))
                except:
                    hARG = hARG2

                En = freeARR[iS,ich]
                Sz = SzARR[iS,ich]/SARR[iS]
                nc = ncARR[iS,ich]
                err = errARR[iS,ich]
                
                line = ax.plot(hARR[hARG][:-1],
                               En[hARG][:-1],
                               #ms=10,
                               #marker=next(markers),
                               #marker=markers2[ich],
                               #fillstyle='none',
                               label=LABEL[ich],
                               lw=4,
                               linestyle=lstyle[ich],
                               zorder=ich,
                               #alpha=alpha[ich]
                               )

                line = bx.plot(hARR[hARG][:-1],
                               Sz[hARG][:-1],
                               #ms=10,
                               #marker=next(markers),
                               #marker=markers2[ich],
                               #fillstyle='none',
                               label=LABEL[ich],
                               lw=4,
                               linestyle=lstyle[ich],
                               zorder=ich,
                               #alpha=alpha[ich]
                               )

                line = cx.plot(hARR[hARG][:-1],
                               nc[hARG][:-1],
                               #ms=10,
                               #marker=next(markers),
                               #marker=markers2[ich],
                               #fillstyle='none',
                               label=LABEL[ich],
                               lw=4,
                               linestyle=lstyle[ich],
                               zorder=ich,
                               #alpha=alpha[ich]
                               )
                
                line = dx.plot(hARR[hARG],
                               err[hARG],
                               ms=10,
                               #marker=next(markers),
                               marker=markers2[ich],
                               fillstyle='none',
                               label=LABEL[ich],
                               lw=4,
                               linestyle=lstyle[ich],
                               zorder=ich,
                               #alpha=0.5
                               )
 
        ax.tick_params(which='major',width=2,
                       labelsize=22,direction='in',
                       length=10,
                       #pad=15,
                       bottom=True,top=False,
                       left=True,right=False)

        legend = ax.legend(
                loc='best',
                #loc='upper left',
                #bbox_to_anchor=(1.45,0.85),
                #bbox_transform=afig.transFigure,
                fontsize=22,
                #markerscale=0.75,
                facecolor='w',
                edgecolor='k',
                framealpha=1,
                ncol=1,
                borderpad=0.5,
                handletextpad=0.5,
                handlelength=0.4,
                #labelspacing=0.4,
                #frameon=False
                )


        ax.set_xlabel(r'$h/J$',fontsize=22)
        ax.set_ylabel(r'$\varepsilon_0/N$',fontsize=22)
        #ax.set_xticks(hAXIS)
        ax.set_xlim(left=0)
        


        afig.savefig("figs/FreeEn-kappa-"+str("%.2f" % kappa)+".pdf",
                     bbox_inches='tight',transparent=True,
                     )


        bx.tick_params(which='major',width=2,
                       labelsize=22,direction='in',
                       length=10,
                       #pad=15,
                       bottom=True,top=False,
                       left=True,right=False)

        legend = bx.legend(
                loc='best',
                #loc='upper left',
                #bbox_to_anchor=(1.45,0.85),
                #bbox_transform=afig.transFigure,
                fontsize=22,
                #markerscale=0.75,
                facecolor='w',
                edgecolor='k',
                framealpha=1,
                ncol=1,
                borderpad=0.5,
                handletextpad=0.5,
                handlelength=0.4,
                #labelspacing=0.4,
                #frameon=False
                )

        bx.set_xlabel(r'$h/J$',fontsize=22)
        bx.set_ylabel(r'$m/S$',fontsize=22)
        #bx.set_xticks(hAXIS)
        bx.set_xlim(left=0)
        bx.set_ylim(bottom=0)

        bfig.savefig("figs/magnetization-kappa-"+str("%.2f" % kappa)+".pdf",
                     bbox_inches='tight',transparent=True,
                     )

        cx.tick_params(which='major',width=2,
                       labelsize=22,direction='in',
                       length=10,
                       #pad=15,
                       bottom=True,top=False,
                       left=True,right=False)

        legend = cx.legend(
                loc='best',
                #loc='upper left',
                #bbox_to_anchor=(1.45,0.85),
                #bbox_transform=afig.transFigure,
                fontsize=22,
                #markerscale=0.75,
                facecolor='w',
                edgecolor='k',
                framealpha=1,
                ncol=1,
                borderpad=0.5,
                handletextpad=0.5,
                handlelength=0.4,
                #labelspacing=0.4,
                #frameon=False
                )

        cx.set_xlabel(r'$h/J$',fontsize=22)
        cx.set_ylabel(r'$n_c$',fontsize=22)
        cx.set_xlim(left=0)
        #cx.set_xticks(hAXIS)
        cx.set_ylim(bottom=0)

        cfig.savefig("figs/nc-kappa-"+str("%.2f" % kappa)+".pdf",
                     bbox_inches='tight',transparent=True,
                     )

        dx.tick_params(which='major',width=2,
                       labelsize=22,direction='in',
                       length=10,
                       #pad=15,
                       bottom=True,top=False,
                       left=True,right=False)

        legend = dx.legend(
                loc='best',
                #loc='upper left',
                #bbox_to_anchor=(1.45,0.85),
                #bbox_transform=afig.transFigure,
                fontsize=22,
                #markerscale=0.75,
                facecolor='w',
                edgecolor='k',
                framealpha=1,
                ncol=1,
                borderpad=0.5,
                handletextpad=0.5,
                handlelength=0.4,
                #labelspacing=0.4,
                #frameon=False
                )

        dx.set_xlabel(r'$h/J$',fontsize=22)
        dx.set_ylabel(r'$\sigma/\sqrt{\rm mf_{DIM}}$',fontsize=22)
        dx.set_yscale('log')

        dfig.savefig("figs/sol_err-kappa-"+str("%.2f" % kappa)+".pdf",
                     bbox_inches='tight',transparent=True,
                     )

        plt.close('all')
