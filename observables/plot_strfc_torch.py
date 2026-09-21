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

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.figure as figure
import matplotlib.ticker as mtick
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from matplotlib.ticker import PercentFormatter
from matplotlib.ticker import FormatStrFormatter

params = np.load('sb_sfcdata.npz')

SARR = params['SARR']
CONFIG = params['CONFIG']
hARR=params['hARR']
sfcARR=params['sfcARR']
sfcARR3=params['sfcARR3']
FREQ=params['FREQ']
T=params['T']
ETA=params['ETA']
L=params['L']
LT=params['LT']

indx = sfcARR.shape


FMAX = FREQ.max()

methods = ['none', 'nearest', 'bilinear', 'hanning', 'hamming', 'hermite', 'kaiser']

if __name__=='__main__':

    #for meth in methods:
    for meth in ['hanning']:

        for iS in range(SARR.size):
        #for iS in indxSARR:

            kappa = 2*SARR[iS]
            
            for ich in range(CONFIG.size):
            
                for ih in range(hARR.size):
                    
                    VFX = np.real(1/(1-np.exp(-(FREQ+1j*ETA)/T)))*np.imag(sfcARR[iS,ich,ih])/(np.pi)
                    VFX3 = np.real(1/(1-np.exp(-(FREQ+1j*ETA)/T)))*np.imag(sfcARR3[iS,ich,ih])/(np.pi)
                    #VFX = 2*np.imag(sfcARR[iS,ich,ih])
                    #VFX3 = 2*np.imag(sfcARR3[iS,ich,ih])

                   
                    ## plot routines

                    # palette
                    plt.style.use('tableau-colorblind10')


                    ## mathtext style
                    plt.rcParams['mathtext.fontset']='cm'
                    plt.rc('font',family='serif')

                    # marker strings

                    markers = itertools.cycle(['o','s','^','D','X','p','H','P',
                        "*","|","1"])

                    #markers = ['o','s','^','D','X','p','H','P',
                    #    "*","|","1","+"]

                    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
    
                    afig, ax = plt.subplots()

                    QC = int(LT//2+LT//6)

                    #QC += 2
                    
                    ax.plot(FREQ,VFX[QC,:],label=r'$S^{(0)}$')
                    ax.plot(FREQ,VFX3[QC,:],ls='--',label=r'$S^{(1)}$')
                    ax.plot(FREQ,VFX[QC,:]+VFX3[QC,:],label=r'$S^{(0)}+S^{(1)}/N$')
                    #ax.set_yscale('log')
                    ax.set_xlabel(r'$\omega/J$',fontsize=22)
                    ax.set_ylabel(r'$S(\omega,\bf \it k_c)$',fontsize=22)
                    #ax.set_xlim(left=0)
                    #ax.set_ylim(bottom=0)
                    legend = ax.legend(
                            loc='best',
                            #loc='lower right',
                            #bbox_to_anchor=(1.45,0.85),
                            #bbox_transform=afig.transFigure,
                            #fontsize=22,
                            fontsize=12,
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
                    ax.tick_params(which='major',width=2,
                                   labelsize=22,direction='in',
                                   length=10,
                                   #pad=15,
                                   bottom=True,top=False,
                                   left=True,right=False)


                    afig.savefig("figs/STRFC_KC-"+CONFIG[ich]+"-kappa-"+str("%.2f" % kappa)+"-h-"+str("%.2f" % hARR[ih])+".pdf",
                                 bbox_inches='tight',transparent=True,
                                 )

                    SFX = (VFX+VFX3).T

                    SFX = ndimage.uniform_filter1d(SFX,size=40,axis=0)
                    #SFX = ndimage.uniform_filter1d(SFX,size=4,axis=1)
                    SFX = np.clip(SFX,1e-4,10)
                    SFX = SFX/np.amax(SFX)
                    vmax= 1
                    vmin = vmax/1e3
                    SFX[SFX < vmin ] = vmin
                    #SFX[SFX > vmax] = vmax
                    #SFX[SFX < 0 ] = 1e-20

                    cfig, cx = plt.subplots()

                    cax = cx.imshow(SFX,
                                    #interpolation='none',
                                    interpolation=meth,
                                    extent=(0,1/2+1/6+1/3,0,FMAX),
                                    cmap=cm.plasma,
                                    #cmap=cm.cividis,
                                    #vmin=vmin,
                                    #vmax=vmax,
                                    #norm=LogNorm(vmin=vmin,vmax=vmax),
                                    origin='lower',
                                    aspect=1/FMAX,
                                    )

                    #cbar = cfig.colorbar(cax,shrink=0.5,format='%.0e')
                    cbar = cfig.colorbar(cax,shrink=0.5)
                    cbar.ax.tick_params(labelsize=22)
                    cx.set_xticks([0,1/2,1/2+1/6,1/2+1/6+1/3])
                    cx.set_yticks([0,FMAX/2,FMAX])
                    cx.set_xticklabels([r'$\Gamma$',r'M',r'K',r'$\Gamma$'],fontsize=22)
                    #cx.set_yticklabels(['wrong','1','2'])
                    #cx.set_title(r'$S(\omega,\vec q)$',fontsize=20)
                    cx.tick_params(labelsize=22)
                    cx.set_xlabel(r'$\bf\it k$',fontsize=22)
                    cx.set_ylabel(r'$\omega/J$',fontsize=22)
                    cfig.savefig("figs/"+meth+"_STRFC_kappa_"+str("%f" % kappa)+"_"+CONFIG[ich]+"_h_"+str("%f" % hARR[ih])+".pdf",
                                 bbox_inches='tight',
                                 transparent=True,
                                 )
                    plt.close()

