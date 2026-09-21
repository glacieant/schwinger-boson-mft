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

params = np.load('largeS_data.npz',allow_pickle=True)

SARR = params['SARR']
CONFIG = params['CONFIG']
hARR=params['hARR']
QARR=params['QARR']
muARR=params['muARR']
ZARR=params['ZARR']
freeARR=params['freeARR']
SzARR=params['SzARR']
mARR=params['mARR']

## plot routines

# palette
plt.style.use('tableau-colorblind10')


## mathtext style
plt.rcParams['mathtext.fontset']='cm'
plt.rc('font',family='serif')

def modCONFIG(char):

    if char == 'yv':

        return "Coplanar"

    elif char == "umbrella":

        return "Noncoplanar"

    elif char == "uud":

        return "UUD"
   
if __name__=='__main__':
    

    for iS in range(SARR.size):
        
        ## plot routines

        # palette
        plt.style.use('tableau-colorblind10')

        # marker strings

        markers = itertools.cycle(['o','s','^','D','X','p','H','P',
                                   "*","|","1"])

        #markers = ['o','s','^','D','X','p','H','P',
        #    "*","|","1","+"]


        ## mathtext style
        plt.rcParams['mathtext.fontset']='cm'
        plt.rc('font',family='serif')

        kappa = 2*SARR[iS]
        
        frame = [0.2,0.2,0.8,0.8]
        w,h = figure.figaspect(1.0)
        bfig = plt.figure(figsize=(w,h))
        bx = bfig.add_axes(frame)

        afig = plt.figure(figsize=(w,h))
        ax = afig.add_axes(frame)
        cfig = plt.figure(figsize=(w,h))
        cx = cfig.add_axes(frame)

        SzX = []
        
        for ih in range(hARR.size):

            ichmin = np.argmin(freeARR[iS,:,ih])
            SzX.append((SzARR[iS,ichmin,ih]/SARR[iS]))
        
        line = bx.plot(hARR,
                       SzX,
                       ms=4,
                        marker=next(markers),
                       label=r'$m^z = -(1/\mathcal{N})\partial E_{\rm min}/\partial B^z$',
                       lw=4,
                       linestyle=':',
                       zorder=1,
                       )
        
       
        for ich in range(CONFIG.size):

            EX = freeARR[iS,ich,:]
            MX = mARR[iS,ich,:]/SARR[iS]

            line = ax.plot(hARR,
                           EX,
                           ms=4,
                           #mfc=None, 
                           marker=next(markers),
                           label=modCONFIG(CONFIG[ich]),
                           #lw=4,
                           linestyle='None',
                           zorder=1,
                           )

            line = cx.plot(hARR,
                           MX,
                           ms=4,
                            #mfc=None, 
                            marker=next(markers),
                           label=modCONFIG(CONFIG[ich]),
                           #lw=4,
                           linestyle='None',
                           zorder=1,
                           )

        ax.tick_params(which='major',width=2,
                       labelsize=18,direction='in',
                       length=10,
                       #pad=15,
                       bottom=True,top=False,
                       left=True,right=False)

        legend = ax.legend(
                loc='best',
                #loc='upper left',
                #bbox_to_anchor=(1.45,0.85),
                #bbox_transform=afig.transFigure,
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


        #ax.set_title(r'Free energy, $\kappa = $'+str("%.2f" % kappa),fontsize=20)
        #ax.set_title(r'$\kappa = $ '+str("%d" % kappa),fontsize=20)
        ax.set_xlabel(r'$h/J$',fontsize=20)
        ax.set_ylabel(r'$E_{\rm CND}/JN\mathcal{N}$',fontsize=20)
        #ax.set_ylim(top=-.1,bottom=-.3)
        #ax.set_xlim(left=0)

        afig.savefig("figs/largeS-FreeEn-kappa-"+str("%.2f" % kappa)+".pdf",
                     bbox_inches='tight'
                     )
        
        bx.tick_params(which='major',width=2,
                       labelsize=18,direction='in',
                       length=10,
                       #pad=15,
                       bottom=True,top=False,
                       left=True,right=False)
        
        legend = bx.legend(
                loc='best',
                #loc='upper left',
                #bbox_to_anchor=(1.45,0.85),
                #bbox_transform=afig.transFigure,
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

        #bx.set_title(r'$\kappa = $ '+str("%d" % kappa),fontsize=20)
        #bx.set_title(r'Magnetization, $\kappa = $'+str("%.2f" % kappa),fontsize=20)
        bx.set_xlabel(r'$h/J$',fontsize=20)
        bx.set_ylabel(r'$m^z/m^z_{\rm sat.}$',fontsize=20)
        bx.set_xlim(left=0)
        bx.set_ylim(bottom=0,top=1)

        bfig.savefig("figs/largeS-magnetization-kappa-"+str("%.2f" % kappa)+".pdf",
                     bbox_inches='tight'
                     )

        cx.tick_params(which='major',width=2,
                       labelsize=18,direction='in',
                       length=10,
                       #pad=15,
                       bottom=True,top=False,
                       left=True,right=False)
        
        legend = cx.legend(
                loc='best',
                #loc='upper left',
                #bbox_to_anchor=(1.45,0.85),
                #bbox_transform=afig.transFigure,
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

        cx.set_xlabel(r'$h/J$',fontsize=20)
        cx.set_ylabel(r'$m/(S\mathcal{N})$',fontsize=20)
        cx.set_xlim(left=0)
        #cx.set_ylim(bottom=0,top=1)
        cx.set_ylim(bottom=0)

        cfig.savefig("figs/stag_magnetization-kappa-"+str("%.2f" % kappa)+".pdf",
                     bbox_inches='tight'
                     )



        plt.close('all')


