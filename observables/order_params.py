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
ncARR=params['ncARR']
errARR=params['errARR']
L = params['L']
# mfTol = params['mfTol']

mfTol = 1e-5

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

def Zravel(Z):
    
    z1 = np.array([Z[0],Z[3].conj()])
    z2 = np.array([Z[1],Z[4].conj()])
    z3 = np.array([Z[2],Z[5].conj()])

    return z1, z2, z3

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
    
    ChiralOP = np.zeros((SARR.size,hARR.size),dtype=float) 
    SzOP = np.zeros((SARR.size,hARR.size),dtype=float) 
    ncOP = np.zeros((SARR.size,hARR.size),dtype=float) 

    for iS in range(SARR.size):

        for ih in range(hARR.size):

            E0 = 1e4

            for ich in range(CONFIG.size):

                if CONFIG[ich] == "yv":

                    ich_min = ich

            z1, z2, z3 = Zravel(ZARR[iS,ich_min,ih])
            
            S1 = np.array([np.vdot(z1,np.dot(pauli[0],z1)),np.vdot(z1,np.dot(pauli[1],z1)),np.vdot(z1,np.dot(pauli[2],z1))])/2
            S2 = np.array([np.vdot(z2,np.dot(pauli[0],z2)),np.vdot(z2,np.dot(pauli[1],z2)),np.vdot(z2,np.dot(pauli[2],z2))])/2
            S3 = np.array([np.vdot(z3,np.dot(pauli[0],z3)),np.vdot(z3,np.dot(pauli[1],z3)),np.vdot(z3,np.dot(pauli[2],z3))])/2
            
            chi = np.abs(S1.dot(np.cross(S2,S3)))
            Sz = np.abs(np.sum(S1+S2+S3))/3

            ChiralOP[iS,ih] = chi/(SARR[iS]**3)
            SzOP[iS,ih] = Sz/SARR[iS]
            ncOP[iS,ih] = ncARR[iS,ich_min,ih]/(2*SARR[iS])

    
    #color_cycle =['white','bisque','cornflowerblue','tan','orange','crimson','yellow','pink']
    colors_cycle =['white','bisque','lime','lightgrey','forestgreen']

    cnum0 = 20
    cnum = 256
    #cnumuud = 40

    pdTol = 1e-4

    level1 = [i for i in np.linspace(0,pdTol,endpoint=False,num=cnum0)]
    level2 = [i for i in np.linspace(pdTol,1,endpoint=True,num=cnum)]
    #level2 = [i for i in np.linspace(pdTol,1/3-pdTol,endpoint=False,num=cnum)]
    #level3 = [i for i in np.linspace(1/3-pdTol,1/3+pdTol,endpoint=False,num=cnumuud)]
    #level4 = [i for i in np.linspace(1/3+pdTol,1-pdTol,endpoint=False,num=cnum)]

    levels = np.concatenate((level1,level2))
    
    color1 = ['white']*cnum0
    color2 = [mcolors.to_hex(color) for color in plt.cm.Blues(np.linspace(0, 1,cnum))]

    colors = np.concatenate((color1,color2))
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    
    lnum = len(levels)
    
    x = 1/(2*SARR)
    y = hARR
    hc = 2*gapDat/SARR
    x120 = x[hc==0]
    hc120 = hc[hc==0]

    xm, ym = np.meshgrid(x,y)
    
    zm = SzOP.T
    
    afig = plt.figure(figsize=(w,h))
    ax = afig.add_axes(frame)


    line = ax.plot(x,
                    hc,
                    #ms=10,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    label=r'$h_c(\kappa)$',
                    color='k',
                    lw=4,
                    linestyle='-',
                    #dashes=(2,2),
                    zorder=2,
                    )

    
    line = ax.plot(x120,
                    hc120,
                    #ms=10,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    #label=r'$120^\circ$',
                    color='r',
                    lw=6,
                    linestyle='-',
                    #dashes=(2,2),
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

    ax.text(1,4.5,'Canted \n Ordering',fontsize=22)
    ax.text(1,0.5,r'$120^\circ$',color='r',fontsize=22)
    ax.text(7,4,r'$\mathbb{Z}_2$ SL',color='blue',fontsize=22)
    
    ax.set_xlabel(r'$1/\kappa$',fontsize=22)
    ax.set_ylabel(r'$h/J$',fontsize=22)
    #x.set_xlim(left=0,right=10)
    ax.set_xlim(left=0,right=10)
    ax.set_ylim(bottom=0,top=9)

    afig.savefig("figs/pd-Sz-large-N.pdf",
                 bbox_inches='tight',transparent=True,
                 )
    
    plt.close('all')
    
    zm = ChiralOP.T

    afig, ax = plt.subplots(layout='constrained')
    aspect_ratio = 1.1
    ax.set_aspect(aspect=aspect_ratio)

    CSF = ax.pcolormesh(xm,ym,zm,
                        cmap=cmap,
                        #shading='gouraud',
                        vmin=zm.min(),
                        vmax=zm.max())
    
    cbar = afig.colorbar(CSF,
                         shrink=0.75
                         )
    cbar.ax.set_title(r'$\chi/(\kappa/2)^3$',fontsize=22)
    
    line = ax.plot(x,
                    hc,
                    #ms=10,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    label=r'$h_{\rm SL}$',
                    color='k',
                    lw=4,
                    linestyle='-',
                    #dashes=(2,2),
                    zorder=2,
                    )
    line = ax.plot(x120,
                    hc120,
                    #ms=10,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    label=r'$120^\circ$',
                    color='r',
                    lw=4,
                    linestyle='-',
                    #dashes=(2,2),
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

    ax.set_xlabel(r'$1/\kappa$',fontsize=22)
    ax.set_ylabel(r'$h/J$',fontsize=22)
    #x.set_xlim(left=0,right=10)
    ax.set_xlim(left=0,right=10)
    ax.set_ylim(bottom=0,top=9)

    afig.savefig("figs/pd-Chiral-large-N.pdf",
                 bbox_inches='tight',transparent=True,
                 )
    
    plt.close('all')
    
    #zm = ndimage.uniform_filter1d(ncOP,size=4,axis=-1).T
    zm = ncOP.T
    
    afig, ax = plt.subplots(layout='constrained')
    aspect_ratio = 1.1
    ax.set_aspect(aspect=aspect_ratio)
   
    lnum = len(levels)

    
    CSF = ax.contourf(xm,ym,zm,
                      levels=lnum-1,
                      #levels=40,
                      #antialiased=True,
                      #edgecolor='None',
                      #levels=levels,
                      #colors=colors[:lnum],
                        cmap=cmap,
                        #shading='gouraud',
                        vmin=0,
                        vmax=1,
                      #alpha=0,
                      )
    
    CSF = ax.contourf(xm,ym,zm,
                      levels=lnum-1,
                      #levels=40,
                      #antialiased=True,
                      #edgecolor='None',
                      #levels=levels,
                      #colors=colors[:lnum],
                        cmap=cmap,
                        #shading='gouraud',
                        vmin=0,
                        vmax=1,
                      #alpha=0,
                      )
     
    CSF = ax.contourf(xm,ym,zm,
                      levels=lnum-1,
                      #levels=40,
                      #antialiased=True,
                      #edgecolor='None',
                      #levels=levels,
                      #colors=colors[:lnum],
                        cmap=cmap,
                        #shading='gouraud',
                        vmin=0,
                        vmax=1,
                      #alpha=0,
                      )
 
    
    cbar = afig.colorbar(CSF,
                         shrink=0.75
                         )
   
   
    #cbar.set_ticks([0,pdTol,1/3-pdTol,1/2,1])
    cbar.set_ticks([0,1/4,1/2,3/4,1])
    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.2f'))


    cbar.ax.set_title(r'$n_c/\kappa$',fontsize=22,pad=15)
    
    line = ax.plot(x,
                    hc,
                    #ms=10,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    label=r'$h_c/J$',
                    color='k',
                    lw=4,
                    linestyle='-',
                    #dashes=(2,2),
                    zorder=2,
                    )
    
    line = ax.plot(x120,
                    hc120,
                    #ms=10,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    label=r'$120^\circ$',
                    color='r',
                    lw=4,
                    linestyle='-',
                    #dashes=(2,2),
                    zorder=3,
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

    ax.set_xlabel(r'$1/\kappa$',fontsize=22)
    ax.set_ylabel(r'$h/J$',fontsize=22)
    #x.set_xlim(left=0,right=10)
    ax.set_xlim(left=0,right=10)
    ax.set_ylim(bottom=0,top=9)
    ax.set_yticks(hAXIS)

    afig.savefig("figs/pd-Condensate-large-N.pdf",
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
    
    afig = plt.figure(figsize=(w,h))
    ax = afig.add_axes(frame)
    
    line = ax.plot(1/(2*ScARR[:,0]),
                   hARR,
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
     
    line = ax.plot(x120,
                    hc120,
                    #ms=10,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    label=r'$120^\circ$',
                    color='r',
                    lw=4,
                    linestyle='-',
                    #dashes=(2,2),
                    zorder=2,
                    )

       
    ax.set_ylabel(r'$h_c/J$',fontsize=22)
    ax.set_xlabel(r'$1/\kappa$',fontsize=22)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    afig.savefig("figs/h_crit.pdf",
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
        mx.set_xlim(left=0,right=9)
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

            #if CONFIG[ich] == 'uud':

            #    continue
            
            lstyle = ['--',':','-']
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
                
                line = ax.plot(hARR[hARG][:-1],
                               En[hARG][:-1],
                               ms=10,
                               #marker=next(markers),
                               marker=markers2[ich],
                               fillstyle='none',
                               label=LABEL[ich],
                               lw=4,
                               linestyle=lstyle[ich],
                               #zorder=2-ich,
                               alpha=alpha[ich]
                               )

                line = bx.plot(hARR[hARG][:-1],
                               Sz[hARG][:-1],
                               ms=10,
                               #marker=next(markers),
                               marker=markers2[ich],
                               fillstyle='none',
                               label=LABEL[ich],
                               lw=4,
                               linestyle=lstyle[ich],
                               #zorder=2-ich,
                               alpha=alpha[ich]
                               )

                line = cx.plot(hARR[hARG][:-1],
                               nc[hARG][:-1],
                               ms=10,
                               #marker=next(markers),
                               marker=markers2[ich],
                               fillstyle='none',
                               label=LABEL[ich],
                               lw=4,
                               linestyle=lstyle[ich],
                               #zorder=2-ich,
                               alpha=alpha[ich]
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

        plt.close('all')
