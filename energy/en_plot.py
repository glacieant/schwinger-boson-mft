#!/usr/bin/env python
import os
import sys
import multiprocessing as mp
import itertools
import time

import numpy as np
from numpy import linalg

from scipy.linalg import ldl
from scipy.linalg import cholesky
from scipy import integrate
from scipy.linalg import null_space
from scipy import optimize
from scipy.linalg import eigvals
from scipy.linalg import eigvalsh
from scipy.signal import resample

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


path = 'figs'
if not os.path.exists(path):
    os.makedirs(path)

#plot routines

# palette
plt.style.use('tableau-colorblind10')

if os.path.isfile('sb_fluctdata_contour.npz'):

    params = np.load('sb_fluctdata_contour.npz')
    SARR = params['SARR']
    CONFIG = params['CONFIG']
    hARR=params['hARR']
    QARR=params['QARR']
    ZARR=params['ZARR']
    muARR=params['muARR']
    freeARR=params['freeARR']
    freeARR2=params['freeARR2']

params = np.load('var1ch_data.npz')

mfTol = params['mfTol']
errARR = params['errARR']
SzARR = params['SzARR']
SpinSL = params['SpinSL']

def plot_and_save(spSARR,SpinSL):

    ## plot routines

    # palette
    plt.style.use('tableau-colorblind10')

    # marker strings

    markers = itertools.cycle(['o','s','^','D','X','p','H','P',
                               "*","|","1"])

    #markers = ['o','s','^','D','X','p','H','P',
    #    "*","|","1","+"]

    #colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']

    ## mathtext style
    plt.rcParams['mathtext.fontset']='cm'
    plt.rc('font',family='serif')

    hAXIS = [0,2,4,6,8,9]
    
    modCONFIG = [None]*len(CONFIG)
    modCONFIG2 = [None]*len(CONFIG)
    for iconf in range(len(CONFIG)):
        conf = CONFIG[iconf]
        if conf == "yv":
            modCONFIG[iconf] = "coplanar"
            modCONFIG2[iconf] = "cop./col."
        elif conf == "umbrella":
            modCONFIG[iconf] = "noncoplanar"
            modCONFIG2[iconf] = "noncop."
        elif conf == "uud":
            modCONFIG[iconf] = "UUD"
            modCONFIG2[iconf] = "cop./col."

    effSzARR = np.zeros((SARR.size,hARR.size),dtype=float)
    effEnARR = np.zeros((SARR.size,hARR.size),dtype=float)
 
    #indxSARR = np.array([np.argwhere(SARR==0.5),np.argwhere(SARR==1.0)]).flatten()
    #indxSARR = np.array([np.argwhere(SARR==0.25)]).flatten()
    #print(CONFIG)
    #input()
   
    #ichYV = np.argwhere((CONFIG=="yv"))[0,0]
    SzARR2 = np.zeros_like(freeARR2)


    for iS in range(SARR.size):
    #for iS in indxSARR:

        ichYV = np.argwhere((CONFIG=="yv"))[0,0]
        SzARR2[iS,ichYV] = -(np.gradient(freeARR2[iS,ichYV],hARR,edge_order=2)/SARR[iS])/SARR[iS]
        
        ichUUD = np.argwhere((CONFIG=="uud"))[0,0]
        SzARR2[iS,ichUUD] = -(np.gradient(freeARR2[iS,ichUUD],hARR,edge_order=2)/SARR[iS])/SARR[iS]
        
        ichUMB = np.argwhere((CONFIG=="umbrella"))[0,0]
        SzARR2[iS,ichUMB] = -(np.gradient(freeARR2[iS,ichUMB],hARR,edge_order=2)/SARR[iS])/SARR[iS]

        for ih in range(hARR.size):

            ix = np.argmin(freeARR2[iS,:,ih])
            #effSzARR[iS,ih] = SzARR2[iS,ix,ih]
            
            if freeARR[iS,ix,ih] == np.nan:
                effEnARR[iS,ih] = (freeARR2[iS,ichYV,ih])
            else:
                effEnARR[iS,ih]=(freeARR2[iS,ix,ih])
            
        effSzARR[iS] = -(np.gradient(effEnARR[iS],hARR,edge_order=2)/SARR[iS])/SARR[iS]

        for ih in range(hARR.size):

            if effSzARR[iS,ih] == np.nan and SzARR2[iS,ix,ih] == np.nan:

                effSzARR[iS,ih] = SzARR[iS,ix,ih]

            elif effSzARR[iS,ih] == np.nan and SzARR2[iS,ix,ih] != np.nan:
        
                effSzARR[iS,ih] = SzARR2[iS,ix,ih]
    for iS in range(SARR.size):
    #for iS in indxSARR:

        kappa = 2*SARR[iS]

        afig, ax = plt.subplots()
        bfig, bx = plt.subplots()

        line = ax.plot(hARR,
                       effSzARR[iS],
                       #color=colors[ich],
                       ms=8,
                       marker='s',
                       mfc='None',
                       #marker=next(markers),
                       # mew=1,
                       #mfc=colors[ich],
                       #label=modCONFIG[ich],
                       #lw=4,
                       linestyle='--',
                       #zorder=ich,
                       #linestyle='none',
                       clip_on = False,
                       )

        line = bx.plot(hARR,
                       effEnARR[iS],
                       #color=colors[ich],
                       ms=10,
                       marker='o',
                       mfc='None',
                       # mew=1,
                       #mfc=colors[ich],
                       #label=modCONFIG[ich],
                       lw=4,
                       linestyle='-',
                       #zorder=ich,
                       )
        
        ax.set_clip_on(False)
        ax.tick_params(which='major', width=2,
                       labelsize=22, direction='in',
                       length=10,
                       # pad=15,
                       bottom=True, top=False,
                       left=True, right=False)

        #bx.set_title(r'Energy Difference, $\kappa = $'+str("%.2f" % kappa),fontsize=22)
        ax.set_xlabel(r'$h/J$', fontsize=22)
        ax.set_ylabel(r'$m^{(0)}+m^{(1)}$', fontsize=22)
        #ax.set_xlim(left=0)
        ax.set_xlim(left=0,right=np.amax(hARR))
        ax.set_ylim(bottom=0)
        #ax.set_xticks(hAXIS)
        ax.set_box_aspect(1)

        afig.savefig("figs/SzFluct-kappa-"+str("%.2f" % kappa)+".pdf",
                     bbox_inches='tight',transparent=True,
         )
        
        bx.tick_params(which='major', width=2,
                       labelsize=18, direction='in',
                       length=10,
                       # pad=15,
                       bottom=True, top=False,
                       left=True, right=False)

 
        bx.set_xlabel(r'$h/J$', fontsize=22)
        bx.set_ylabel(r'$f$', fontsize=22)
        #bx.set_xlim(left=0,right=9)
        bx.set_xlim(left=0,right=np.amax(hARR))
        ax.set_ylim(bottom=0)
        bx.set_xticks(hAXIS)


        bfig.savefig("figs/EnFluct-kappa-"+str("%.2f" % kappa)+".pdf",
                     bbox_inches='tight',transparent=True,
         )

        plt.close('all')
        
        frame = [0.2,0.2,0.8,0.8]
        w,h = figure.figaspect(1.0)
        
        afig, ax = plt.subplots()
        zorder = np.zeros(3)
        lst = [None]*3
        ich = np.argwhere((CONFIG=="uud"))[0,0]
        zorder[ich] = 3
        lst[ich] = 'solid'
        ich = np.argwhere((CONFIG=="yv"))[0,0]
        zorder[ich] = 2
        lst[ich] = 'dashed'
        ich = np.argwhere((CONFIG=="umbrella"))[0,0]
        zorder[ich] = 1
        lst[ich] = 'dotted'

        ich = np.argwhere((CONFIG=="uud"))[0,0]
        EnUUD = freeARR2[iS,ich,:]
        ich = np.argwhere((CONFIG=="yv"))[0,0]
        EnYV = freeARR2[iS,ich,:]
        ich = np.argwhere((CONFIG=="umbrella"))[0,0]
        EnUMB = freeARR2[iS,ich,:]

        Y1 = EnUUD - EnYV
        Y2 = EnUMB - EnYV

        line = ax.plot(hARR,
                       Y1,
                       #ms=10/4,
                       #marker=next(markers),
                       #mfc='none',
                       # mew=1,
                       label=r'$\varepsilon^{\rm UUD}_0-\varepsilon^{\rm cop.}_0$',
                       lw=4,
                       linestyle='solid',
                       # zorder=ich,
                       )

        
        line = ax.plot(hARR,
                       Y2,
                       #ms=10/4,
                       #marker=next(markers),
                       #mfc='none',
                       # mew=1,
                       label=r'$\varepsilon^{\rm noncop.}_0-\varepsilon^{\rm cop.}_0$',
                       lw=4,
                       linestyle='dotted',
                       # zorder=ich,
                       )

       
        line = ax.plot(hARR,
                       np.zeros(hARR.size),
                       #ms=10/4,
                       #marker=next(markers),
                       # mew=1,
                       lw=2,
                       linestyle='dashed',
                       color='k',
                       alpha=0.5,
                       # zorder=ich,
                       )


        ax.tick_params(which='major', width=2,
                       labelsize=22, direction='in',
                       length=10,
                       # pad=15,
                       bottom=True, top=False,
                       left=True, right=False)

        legend = ax.legend(
             #loc='best',
                loc='upper right',
            # bbox_to_anchor=(1.45,0.85),
                # bbox_transform=afig.transFigure,
                fontsize=22,
                # markerscale=0.75,
                facecolor='w',
                edgecolor='k',
                framealpha=1,
                # ncol=1,
                # borderpad=0.5,
                # handletextpad=0.5,
                # handlelength=0.4,
                # labelspacing=0.4,
                # frameon=False
        )

        #ax.set_title(r'Free energy, $\kappa = $'+str("%.2f" % kappa),fontsize=22)
        ax.set_xlabel(r'$h/J$', fontsize=22)
        ax.set_ylabel(r'$\Delta\varepsilon^{(0)}_0+\Delta\varepsilon_0^{(1)}$', fontsize=22)
        Ymax = Y2[0]*(1+1/10)
        ax.set_ylim(top=Ymax,bottom=-Ymax/2)
        ax.set_xlim(left=0,right=np.amax(hARR))
        #ax.set_xticks(hAXIS)
        ax.set_box_aspect(1)

        afig.savefig("figs/FreeEnFluct-kappa-"+str("%.2f" % kappa)+"-contour.pdf",
                     bbox_inches='tight',transparent=True,
                     )
        

        plt.close()
    
    gapDat = SpinSL[:,2]

    hAXIS = [0,2,4,6,8,9]
    
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
    
    #color_cycle =['white','bisque','cornflowerblue','tan','orange','crimson','yellow','pink']
    colors_cycle =['white','bisque','lime','lightgrey','forestgreen']

    colors10 = ['#006BA4', '#FF800E', '#ABABAB','#595959',
                 '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

    tab10 = {
    "Blue": "#1F77B4",
    "Orange": "#FF7F0E",
    "Green": "#2CA02C",
    "Red": "#D62728",
    "Purple": "#9467BD",
    "Brown": "#8C564B",
    "Pink": "#E377C2",
    "Gray": "#7F7F7F",
    "Olive": "#BCBD22",
    "Cyan": "#17becf"
    }

    frame = [0.2,0.2,0.8,0.8]
    w,h = figure.figaspect(1.0)

    afig = plt.figure(figsize=(w,h))
    ax = afig.add_axes(frame)
   
    x = np.append(1/(2*spSARR),[0])
    hzero = np.zeros(x.size)
    hc = np.append(2*gapDat/spSARR,[0])
    x120 = x[hc==0]
    hc120 = hc[hc==0]

    line = ax.plot(x,
                    hc,
                    #ms=10,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    label=r'$h_c/J$',
                    color=tab10['Blue'],
                    lw=2,
                    linestyle='--',
                    #dashes=(2,2),
                    zorder=4,
                    )

    # arrows

    #arrow1color = colors10[0]
    #arrow2color = colors10[1]
    #arrow3color = colors10[2]
    
    arrow1color = 'r'
    arrow2color = 'b'
    arrow3color = 'k'
    
    dw = 0.75
    
    arrow_1 = ax.arrow(.9+2.1-.7, 0.5+1.4, dw/(3**0.5), dw/(3**0.5), width=0.05, color=arrow1color,zorder=4)
    arrow_2 = ax.arrow(.9+1.9-.7, 0.5+1.4, -dw/(3**0.5), dw/(3**0.5), width=0.05, color=arrow2color,zorder=4)
    arrow_3 = ax.arrow(.9+2.-.7, 0.5+1.3, 0, -dw, width=0.05, color=arrow3color,zorder=4)
    
    dw = 0.75
    
    arrow_1 = ax.arrow(3.5-0.2, 6.25+1.3-0.2, 0, dw, width=0.05, color=arrow1color,zorder=4)
    arrow_2 = ax.arrow(3.3-0.2,6.25+ 1.3-0.2, 0, dw, width=0.05, color=arrow2color,zorder=4)
    arrow_3 = ax.arrow(3.4-0.2,6.25+ 1.2-0.2, 0, -dw, width=0.05, color=arrow3color,zorder=4)
    
    dw = 0.5
    
    arrow_1 = ax.arrow(1.1-0.4, 10.6, 0, dw, width=0.05, color=arrow1color,zorder=4)
    arrow_2 = ax.arrow(.7-0.4, 10.6, 0, dw, width=0.05, color=arrow1color,zorder=4)
    arrow_3 = ax.arrow(.9-0.4, 10.6, 0, dw, width=0.05, color=arrow1color,zorder=4)

    dw = 0.75
    
    arrow_1 = ax.arrow(0.9+.6, 3+5.9, dw/(2**0.5), dw/(2**0.5), width=0.05, color=arrow1color,zorder=4)
    arrow_2 = ax.arrow(0.9+.4, 3+5.9, -dw/(2**0.5), dw/(2**0.5), width=0.05, color=arrow2color,zorder=4)
    arrow_3 = ax.arrow(0.9+.6, 3+6.1, dw/(2**0.5), dw/(2**0.5), width=0.05, color=arrow3color,zorder=4)
    

    ax.text(0.5,-0.75,r'$120^\circ$',color='red',fontsize=22,zorder=4)
    ax.text(3,3.2,r'Y',fontsize=22,zorder=4)
    ax.text(2.95,9.,r'UUD',fontsize=22,zorder=4)
    ax.text(.5,1.5+5.8,r'V',fontsize=22,zorder=4)
    ax.text(0.38,11.5,r'FP',fontsize=12,zorder=4)
    ax.text(6.4,6.5,r'$\mathbb{Z}_2$ SL',fontsize=22,zorder=4)
   
    line = ax.fill_between(x,
                           hzero,
                           hc,
                    #color='bisque',
                           lw=0,
                           alpha=0.25,
                           zorder=1,
                    )

    line = ax.plot(x120,
                    hc120,
                    #ms=10,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    #label=r'$120^\circ$',
                    color=tab10['Red'],
                    lw=4,
                    linestyle='-',
                    #dashes=(2,2),
                    zorder=4,
                    )


    UUD1 = np.full(SARR.size,np.nan)
    UUD2 = np.full(SARR.size,np.nan)
    VCOP = np.full(SARR.size,np.nan)
    POLAR = np.full(SARR.size,np.nan)
    MAX = np.full(SARR.size,15)
    

    for iS in range(SARR.size):

        SzX = effSzARR[iS]
        #SzX = SzX[~np.isnan(SzX)]

        ystate = np.array(np.nonzero(SzX < 0.3332))
        uud = np.array(np.nonzero((SzX > 0.3332) & (SzX < 0.3334)))
        vstate = np.array(np.nonzero((SzX > 0.3334) & (SzX < 0.9995)))
        polar = np.array(np.nonzero((SzX>0.9995)))

        if polar.size != 0:

            POLAR[iS] = hARR[np.amin(polar)]

        if uud.size != 0:

            UUD1[iS] = hARR[np.amin(uud)]
            UUD2[iS] = hARR[np.amax(uud)]

        if vstate.size != 0:

            VCOP[iS] = hARR[np.amin(vstate)]


    xp = np.append(1/(2*SARR),[0])
    UUD1 = np.append(UUD1,[3])
    UUD2 = np.append(UUD2,[3])
    VCOP = np.append(VCOP,[3])
    POLAR = np.append(POLAR,[9])
    MAX = np.append(MAX,[15])
    
    line = ax.fill_between(x,
                           hc,
                    np.full(x.size,15),
                    color='bisque',
                           lw=0,
                           alpha=0.25,
                           zorder=1
                    )
    
    lineUUD = ax.fill_between(xp,UUD1,UUD2,alpha=0.5,lw=2,color=tab10['Orange'],zorder=1)
    lineUUDMAX = ax.fill_between(xp,np.fmin(UUD2,VCOP),MAX,alpha=0.5,lw=0,color=tab10['Gray'],zorder=2)
    #lineUUDMAX = ax.fill_between(xp,VCOP,MAX,alpha=0.5,lw=0,color=tab10['Gray'],zorder=2)
    linePOLMAX = ax.fill_between(xp,POLAR,MAX,alpha=1,lw=2,zorder=3,color='white',
                                 edgecolor=tab10['Gray'])
    
    line = ax.plot(xp,
                    UUD1,
                    ms=6,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    marker='s',
                    label=r'$h^{\pm}_{\rm UUD}/J$',
                    color=tab10['Orange'],
                    lw=2,
                    linestyle='-',
                    #dashes=(2,2),
                    zorder=4,
                    )
    
    line = ax.plot(xp,
                    UUD2,
                    ms=6,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    marker='s',
                    #label=r'$h_c/J$',
                    #label=r'$h_c/J$',
                    color=tab10['Orange'],
                    lw=2,
                    linestyle='-',
                    #dashes=(2,2),
                    zorder=4,
                    )
    
    line = ax.plot(xp,
                    POLAR,
                    ms=6,
                    #mew=1,
                    #mfc='None',
                    #marker=next(markers),
                    marker='o',
                    label=r'$h_{\rm FP}/J$',
                    #label=r'$h_c/J$',
                    color=tab10['Gray'],
                    lw=2,
                    linestyle='-',
                    #dashes=(2,2),
                    zorder=4,
                    )
    
    ax.tick_params(which='major',width=2,
                   labelsize=22,direction='out',
                   length=10,
                   pad=10,
                   bottom=True,top=False,
                   left=True,right=False,
                   zorder=10)
    
    legend = ax.legend(
            #loc='best',
            loc='lower right',
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
    ax.set_ylim(bottom=0,top=hARR.max())
    ax.set_xlim(left=0,right=8)

    afig.savefig("figs/pd-Sz-fluct.pdf",
                 bbox_inches='tight',transparent=True,
                 )
    
    plt.close('all')
   

if __name__=='__main__':

    #with mp.Pool() as pool:
    #    SpinCrit = np.array(pool.starmap(critS,[(h,L,) for h in hARR]))


    if os.path.isfile('sb_fluctdata_contour.npz'):

        plot_and_save(SARR,SpinSL)
