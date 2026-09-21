#!/usr/bin/env python
import os
import glob
import time
import datetime

import numpy as np

path = 'run_data'

fstring = max(glob.glob(path+"/runningData_*.npz"),key=os.path.getmtime)[:-4]

params = np.load(fstring+".npz")

L = params['L']
SARR = params['SARR']
CONFIG = params['CONFIG']
hARR = params['hARR']

muARR = np.zeros((SARR.size,CONFIG.size,hARR.size,3),dtype=float)
QARR = np.zeros((SARR.size,CONFIG.size,hARR.size,3),dtype=complex)
ZARR = np.zeros((SARR.size,CONFIG.size,hARR.size,6),dtype=complex)
freeARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float)
SzARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float)
ncARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float)
errARR = np.zeros((SARR.size,CONFIG.size,hARR.size),dtype=float)

with open(fstring+".dat", "r") as file:
    for line in file:
        X = line.split()
        iS = int(X[0])
        ich = int(X[1])
        ih = int(X[2])
        SzARR[iS,ich,ih] = float(X[3])
        freeARR[iS,ich,ih] = float(X[4])
        ncARR[iS,ich,ih] = float(X[5])
        if 'SpinSL' in params:
            errARR[iS,ich,ih] = float(X[6])
            i = 7
        else:
            i = 6
        muARR[iS,ich,ih] = [float(x) for x in X[i:i+3]]
        QARR[iS,ich,ih] = [complex(x) for x in X[i+3:i+6]]
        ZARR[iS,ich,ih] = [complex(x) for x in X[i+6:i+12]]

TSTAMP = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y_%H-%M-%S')

if 'SpinSL' in params:

    np.savez_compressed('L_'+str(L)+'_TS_'+TSTAMP+'_'+'var1ch_data.npz',
                        SARR=SARR,
                        CONFIG=CONFIG,
                        hARR=hARR,
                        SpinSL=params['SpinSL'],
                        SpinCrit=params['SpinCrit'],
                        muARR=muARR,
                        QARR=QARR,
                        ZARR=ZARR,
                        freeARR=freeARR,
                        SzARR=SzARR,
                        ncARR=ncARR,
                        errARR=errARR,
                        L=L,
                        T=params['T'],
                        mfTol=params['mfTol'],
                        MaxIter=params['MaxIter']
                        )

else:

    np.savez_compressed('L_'+str(L)+'_TS_'+TSTAMP+'_'+'largeS_data.npz',
                        SARR=SARR,
                        hARR=hARR,
                        CONFIG=CONFIG,
                        muARR=muARR,
                        QARR=QARR,
                        ZARR=ZARR,
                        freeARR=freeARR,
                        SzARR=SzARR,
                        mARR=ncARR,
                        )
