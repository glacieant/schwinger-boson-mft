#!/bin/bash

##################################################
# triangular lattice large-N mean-field          #
# local pipeline                                 #
##################################################

# script parameters
# data and logs in data
# figures in figs

# chain
CHAIN=run
#CHAIN=response
# solver
SOLVER=torch
#SOLVER=remake
# skip for replots
#SOLVER=skip
# structure factor
RESPONSE=numpy
#RESPONSE=torch
# large-S background
LARGE_S=0
#LARGE_S=1
#LARGE_S=remake

set -e
cd "$(dirname "$0")"
mkdir -p data figs

# virtual environment
python3 -m venv ENV
source ENV/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# cython kernel
(cd energy/cython && python setup.py build_ext --inplace)
cp energy/cython/cython_en1ch*.so energy/

# working directory data
cd data

# solver
if [ "$SOLVER" != skip ]; then
    python ../variation/var_$SOLVER.py
    cp "$(ls -t L_*_var1ch_data.npz | head -1)" var1ch_data.npz
fi

if [ "$CHAIN" = run ]; then

    # observables
    python ../observables/order_params.py
    python ../observables/spectra.py

    # large-S background
    if [ "$LARGE_S" != 0 ]; then
        if [ "$LARGE_S" = remake ]; then
            python ../variation/var_remake.py
        else
            python ../variation/var_large_S.py
        fi
        cp "$(ls -t L_*_largeS_data.npz | head -1)" largeS_data.npz
        python ../observables/order_params_large_S.py
    fi

    # energy correction
    python ../energy/en_cython.py
    python ../energy/en_plot.py

fi

if [ "$CHAIN" = response ]; then

    # structure factor
    if [ "$RESPONSE" = torch ]; then
        python ../observables/make_strfc_torch.py
        python ../observables/plot_strfc_torch.py
    else
        python ../observables/make_strfc.py
        python ../observables/plot_strfc.py
    fi

fi

# figures
cp -a figs/. ../figs/
rm -r figs
