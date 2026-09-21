#!/bin/bash
#SBATCH --account=def-maciejko

# solver
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=168:00:00
#SBATCH --job-name=var
# large-S solver
##SBATCH --nodes=2
##SBATCH --ntasks-per-node=64
##SBATCH --cpus-per-task=1
##SBATCH --time=03:00:00
##SBATCH --job-name=largeS
# energy correction
##SBATCH --nodes=20
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=64
##SBATCH --time=12:00:00
##SBATCH --job-name=fluct
# structure factor
##SBATCH --nodes=12
##SBATCH --ntasks-per-node=48
##SBATCH --cpus-per-task=1
##SBATCH --time=12:00:00
##SBATCH --job-name=strfc

#SBATCH --output=output.log

##################################################
# triangular lattice large-N mean-field          #
# cluster stages                                 #
##################################################

# mode cluster in the script
# mpi4py imports in the script
# stage
STAGE=solver
#STAGE=large_S
#STAGE=energy
#STAGE=response

##module load CCEnv arch/avx512
##module load StdEnv
##module load python/3.11.2
module load python/3.10
module load scipy-stack
module load mpi4py

# virtual environment
python -m venv ENV
source ENV/bin/activate

pip install --no-index --upgrade pip
pip install --no-index --upgrade -r requirements.txt

# submission from the repository root
# run.sh with SOLVER=skip for replots
cd "$SLURM_SUBMIT_DIR"
mkdir -p data

# cython kernel
(cd energy/cython && python setup.py build_ext --inplace)
cp energy/cython/cython_en1ch*.so energy/

# working directory data
cd data

if [ "$STAGE" = solver ]; then
    mpiexec -n 768 python -m mpi4py.futures ../variation/var_torch.py
    cp "$(ls -t L_*_var1ch_data.npz | head -1)" var1ch_data.npz
fi
if [ "$STAGE" = large_S ]; then
    mpiexec -n 128 python -m mpi4py.futures ../variation/var_large_S.py
    cp "$(ls -t L_*_largeS_data.npz | head -1)" largeS_data.npz
fi
if [ "$STAGE" = energy ]; then
    mpiexec -n 20 --bind-to none --map-by ppr:1:node python ../energy/en_cython.py
fi
if [ "$STAGE" = response ]; then
    mpiexec -n 576 python -m mpi4py.futures ../observables/make_strfc_torch.py
fi
