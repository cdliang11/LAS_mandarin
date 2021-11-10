export SEE_ROOT=$PWD/../../..

export PATH=$SEE_ROOT/utils:$PWD/utils/:$PATH

export LC_ALL=C

### Python
# source $CONDA/etc/profile.d/conda.sh && conda deactivate && conda activate
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1