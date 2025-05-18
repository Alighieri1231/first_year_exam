#!/bin/bash

#SBATCH -p reserved --reservation=bcastane_12222024
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=test_reserved1.log

source /software/anaconda3/5.3.0b/bin/activate castane_lab

#set the library cuda to priority
export LD_LIBRARY_PATH=/scratch/bcastane_lab/lab-conda/envs/castane_lab/lib:$LD_LIBRARY_PATH

# Force PyTorch to use a specific CUDA runtime if needed
export CUDA_HOME=/scratch/bcastane_lab/lab-conda/envs/castane_lab

echo "Using LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Using CUDA_HOME: $CUDA_HOME"

echo 'Node memory info:'
free -h

nvidia-smi

srun python test.py

