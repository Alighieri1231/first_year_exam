#!/bin/bash

#SBATCH -p reserved --reservation=bcastane_12222024
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --output=test_reserved2.log

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

export PYTHONPATH=$PYTHONPATH:/gpfs/fs2/scratch/bcastane_lab/eochoaal/first_year_exam/src

wandb login 99d93a8c3ae2885fe88d1d7a5b3d2892f535e967

#run the script

echo "=== Lanzando todas las corridas de ASSGAN ==="
# Asegúrate de que run_experiments.sh es ejecutable (chmod +x)
# y está en la ruta correcta, o usa la ruta absoluta:
srun bash /gpfs/fs2/scratch/bcastane_lab/eochoaal/first_year_exam/scripts/run_assgan.sh

