#!/bin/bash

#SBATCH -p reserved --reservation=bcastane_12222024
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --output=test_reserved.log

source /software/anaconda3/5.3.0b/bin/activate /scratch/eochoaal/my-conda/lightning

#set the library cuda to priority
export LD_LIBRARY_PATH=/scratch/eochoaal/my-conda/lightning/lib:$LD_LIBRARY_PATH

# Force PyTorch to use a specific CUDA runtime if needed
export CUDA_HOME=/scratch/eochoaal/my-conda/lightning

echo "Using LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Using CUDA_HOME: $CUDA_HOME"

echo 'Node memory info:'
free -h

echo "CPU information:"
lscpu

nvidia-smi

export PYTHONPATH=$PYTHONPATH:/gpfs/fs2/scratch/eochoaal/ligthing-template/src

wandb login 99d93a8c3ae2885fe88d1d7a5b3d2892f535e967
srun python src/train.py fit --config=configs/unet.yaml --trainer.logger=WandbLogger

