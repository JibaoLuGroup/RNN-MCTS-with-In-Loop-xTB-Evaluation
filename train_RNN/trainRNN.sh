#!/bin/bash -l
#SBATCH --job-name=train_RNN    # Job name
#SBATCH --output=train_RNN_%j.out # Stdout (%j expands to jobId)
#SBATCH --error=train_RNN_%j.err # Stderr (%j expands to jobId)
#SBATCH -N 1 	# The number of Nodes to be assigned
#SBATCH --gres=gpu:1	# Number of processes
#SBATCH -p gpu4-80GB 

##export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Clean up loaded modules
module purge
# Load necessary modulesc

source /home/g5h6i/soft/anaconda3/ENTER/etc/profile.d/conda.sh
conda activate chemtsv2_3

# Run your command
python train_RNN.py -c model_setting.yaml
