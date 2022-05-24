#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --output=train_%j.txt
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=100G

conda activate my_env
srun python3 train_classifier_slurm.py -cp configs -cn config_cifar.yml
