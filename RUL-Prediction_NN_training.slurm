#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --time=00:10:00
#SBATCH --begin=now
#SBATCH --mail-user=paul.strahle@student.isae-supaero.fr
#SBATCH --mail-type=FAIL,END
#SBATCH --job-name=RUL-prediction_NN_training
#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR
#SBATCH --partition=p16
#SBATCH --priority=TOP

module load openmpi
conda init bash
conda activate RUL-Prediction
cd /scratch/students/p.strahle
mpirun -np 16 python PIR_CNN_Model3_5_script_distributed.py