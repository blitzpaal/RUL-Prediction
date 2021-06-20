#!/bin/sh

## Specify the name for your job, this is the job name by which Slurm will
## refer to your job.  This can be different from the name of your executable
## or the name of your script file.
#SBATCH --job-name RUL-Prediction

#SBATCH --qos normal  # normal,cdsqos,phyqos,csqos,statsqos,hhqos,gaqos,esqos
#SBATCH --partition=gpu       # partition (queue): all-LoPri, all-HiPri,
                      #   bigmem-LoPri, bigmem-HiPri, gpuq, CS_q, CDS_q, ...

## Deal with output and errors.  Separate into 2 files (not the default).
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayMain, %a=arraySub
#SBATCH -o /scratch/students/p.strahle/%x-%N-%j.out    # Output file
#SBATCH -e /scratch/students/p.strahle/%x-%N-%j.err    # Error file
#SBATCH --mail-type=BEGIN,END,FAIL     # NONE,BEGIN,END,FAIL,REQUEUE,ALL,...
#SBATCH --mail-user=paul.strahle@student.isae-supaero.fr   # Put your GMU email address here

## Specifying an upper limit on needed resources will improve your scheduling
## priority, but if you exceed these values, your job will be terminated.
## Check your "Job Ended" emails for actual resource usage info.
##SBATCH --mem=190G          # Total memory needed for your job (suffixes: K,M,G,T)
#SBATCH --time=0-24:00    # Total time needed for your job: Days-Hours:Minutes

## These options are more useful when running parallel and array jobs
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --ntasks-per-node=12
##SBATCH --exclusive
#SBATCH --gres=gpu:1

## Load the relevant modules needed for the job
module load cuda/11.0
conda activate RUL-Prediction

## Start the job
srun -n 1 python PIR_CNN_Model1_3_script_distributed_adaptiveLR.py
