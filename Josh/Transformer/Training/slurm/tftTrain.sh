#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=16   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -A CS156b
#SBATCH -J "CountyTraining"   # job name
#SBATCH --mail-user=jlassman@caltech.edu   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source /central/home/jlassman/Documents/Environments/TFTenv/bin/activate
python3 ../CountyTraining.py >> CountyTraining.out