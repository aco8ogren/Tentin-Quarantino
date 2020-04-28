#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=00:05:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH -A CS156b
#SBATCH -J "Test"   # job name
#SBATCH --mail-user=jlassman@caltech.edu   # email address

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source /central/home/jlassman/Documents/Environments/TFTenv/bin/activate
python3 ../Test.py > out.txt


# python3 ../Test.py