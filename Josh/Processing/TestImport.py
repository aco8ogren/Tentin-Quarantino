import os
import numpy as np

DIR='Tentin-Quarantino'
direc=os.getcwd()
TQ_dir=direc[:direc.find(DIR)+len(DIR)]
os.chdir(TQ_dir)


data=np.loadtxt('data/us/covid/nyt_us_states.csv',dtype=str,delimiter=',')


