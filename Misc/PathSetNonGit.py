import os 
import sys
HomeDIR='Tentin-Quarantino'
wd=os.getcwd()
DIR=wd[:wd.find(HomeDIR)+len(HomeDIR)]
os.chdir(DIR)