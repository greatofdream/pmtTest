'''
store the result of PDE into Database (Currently use csv).
'''
import argparse
import pandas as pd, numpy as np
import h5py
import config
from csvDatabase import OriginINFO
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt')
psr.add_argument('--runs', nargs='+', help='run number')
psr.add_argument('--calibcsv', help='dark result')
psr.add_argument('--csv', help='laser result')
args = psr.parse_args()
with h5py.File(args.ipt, 'r') as ipt:
    ratio = ipt['splitter'][:]
    pdes = ipt['QE'][:]
# storage calibratio
calibDF = pd.read_csv(args.calibcsv)
calibDF = calibDF.set_index('RUNNOS')
calibDF.loc['-'.join(args.runs)] = np.insert(ratio[0], 0, 1)
calibDF.reset_index().to_csv(args.calibcsv, index=False)
# store the PDE
storecsv = pd.read_csv(args.csv)
# 需要按字母排序
pmts = np.unique(storecsv.set_index('RUNNO').loc[int(args.runs[-1])]['PMT'])
testpmts = []
for pmt in pmts:
    if pmt.startswith('PM'):
        testpmts.append(pmt)
print(list(zip(testpmts, pdes)))
for pmt,pde in zip(testpmts, pdes):
    storecsv.loc[(storecsv['PMT']==pmt)&(storecsv['RUNNO']==int(args.runs[-1])),'PDE'] = pde
storecsv.to_csv(args.csv)