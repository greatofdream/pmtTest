'''
store the result of analysis into Database (Currently use csv).
'''
import argparse
import pandas as pd
import h5py
import config
from csvDatabase import OriginINFO
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input PMT summary csv')
psr.add_argument('-o', dest='opt', help='output csv')
psr.add_argument('--pmt', help='run number')
psr.add_argument('--dark', help='dark result')
psr.add_argument('--laser', help='laser result')
args = psr.parse_args()
with h5py.File(args.laser, 'r') as ipt:
    pulse = ipt['mergepulse'][:]
storecsv = pd.read_csv(args.opt)
tmpcsv = storecsv.set_index('PMT')
tmpcsv.loc[args.pmt, 'Pre'] = pulse[3]
tmpcsv.loc[args.pmt, 'After1'] = pulse[4]
tmpcsv.loc[args.pmt, 'After2'] = pulse[5]
tmpcsv.sort_index().reset_index().to_csv(args.opt, index=False)