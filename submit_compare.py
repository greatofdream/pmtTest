#!/usr/bin/env python

import argparse, config, pandas as pd
import os
psr = argparse.ArgumentParser()
psr.add_argument('--phase', default='sk7', help='phase')
psr.add_argument('--datadir', help='data dir')
psr.add_argument('--mcdir', help='mcdir')
psr.add_argument('--summary', help='summary file')
args = psr.parse_args()

# Loop for runs
run_list = pd.read_csv(args.summary, sep=' ', names=config.dat_header)
run_list['LinacRunNo'] = run_list['LinacRunNo'].astype(pd.Int64Dtype())
run_list['NormalRunNo'] = run_list['NormalRunNo'].astype(pd.Int64Dtype())
for index, row in run_list.iterrows():
    run_this = row['LinacRunNo']
    mode_this = row['RunMode']
    normal_this = row['NormalRunNo']

    # Select mode=0
    if mode_this != 0:
        continue

    # Make a script file
    command = 'make phase={} datadir={} mcdir={} run={:06d} compareDataMCone -f compare.mk'.format(args.phase, args.datadir, args.mcdir, run_this) 
    # Submit a job
    print(command)
    os.system(command)
