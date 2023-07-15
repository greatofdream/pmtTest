#!/usr/bin/env python

import argparse, config, pandas as pd
import os, time, subprocess, glob
psr = argparse.ArgumentParser()
psr.add_argument('--phase', default='sk7', help='phase')
psr.add_argument('-o', dest='opt', help='output directory')
psr.add_argument('--summary', help='summary file')
psr.add_argument('--card', help='MC card')
args = psr.parse_args()
# Set parameters
phase_setting = config.setting[args.phase]
# dir_make = phase_setting['simcode']
dir_out_base = args.opt

# Loop for runs
run_list = pd.read_csv(args.summary, sep=' ', names=config.dat_header)
run_list['LinacRunNo'] = run_list['LinacRunNo'].astype(pd.Int64Dtype())
run_list['NormalRunNo'] = run_list['NormalRunNo'].astype(pd.Int64Dtype())
for index, row in run_list.iterrows():
    run_this = row['LinacRunNo']
    mode_this = row['RunMode']
    normal_this = row['NormalRunNo']
    if mode_this != 0:
       continue
    # Make directories
    command = 'make phase={} run={:06d} normalrun={:06d} MC -f mc.mk'.format(args.phase, run_this, normal_this)
    print(command)
    os.system(command)

