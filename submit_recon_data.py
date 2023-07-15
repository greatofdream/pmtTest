#!/usr/bin/env python

import argparse, config, pandas as pd
import os, time, subprocess, glob
psr = argparse.ArgumentParser()
psr.add_argument('--phase', default='sk7', help='phase')
psr.add_argument('-o', dest='opt', help='output directory')
psr.add_argument('--summary', help='summary file')
psr.add_argument('--tentative', default=False, action='store_true')
args = psr.parse_args()
# Set parameters
# dir_in = '/disk02/data7/sk5/lin'
phase_setting = config.setting[args.phase]
dir_format = phase_setting['dataPathFormat']
# dir_out_base = 'output/data_g4'
dir_out_base = args.opt
# run_first = 81588  # First LINAC run of SK5
# run_last = 81844 # Last LINAC run of SK5
# run_first = 86119 # First LINAC run of SK6
# run_last = 86161 # Last LINAC run of SK6

# Loop for runs
# run_list = '/home/sklowe/linac/const/linac_runsum.dat'
run_list = pd.read_csv(args.summary, sep=' ', names=config.dat_header)
run_list['LinacRunNo'] = run_list['LinacRunNo'].astype(pd.Int64Dtype())
run_list['NormalRunNo'] = run_list['NormalRunNo'].astype(pd.Int64Dtype())
for index, row in run_list.iterrows():
    run_this = row['LinacRunNo']
    mode_this = row['RunMode']
    # normal_this = int(vars[6])

    # Select mode=0
    if mode_this != 0:
       continue

    # Make directories
    if not args.tentative:
        command = 'make phase={} phasedir={} run={:06d} RECON -f recon.mk'.format(args.phase, dir_format.format(int(run_this/100), run_this), run_this)
    else:
        command = 'make phase={} phasedir={} run={:06d} tentative=t RECON -f recon.mk'.format(args.phase, dir_format.format(int(run_this/100), run_this), run_this)
    print(command)
    os.system(command)

