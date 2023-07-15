#!/usr/bin/env python

import argparse, config, pandas as pd
import os, time, subprocess, glob
psr = argparse.ArgumentParser()
psr.add_argument('--phase', default='sk7', help='phase')
psr.add_argument('-i', dest='ipt', help='input directory')
psr.add_argument('-o', dest='opt', help='output directory')
psr.add_argument('--summary', help='summary file')
psr.add_argument('--tentative', default=False, action='store_true')
args = psr.parse_args()
# Set parameters
# dir_in = '/disk03/usr8/iekikei/corepmt/make_skg4/output'
dir_out_base = args.opt
run_list = pd.read_csv(args.summary, sep=' ', names=config.dat_header)
run_list['LinacRunNo'] = run_list['LinacRunNo'].astype(pd.Int64Dtype())
run_list['NormalRunNo'] = run_list['NormalRunNo'].astype(pd.Int64Dtype())
# Loop for runs
for index, row in run_list.iterrows():
    run_this = row['LinacRunNo']
    mode_this = row['RunMode']
    normal_this = row['NormalRunNo']

    # Select mode=0
    if mode_this != 0:
        continue

    # Make directories
    if not args.tentative:
        command = 'make phase={} phasedir={} run={:06d} normalrun={:06d} RECONMC -f recon.mk'.format(args.phase, args.ipt + '/{:06d}'.format(run_this), run_this, normal_this)
    else:
        command = 'make phase={} phasedir={} run={:06d} normalrun={:06d} tentative=t RECONMC -f recon.mk'.format(args.phase, args.ipt + '/{:06d}'.format(run_this), run_this, normal_this)
    print(command)
    os.system(command)

