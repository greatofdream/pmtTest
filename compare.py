import argparse, os, pandas as pd, numpy as np
import config
psr = argparse.ArgumentParser()
psr.add_argument('--phase1', default='sk6', help='phases')
psr.add_argument('--phase2', default='sk7', help='phases')
psr.add_argument('-n', default=False, action='store_true', help='verbose')
args = psr.parse_args()
# response = os.popen('make phase1={} phase2={} -f initRuns.mk {}_{}.csv'.format(args.phase1, args.phase2, args.phase1, args.phase2))
runs_df = pd.read_csv('{}_{}.csv'.format(args.phase1, args.phase2))
for index, row in runs_df.iterrows():
    state = 'Res_' + '_'.join([str(row['X']), str(row['Z']), str(row['E'])])
    command = 'make run1={} run2={} phase1={} phase2={} state={} -f compare.mk'.format(row['RunNo' + args.phase1], row['RunNo' + args.phase2], config.setting[args.phase1]['path'], config.setting[args.phase2]['path'], state)
    if args.n:
        print(command)
    else:
        response = os.popen(command)
        print(response.read())
        response.close()

