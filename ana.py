import argparse, os, pandas as pd, numpy as np
import config
psr = argparse.ArgumentParser()
psr.add_argument('--phase1', default='sk6', help='phases')
psr.add_argument('--phase2', default='sk7', help='phases')
psr.add_argument('-n', default=False, action='store_true', help='verbose')
args = psr.parse_args()
# response = os.popen('make phase1={} phase2={} -f initRuns.mk {}_{}.csv'.format(args.phase1, args.phase2, args.phase1, args.phase2))
runs_df = pd.read_csv('{}_{}.csv'.format(args.phase1, args.phase2))
for phase in [args.phase1, args.phase2]:
    for state, runs in runs_df.groupby(['X', 'Z', 'E']):
        command = 'make runs="{}" state={} directory={} -f ana.mk'.format(' '.join([str(i) for i in np.unique(runs['RunNo' + phase])]), 'Res_' + '_'.join([str(state[0]), str(state[1]), str(state[2])]), config.setting[phase]['path'])
        if args.n:
            print(command)
        else:
            response = os.popen(command)
            print(response.read())
            response.close()
# os.popen('make runTuples={}  phase1={} phase2={} -f initRuns.mk compare'.format(' '.join([i+'_'+j for i,j in zip(runs_df.iloc[:,4], run_df.iloc[:,5])]), args.phase1, args.phase2))

