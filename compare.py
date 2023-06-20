import argparse, os, pandas as pd, numpy as np
import config
def execute(command, verbose):
    if verbose:
        print(command)
    else:
        response = os.popen(command)
        print(response.read())
        response.close()
psr = argparse.ArgumentParser()
psr.add_argument('--phase1', default='sk6', help='phases')
psr.add_argument('--phase2', default='sk7', help='phases')
psr.add_argument('-i', dest='ipt', help='summary file')
psr.add_argument('--onebyone', default=False, action='store_true')
psr.add_argument('-n', default=False, action='store_true', help='verbose')
args = psr.parse_args()
# response = os.popen('make phase1={} phase2={} -f initRuns.mk {}_{}.csv'.format(args.phase1, args.phase2, args.phase1, args.phase2))
runs_df = pd.read_csv(args.ipt)
runs_df['RunNo'+args.phase1] = runs_df['RunNo'+args.phase1].astype(pd.Int64Dtype())
runs_df['X'+args.phase1] = runs_df['X'+args.phase1].astype(pd.Int64Dtype())
if args.onebyone:
    for index, row in runs_df.iterrows():
        state = 'Res_' + '_'.join([str(row['X']), str(row['Z']), str(row['E'])])
        command = 'make run1={} run2={} phase1={} phase2={} state={} -f compare.mk'.format(row['RunNo' + args.phase1], row['RunNo' + args.phase2], config.setting[args.phase1]['path'], config.setting[args.phase2]['path'], state)
        execute(command, args.n)
else:
    for index, row in runs_df.groupby(['Z', 'E']):
        state = 'Res_' + '_'.join([str(index[0]), str(index[1])])
        run1 = row[['RunNo' + args.phase1, 'X'+args.phase1]]
        if np.isnan(run1['RunNo' + args.phase1]).any():
            run1_no = ""
            run1_x = ""
        else:
            run1_no = ' '.join([str(i) for i in np.unique(run1['RunNo' + args.phase1].to_numpy())])
            run1_x = ' '.join([str(i) for i in np.unique(run1['X' +args.phase1].to_numpy())])
        run2_no = ' '.join([str(i) for i in row['RunNo' + args.phase2].to_numpy()])
        run2_x = ' '.join([str(i) for i in row['X' + args.phase2].to_numpy()])
        # print(run1_no,run1_x, run2_no, run2_x)
        command = 'make run1="{}" run2="{}" run1x="{}" run2x="{}" phase1={} phase2={} Z={} E={} -f compare.mk {}'.format(run1_no, run2_no, run1_x, run2_x, config.setting[args.phase1]['path'], config.setting[args.phase2]['path'], str(index[0]), str(index[1]), state+'.pdf')
        execute(command, args.n)
