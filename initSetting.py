import argparse, pandas as pd
summary_header = ['RunNo', 'X', 'Z', 'E', 'Mode', 'Time', 'Comment']
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', nargs='+', help='input summary files')
    psr.add_argument('-o', dest='opt', help='ouput file')
    psr.add_argument('--label', dest='label', nargs='+', help='phase tags')
    psr.add_argument('--on', dest='on', nargs='+', default=['X', 'Z', 'E'])
    psr.add_argument('--how', dest='how', default='inner')
    args = psr.parse_args()
    if len(args.ipt) != len(args.label):
        print('error! length of summary files and labels different')
        exit(0)
    phase1_df, phase2_df = pd.read_csv(args.ipt[0], names=summary_header), pd.read_csv(args.ipt[1], names=summary_header)
    phase1_df['RunNo'] = phase1_df['RunNo'].astype(pd.Int64Dtype())
    phase1_df['X'] = phase1_df['X'].astype(pd.Int64Dtype())
    phase1_df, phase2_df = phase1_df.loc[phase1_df['Mode']==0][['RunNo', 'X', 'Z', 'E']], phase2_df.loc[phase2_df['Mode']==0][['RunNo', 'X', 'Z', 'E']]
    print(phase1_df)
    print(phase2_df)
    combineDf = pd.merge(phase1_df, phase2_df, on=args.on, how=args.how, suffixes=args.label)
    # phase1_gp, phase2_gp = phase1_df.groupby(['X', 'Z', 'E']), phase2_df.groupby(['X', 'Z', 'E'])
    combineDf.to_csv(args.opt, index=False)
