import argparse, os, pandas as pd
import config
def getMeta(run):
    response = os.popen('summary {}'.format(run))
    summary = {}
    for res in response:
        kv = [i.strip() for i in res.split('=', 1)]
        if len(kv)==2:
            summary[kv[0]] = kv[1]
    return summary
def getNormalRun(run):
    summary = getMeta(run-1)
    while summary['Run Mode']!='1:Normal':
        run -= 1
        summary = getMeta(run-1)
    return run-1
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='linac run number')
    psr.add_argument('-o', dest='opt', help='output dat file')
    psr.add_argument('--phase', dest='phase', help='SK stage')
    args = psr.parse_args()
    phase_df = pd.read_csv(args.ipt, names=config.summary_header) 
    phase_df['RunNo'] = phase_df['RunNo'].astype(pd.Int64Dtype())
    phase_df['X'] = phase_df['X'].astype(pd.Int64Dtype())
    rows = []
    former_run, former_normalrun = 0, 0
    former_X, former_Z, former_E = 0, 0, 0
    for index, p_row in phase_df.iterrows():
        if p_row['RunNo']-former_run==1:
            normalrun = former_normalrun
            if former_X!=p_row['X'] or former_E!=p_row['E'] or former_Z!=p_row['Z']:
                print('{} use former run {} info'.format(p_row['RunNo'], former_run))
                X, E, Z = former_X, former_E, former_Z
            else:
                X, E, Z = p_row['X'], p_row['E'], p_row['Z']
        else:
            normalrun = getNormalRun(p_row['RunNo'])
            X, E, Z = p_row['X'], p_row['E'], p_row['Z']
        rows.append('{} {} {} {} {} {} {}\n'.format(p_row['RunNo'], E, p_row['Mode'], config.vertexMap['X'][X], config.vertexMap['Y'][-7], config.vertexMap['Z'][Z], normalrun))
        former_run, former_normalrun = p_row['RunNo'], normalrun
        former_X, former_E, former_Z = X, E, Z
    with open(args.opt, 'w') as opt:
        for row in rows:
            opt.write(row)
