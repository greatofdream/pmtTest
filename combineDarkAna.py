'''
合并多个Dark Mode run的结果
+ DCR
'''
import argparse
import h5py, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('./journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
def loadh5(f, channel, run):
    with h5py.File(f, 'r') as ipt:
        res = ipt['res'][:]
    result = pd.DataFrame(res[res['Channel']==channel])
    result['Run'] = run
    return result
psr = argparse.ArgumentParser()
psr.add_argument('--dir', default='ExResult/{}/0ns/charge.h5', help='the ana result directory')
psr.add_argument('--config', help='test run no csv')
psr.add_argument('--badrun', help='runs excluded')
psr.add_argument('-o', dest='opt', help='name of output file')
args = psr.parse_args()

# 统计分析结果路径和对应ch号
configcsv = pd.read_csv(args.config)
runs = configcsv[configcsv['TRIGGER']==-1].to_records()
badruns = np.loadtxt(args.badrun)
selectruns = []
for run in runs:
    if not run['RUNNO'] in badruns:
        selectruns.append(run)
if len(selectruns)==0:
    print('Error: No runs contains noise stage run')
    exit(0)
results = pd.concat([loadh5(args.dir.format(run['RUNNO']), run['CHANNEL'], run['RUNNO']) for run in selectruns])
mergeresults = results[['peakC', 'vallyC', 'Gain', 'GainSigma']].mean()
mergeresults['DCR'] = np.sum(results['DCR']*results['TotalNum'])/np.sum(results['TotalNum'])
mergeresultsA = np.zeros((2,), dtype=[
    ('peaC', '<f4'), ('vallyC', '<f4'), ('Gain', '<f4'), ('GainSigma', '<f4'), ('DCR', '<f4')
])
mergeresultsA[0] = tuple(mergeresults.values)
mergeresultsA[1] = (*results[['peakC', 'vallyC', 'Gain', 'GainSigma']].std().values, np.std(results['DCR']))
# 统计结果并合并存储
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('concat', data=results.to_records(), compression='gzip')
    opt.create_dataset('merge', data=mergeresultsA, compression='gzip')
# 绘制变化曲线
with PdfPages(args.opt + '.pdf') as pdf:
    fig, ax = plt.subplots()
    ax.plot(results['Run'], results['DCR'], marker='o')
    ax.axhline(mergeresults['DCR'])
    ax.set_xlabel('Run')
    ax.set_ylabel('DCR')
    pdf.savefig(fig)
