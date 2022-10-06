'''
合并多个Dark Mode run的结果
+ DCR
'''
import argparse
import h5py, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config
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
runs = configcsv[configcsv['MODE']==0].to_records()
badruns = np.loadtxt(args.badrun)
selectruns = []
for run in runs:
    if not run['RUNNO'] in badruns:
        selectruns.append(run)
if len(selectruns)==0:
    print('Error: No runs contains noise stage run')
    exit(0)
results = pd.concat([loadh5(args.dir.format(run['RUNNO']), run['CHANNEL'], run['RUNNO']) for run in selectruns])
mergeresults = results[['peakC', 'vallyC', 'Gain', 'GainSigma', 'PV', 'Rise', 'Fall', 'TH', 'FWHM', 'RiseSigma', 'FallSigma', 'THSigma', 'FWHMSigma']].mean()
# 1st row store mean, 2nd row store std
mergeresultsA = np.zeros((2,), dtype=[
    ('peaC', '<f4'), ('vallyC', '<f4'), ('Gain', '<f4'), ('GainSigma', '<f4'), ('PV', '<f4'),
    ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'), ('RiseSigma', '<f4'), ('FallSigma', '<f4'), ('THSigma', '<f4'), ('FWHMSigma', '<f4'),
    ('DCR', '<f4'), ('Res', '<f4'), ('chargeMu', '<f4'), ('chargeSigma', '<f4'), ('chargeRes', '<f4')
])
rise_weights = results['RiseSigma']**2/results['TriggerNum']
fall_weights = results['FallSigma']**2/results['TriggerNum']
th_weights = results['THSigma']**2/results['TriggerNum']
fwhm_weights = results['FWHMSigma']**2/results['TriggerNum']
# peakC...,RiseSigma直接使用均值
mergeresultsA[0] = (
    np.mean(results['peakC']), np.mean(results['vallyC']), np.mean(results['Gain']), np.mean(results['GainSigma']), np.mean(results['PV']),
    np.sum(results['Rise']/rise_weights)/np.sum(1/rise_weights), np.sum(results['Fall']/fall_weights)/np.sum(1/fall_weights), np.sum(results['TH']/th_weights)/np.sum(1/th_weights), np.sum(results['FWHM']/fwhm_weights)/np.sum(1/fwhm_weights),
    np.mean(results['RiseSigma']), np.mean(results['FallSigma']), np.mean(results['THSigma']), np.mean(results['FWHMSigma']),
    np.sum(results['DCR'] * results['TotalNum']) / np.sum(results['TotalNum']), np.mean(results['GainSigma'] / results['Gain']),
    np.mean(results['chargeMu']), np.mean(results['chargeSigma']), np.mean(results['chargeSigma']/results['chargeMu'])
    )
# 暂时不用误差这一行
# mergeresultsA[1] = (*results[['peakC', 'vallyC', 'Gain', 'GainSigma', 'PV', 'Rise', 'Fall', 'TH', 'FWHM', 'RiseSigma', 'FallSigma', 'THSigma', 'FWHMSigma']].std().values, np.std(results['DCR']), np.std(results['GainSigma']/results['Gain']))
# 统计结果并合并存储
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('concat', data=results.to_records(), compression='gzip')
    opt.create_dataset('merge', data=mergeresultsA, compression='gzip')
# 绘制变化曲线
with PdfPages(args.opt + '.pdf') as pdf:
    fig, ax = plt.subplots()
    ax.plot(results['Run'], results['DCR'], marker='o')
    ax.axhline(mergeresultsA[0]['DCR'])
    ax.set_xlabel('Run')
    ax.set_ylabel('DCR')
    pdf.savefig(fig)
