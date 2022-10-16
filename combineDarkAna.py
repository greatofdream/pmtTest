'''
合并多个Dark Mode run的结果
+ DCR
'''
import argparse
import h5py, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config
ADC2mV = config.ADC2mV
plt.style.use('./journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
def loadh5(f, channel, run):
    with h5py.File(f, 'r') as ipt:
        res = ipt['res'][:]
        resSigma2 = ipt['resSigma2'][:]
    result = pd.DataFrame(res[res['Channel']==channel])
    resultSigma2 = pd.DataFrame(resSigma2[resSigma2['Channel']==channel])
    result['Run'] = run
    resultSigma2['Run'] = run
    return result, resultSigma2
psr = argparse.ArgumentParser()
psr.add_argument('--dir', default='ExResult/{}/0ns/charge.h5', help='the ana result directory')
psr.add_argument('--config', help='test run no csv')
psr.add_argument('--badrun', help='runs excluded')
psr.add_argument('-o', dest='opt', help='name of output file')
args = psr.parse_args()

# 统计分析结果路径和对应ch号
configcsv = pd.read_csv(args.config)
runs = configcsv[configcsv['MODE']==0].to_records()
badruns = np.unique(np.append(np.loadtxt(args.badrun), np.loadtxt('ExPMT/ExcludeRun.csv')))
selectruns = []
for run in runs:
    if not run['RUNNO'] in badruns:
        selectruns.append(run)
if len(selectruns)==0:
    print('Error: No runs contains noise stage run')
    exit(0)
infos = [loadh5(args.dir.format(run['RUNNO']), run['CHANNEL'], run['RUNNO']) for run in selectruns]
results = pd.concat([i[0] for i in infos])
resultsSigma2 = pd.concat([i[1] for i in infos])
# 1st row store mean, 2nd row store std
mergeresultsA = np.zeros((2,), dtype=[
    ('peakC', '<f4'), ('vallyC', '<f4'), ('Gain', '<f4'), ('GainSigma', '<f4'), ('PV', '<f4'),
    ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'),
    ('DCR', '<f4'), ('Res', '<f4'),
    ('chargeMu', '<f4'), ('chargeSigma2', '<f4'), ('chargeRes', '<f4')
])
ress = results['GainSigma'] / results['Gain']
res_weights = ress**2 * (resultsSigma2['GainSigma']/results['GainSigma']**2 + resultsSigma2['Gain']/results['Gain']**2)
chargeress = np.sqrt(results['chargeSigma2']) / results['chargeMu']
chargeress_weights = chargeress**2 * (resultsSigma2['chargeSigma2']/results['chargeSigma2']**2/4 + resultsSigma2['chargeMu']/results['chargeMu']**2)

# peakC...,RiseSigma使用最小二乘加权
mergeresultsA[0] = (
    np.sum(results['peakC']/resultsSigma2['peakC']) / np.sum(1/resultsSigma2['peakC']),
    np.sum(results['vallyC']/resultsSigma2['vallyC']) / np.sum(1/resultsSigma2['vallyC']),
    np.sum(results['Gain']/resultsSigma2['Gain']) / np.sum(1/resultsSigma2['Gain']),
    np.sum(results['GainSigma']/resultsSigma2['GainSigma']) / np.sum(1/resultsSigma2['GainSigma']),
    np.sum(results['PV']/resultsSigma2['PV']) / np.sum(1/resultsSigma2['PV']),
    np.sum(results['Rise']/resultsSigma2['Rise']) / np.sum(1/resultsSigma2['Rise']),
    np.sum(results['Fall']/resultsSigma2['Fall'])/np.sum(1/resultsSigma2['Fall']),
    np.sum(results['TH']/resultsSigma2['TH'])/np.sum(1/resultsSigma2['TH']),
    np.sum(results['FWHM']/resultsSigma2['FWHM'])/np.sum(1/resultsSigma2['FWHM']),
    np.sum(results['DCR'] / resultsSigma2['DCR']) / np.sum(1/resultsSigma2['DCR']),
    np.sum(ress/res_weights) / np.sum(1/res_weights),
    np.sum(results['chargeMu'] / resultsSigma2['chargeMu']) / np.sum(1/resultsSigma2['chargeMu']),
    np.sum(results['chargeSigma2']/resultsSigma2['chargeSigma2']) / np.sum(1/resultsSigma2['chargeSigma2']),
    np.sum(chargeress / chargeress_weights) / np.sum(1/chargeress_weights)
    )
# 误差使用最小二乘法
mergeresultsA[['peakC', 'vallyC', 'Gain', 'GainSigma', 'PV', 'Rise', 'Fall', 'TH', 'FWHM', 'chargeMu', 'chargeSigma2', 'Res', 'chargeRes', 'DCR']][1] = (
    1/np.sum(1/resultsSigma2['peakC']),
    1/np.sum(1/resultsSigma2['vallyC']),
    1/np.sum(1/resultsSigma2['Gain']),
    1/np.sum(1/resultsSigma2['GainSigma']),
    1/np.sum(1/resultsSigma2['PV']),
    1/np.sum(1/resultsSigma2['Rise']),
    1/np.sum(1/resultsSigma2['Fall']),
    1/np.sum(1/resultsSigma2['TH']),
    1/np.sum(1/resultsSigma2['FWHM']),
    1 / np.sum(1/resultsSigma2['chargeMu']),
    1 / np.sum(1/resultsSigma2['chargeSigma2']),
    1/np.sum(1/res_weights),
    1/np.sum(1/chargeress_weights),
    1/np.sum(1/resultsSigma2['DCR'])
)
# 统计结果并合并存储
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('concat', data=results.to_records(), compression='gzip')
    opt.create_dataset('merge', data=mergeresultsA, compression='gzip')
# 绘制变化曲线
with PdfPages(args.opt + '.pdf') as pdf:
    fig, ax = plt.subplots()
    ax.errorbar(results['Run'], results['DCR'], yerr=np.sqrt(resultsSigma2['DCR']), marker='o')
    ax.axhline(mergeresultsA[0]['DCR'])
    ax.fill_betweenx([mergeresultsA[0]['DCR']-np.sqrt(mergeresultsA[1]['DCR']), mergeresultsA[0]['DCR']+np.sqrt(mergeresultsA[1]['DCR'])], results['Run'].values[0], results['Run'].values[-1], alpha=0.5)
    ax.set_xlabel('Run')
    ax.set_ylabel('DCR')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.errorbar(results['Run'], results['Gain'], yerr=np.sqrt(resultsSigma2['Gain']), marker='o')
    ax.axhline(mergeresultsA[0]['Gain'])
    ax.fill_betweenx([mergeresultsA[0]['Gain']-np.sqrt(mergeresultsA[1]['Gain']), mergeresultsA[0]['Gain']+np.sqrt(mergeresultsA[1]['Gain'])], results['Run'].values[0], results['Run'].values[-1], alpha=0.5)
    ax.set_xlabel('Run')
    ax.set_ylabel('$G_1$')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.errorbar(results['Run'], results['chargeMu']/50/1.6*ADC2mV, yerr=np.sqrt(resultsSigma2['chargeMu'])/50/1.6*ADC2mV, marker='o')
    ax.axhline(mergeresultsA[0]['chargeMu']/50/1.6*ADC2mV)
    ax.fill_betweenx([(mergeresultsA[0]['chargeMu']-np.sqrt(mergeresultsA[1]['chargeMu']))/50/1.6*ADC2mV, (mergeresultsA[0]['chargeMu']+np.sqrt(mergeresultsA[1]['chargeMu']))/50/1.6*ADC2mV], results['Run'].values[0], results['Run'].values[-1], alpha=0.5)
    ax.set_xlabel('Run')
    ax.set_ylabel('Gain')
    pdf.savefig(fig)