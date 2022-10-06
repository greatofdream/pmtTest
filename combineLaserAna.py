'''
合并多个Laser Run 结果，并扣除Dark Mode run合并的结果
'''
import argparse
import h5py, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('./journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap
from scipy.optimize import minimize
import config
promptB, promptE = config.promptB, config.promptE
delay1B, delay1E = config.delay1B, config.delay1E
delay10B, delay10E = config.delay10B, config.delay10E
spelength, spestart, speend = config.spelength, config.spestart, config.speend

def loadh5(f, channel, run):
    with h5py.File(f, 'r') as ipt:
        res = ipt['res'][:]
    result = pd.DataFrame(res[res['Channel']==channel])
    result['Run'] = run
    return result
def loadPulse(f, channel):
    with h5py.File(f, 'r') as ipt:
        res = ipt['ch{}'.format(channel)][:]
    return pd.DataFrame(res)
def loadRatio(f, channel, run):
    with h5py.File(f, 'r') as ipt:
        res = ipt['ratio'][:]
    result = pd.DataFrame(res[res['Channel']==channel])
    result['Run'] = run
    return result
def loadSER(f, channel, run):
    with h5py.File(f, 'r') as ipt:
        res = ipt['res'][:]
    result = pd.DataFrame(res[res['Channel']==channel])
    result['Run'] = run
    return result
def Afterpulse(x, xs):
    x = x.reshape((-1,3))
    return np.sum(x[:,0]*np.exp(-(xs[:,None]-x[:,1])**2/2/x[:,2]**2)/x[:,2], axis=-1)
def likelihood(x, *args):
    xs, ys = args
    eys = Afterpulse(x, xs)
    return np.sum((ys-eys)**2)
psr = argparse.ArgumentParser()
psr.add_argument('--dir', help='the ana result directory')
psr.add_argument('--config', help='test run no csv')
psr.add_argument('--badrun', help='runs excluded')
psr.add_argument('--dark', help='dark run merge result')
psr.add_argument('-o', dest='opt', help='name of output file')
args = psr.parse_args()
# 读取darkrun 合并结果
try:
    with h5py.File(args.dark, 'r') as ipt:
        darkresult = ipt['merge'][:][0]
    darkExpect = True
except:
    darkExpect = False
try:
    with open(args.dark.replace('dark.h5', 'bounds.csv'), 'r') as ipt:
        boundsSigma = np.loadtxt(ipt, delimiter=',')
except:
    boundsSigma = []
# 统计分析结果路径和对应ch号
configcsv = pd.read_csv(args.config)
PMTName = configcsv['PMT'].values[0]
runs = configcsv[configcsv['MODE']==1].to_records()
badruns = np.loadtxt(args.badrun)
selectruns = []
for run in runs:
    if not run['RUNNO'] in badruns:
        selectruns.append(run)
if len(selectruns)==0:
    print('Error: No runs contains laser stage run')
    exit(0)
results = pd.concat([loadh5(args.dir.format(run['RUNNO']) + '/chargeSelect.h5', run['CHANNEL'], run['RUNNO']) for run in selectruns])
print(results[['Run', 'Channel', 'TTS','TTS_bin']])
if darkExpect:
    results['TriggerRateWODCR'] = (results['TriggerRate'] - darkresult['DCR'] * results['window'] * 1e-6)/(1 - darkresult['DCR'] * results['window'] * 1e-6) 
else:
    # omit the influence of DCR for trigger pulse
    results['TriggerRateWODCR'] = results['TriggerRate']
mergeresults = results[['peakC', 'vallyC', 'Gain', 'GainSigma', 'PV', 'Rise', 'Fall', 'TH', 'FWHM', 'RiseSigma', 'FallSigma', 'THSigma', 'FWHMSigma', 'TTS']]
mergeresultsA = np.zeros((2,), dtype=[
    ('peaC', '<f4'), ('vallyC', '<f4'), ('Gain', '<f4'), ('GainSigma', '<f4'), ('PV', '<f4'),
    ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'), ('RiseSigma', '<f4'), ('FallSigma', '<f4'), ('THSigma', '<f4'), ('FWHMSigma', '<f4'),
    ('TTS', '<f4'), ('TTS_bin', '<f4'), ('Res', '<f4'), ('TriggerRate', '<f4'), ('TriggerRateWODCR', '<f4'), ('chargeMu', '<f4'), ('chargeSigma', '<f4'), ('chargeRes', '<f4'), ('PDE', '<f4')
])
# PDE测量从TestSummary.csv中获取
pdes = pd.read_csv(config.TestSummaryPath).set_index('PMT').loc[PMTName, 'PDE'].values
if np.sum(~(np.isnan(pdes)|(pdes==0)))>0:
    pde = np.mean(pdes[~(np.isnan(pdes)|(pdes==0))])
else:
    pde = np.nan
# 计算去除dark后的触发率,采用合并测量的方式
## 这里peakC,vallyC等拟合参数合并测量时没有考虑每次测量的方差
triggerNum = results['TotalNum'] * results['TriggerRate']
rise_weights = results['RiseSigma']**2/triggerNum
fall_weights = results['FallSigma']**2/triggerNum
th_weights = results['THSigma']**2/triggerNum
fwhm_weights = results['FWHMSigma']**2/triggerNum
mergeresultsA[0] = (
    np.mean(results['peakC']), np.mean(results['vallyC']), np.mean(results['Gain']), np.mean(results['GainSigma']), np.mean(results['PV']),
    np.sum(results['Rise']/rise_weights)/np.sum(1/rise_weights), np.sum(results['Fall']/fall_weights)/np.sum(1/fall_weights), np.sum(results['TH']/th_weights)/np.sum(1/th_weights), np.sum(results['FWHM']/fwhm_weights)/np.sum(1/fwhm_weights),
    np.mean(results['RiseSigma']), np.mean(results['FallSigma']), np.mean(results['THSigma']), np.mean(results['FWHMSigma']),
    np.mean(results['TTS']), np.mean(results['TTS_bin']),
    np.mean(results['GainSigma']/results['Gain']),
    np.sum(results['TriggerRate'] * results['TotalNum'])/np.sum(results['TotalNum']),
    np.sum(results['TriggerRateWODCR'] * results['TotalNum'])/np.sum(results['TotalNum']),
    np.mean(results['chargeMu']), np.mean(results['chargeSigma']), np.mean(results['chargeSigma']/results['chargeMu']),
    pde
    )
# 暂时不用误差这一行
# mergeresultsA[1] = (
#     *mergeresults.std().values,
#     np.std(results['GainSigma']/results['Gain']),
#     np.std(results['TriggerRate']),
#     np.std(results['TriggerRateWODCR'])
#     )

# 统计多批数据的afterpulse
pulseResults = pd.concat([loadPulse(args.dir.format(run['RUNNO']) + '/pulseRatio.h5', run['CHANNEL']) for run in selectruns])
pulseratioResults = pd.concat([loadRatio(args.dir.format(run['RUNNO']) + '/pulseRatio.h5', run['CHANNEL'], run['RUNNO']) for run in selectruns])
promptwindow, delay1window, delay10window = promptB - promptE, delay1E - delay1B, delay10E - delay10B
if darkExpect:
    pulseratioResults['promptWODCR'] = (pulseratioResults['prompt'] - darkresult['DCR'] * promptwindow * 1e-6)/(1 - darkresult['DCR'] * promptwindow * 1e-6)
    pulseratioResults['delay1WODCR'] = (pulseratioResults['delay1'] - darkresult['DCR'] * delay1window * 1e-6)/(1 - darkresult['DCR'] * delay1window * 1e-6)
    pulseratioResults['delay10WODCR'] = (pulseratioResults['delay10'] - darkresult['DCR'] * delay10window * 1e-6)/(1 - darkresult['DCR'] * delay10window * 1e-6)
if not darkExpect or (np.sum(pulseratioResults['promptWODCR']* pulseratioResults['TriggerNum'])<0):
    # use pre pulse ratio to estimate the DCR
    pulseratioResults['promptWODCR'] = 0
    pulseratioResults['delay1WODCR'] = (pulseratioResults['delay1'] - pulseratioResults['prompt'] * delay1window / promptwindow)
    pulseratioResults['delay10WODCR'] = (pulseratioResults['delay10'] -pulseratioResults['prompt'] * delay10window / promptwindow)
mergePulseResults = np.zeros((2,), dtype=[('prompt', '<f4'), ('delay1', '<f4'), ('delay10', '<f4'), ('promptWODCR', '<f4'), ('delay1WODCR', '<f4'), ('delay10WODCR', '<f4')])
totalTriggerNum = np.sum(pulseratioResults['TriggerNum'])
mergePulseResults[0] = (
    np.sum(pulseratioResults['prompt'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['delay1'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['delay10'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['promptWODCR'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['delay1WODCR'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['delay10WODCR'] * pulseratioResults['TriggerNum'])/totalTriggerNum)
# 统计ser参数
## tau, sigma考虑的误差
serResults = pd.concat([loadSER(args.dir.format(run['RUNNO']) + '/serMerge.h5', run['CHANNEL'], run['RUNNO']) for run in selectruns])
mergeSERResults = np.zeros((2,), dtype=[('tau', '<f4'), ('sigma', '<f4'), ('tau_total', '<f4'), ('sigma_total', '<f4')])
mergeSERResults[0] = (
    np.mean(serResults['tau']),
    np.mean(serResults['sigma']),
    np.mean(serResults['tau_total']),
    np.mean(serResults['sigma_total']),
)
mergeSERResults[1] = (
    1/np.sum(1/serResults['tau_sigma']**2),
    1/np.sum(1/serResults['sigma_sigma']**2),
    np.std(serResults['tau_total']),
    np.std(serResults['sigma_total'])
)
# 绘制变化曲线
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)

with PdfPages(args.opt + '.pdf') as pdf:
    # Trigger Rate变化
    fig, ax = plt.subplots()
    ax.plot(results['Run'], results['TriggerRate'], marker='o', label='Trigger Rate')
    if darkExpect:
        ax.plot(results['Run'], results['TriggerRateWODCR'], marker='o', label='Trigger Rate WO Dark Noise')
    ax.set_xlabel('Run')
    ax.set_ylabel('Trigger Rate')
    ax.legend()
    pdf.savefig(fig)
    plt.close()
    # tts对比
    fig, ax = plt.subplots()
    ax.scatter(results['TTS'], results['TTS_bin'], marker='o', label='Trigger Rate')
    ax.set_xlabel('TTS')
    ax.set_ylabel('TTS bin fit')
    ax.legend()
    pdf.savefig(fig)
    plt.close()
    # Afterpulse 变化
    binwidth = 20
    fig, ax = plt.subplots(figsize=(15,6))
    h = ax.hist2d(pulseResults['t'], pulseResults['Q'], bins=[int((delay10E + promptB)/binwidth), 50], range=[[-promptB, delay10E], [0, 1000]], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('Relative t/ns')
    ax.set_ylabel('Equivalent Charge/ADCns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(15,6))
    h1 = ax.hist(pulseResults['t'], bins=int(delay10E/binwidth), range=[0, delay10E], histtype='step', label='After-pulse')
    h2 = ax.hist(pulseResults['t'], bins=int((promptB - promptE)/binwidth), range=[-promptB, -promptE], histtype='step', label='Pre-pulse')
    ax.set_xlabel('Relative t/ns')
    ax.set_ylabel('Entries')
    ax.set_xlim([-promptB, delay10E])
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.legend()
    pdf.savefig(fig)
    # 计算后脉冲比例
    if PMTName.startswith('PM'):
        searchwindows = config.searchwindowsMCP
    else:
        searchwindows = config.searchwindowsHama
    expectPrompt = np.sum((pulseResults['t']>-promptB)&(pulseResults['t']<-promptE))/(promptB-promptE) * binwidth
    MCPPeakNum = np.zeros(len(searchwindows), dtype=[('Group', '<i2'), ('t', '<f4'), ('N', '<i4'), ('pv', '<i4'), ('left', '<f4'), ('right', '<f4'), ('sigma', '<f4')])
    counts, edges = h1[0] - expectPrompt, (h1[1][:-1] + h1[1][1:])/2
    
    fig, ax = plt.subplots(figsize=(15,6))
    h_a = ax.hist(pulseResults['t'], bins=int(delay10E/binwidth), range=[0, delay10E], histtype='step', label='After-pulse')
    h_p = ax.hist(pulseResults['t'], bins=int((promptB - promptE)/binwidth), range=[-promptB, -promptE], histtype='step', label='Pre-pulse')
    # 改变搜索算法为拟合算法
    for i, w in enumerate(searchwindows):
        MCPPeakNum['Group'][i] = i
        area = (edges<w[1]) & (edges>w[0])
        selectCounts = counts[area]
        pi = np.argmax(selectCounts)
        pv = selectCounts[pi]
        MCPPeakNum['t'][i] = edges[area][pi]
        MCPPeakNum['pv'][i] = pv
        selectArea = selectCounts>(0.5*pv)
        MCPPeakNum['N'][i] = np.sum(selectCounts[selectArea])
        MCPPeakNum['left'][i] = edges[area][selectArea][0]
        MCPPeakNum['right'][i] = edges[area][selectArea][-1]
        # ax.fill_between(edges[area][selectArea], selectCounts[selectArea] + expectPrompt, np.ones(np.sum(selectArea)) * expectPrompt)
        print(edges[area][selectArea], selectCounts[selectArea], np.sum(selectArea))
    x0 = np.vstack([MCPPeakNum['pv'], MCPPeakNum['t'], (MCPPeakNum['right'] - MCPPeakNum['left'])/2+10]).T.reshape(-1)
    bounds = []
    for idx, sw in enumerate(searchwindows):
        bounds.append((0, 100*MCPPeakNum['pv'][i]))
        bounds.append(sw)
        if len(boundsSigma)>0:
            bounds.append(boundsSigma[idx])
        else:
            bounds.append((5,100))
    startEdges = int((250)/binwidth)
    endEdges = int(searchwindows[-1][-1]/binwidth)
    aftergroups = minimize(
        likelihood, x0,
        args=(edges[startEdges:endEdges], counts[startEdges:endEdges]),
        bounds=bounds,
        options={'eps':0.0001}
        )
    aftergroupsX = aftergroups.x.reshape((-1,3))
    MCPPeakNum['t'] = aftergroupsX[:, 1]
    MCPPeakNum['pv'] = aftergroupsX[:, 0]
    MCPPeakNum['sigma'] = aftergroupsX[:, 2]

    eys = Afterpulse(aftergroups.x, edges[startEdges:endEdges])
    ax.plot(edges[startEdges:endEdges], eys+expectPrompt, linewidth=1, alpha=0.9, label='fit')
    ax.axhline(expectPrompt, linewidth=1, linestyle='--', label='Average prepulse')
    if darkExpect:
        ax.axhline(totalTriggerNum * darkresult['DCR'] * binwidth * 1e-6, label='Expected Dark Noise')
    # for pn in MCPPeakNum:
    #     ax.annotate(pn['N'], (pn['t'], pn['pv'] + expectPrompt), (pn['t'], pn['pv'] + expectPrompt + 5))
    ax.set_xlabel('Relative t/ns')
    ax.set_ylabel('Entries')
    ax.set_xlim([-promptB, delay10E])
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.legend()
    pdf.savefig(fig)
# 统计结果并合并存储
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('concat', data=results.to_records(), compression='gzip')
    opt.create_dataset('merge', data=mergeresultsA, compression='gzip')
    opt.create_dataset('mergepulse', data=mergePulseResults, compression='gzip')
    opt.create_dataset('mergeSER', data=mergeSERResults, compression='gzip')
    opt.create_dataset('AfterPulse', data=MCPPeakNum, compression='gzip')