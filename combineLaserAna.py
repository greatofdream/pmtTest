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
except:
    exit(0)
# 统计分析结果路径和对应ch号
configcsv = pd.read_csv(args.config)
runs = configcsv[configcsv['TRIGGER']>=0].to_records()
badruns = np.loadtxt(args.badrun)
selectruns = []
for run in runs:
    if not run['RUNNO'] in badruns:
        selectruns.append(run)
if len(selectruns)==0:
    print('Error: No runs contains laser stage run')
    exit(0)
results = pd.concat([loadh5(args.dir.format(run['RUNNO']) + '/chargeSelect.h5', run['CHANNEL'], run['RUNNO']) for run in selectruns])
mergeresults = results[['peakC', 'vallyC', 'Gain', 'GainSigma', 'TTS']].mean()

# 计算去除dark后的触发率
results['TriggerRateWODCR'] = (results['TriggerRate'] - darkresult['DCR'] * results['window'] * 1e-6)/(1 - darkresult['DCR'] * results['window'] * 1e-6) 
mergeresults['TriggerRate'] = np.sum(results['TriggerRate'] * results['TotalNum'])/np.sum(results['TotalNum'])
mergeresults['TriggerRateWODCR'] = np.sum(results['TriggerRateWODCR'] * results['TotalNum'])/np.sum(results['TotalNum'])
# 统计多批数据的afterpulse
pulseResults = pd.concat([loadPulse(args.dir.format(run['RUNNO']) + '/pulseRatio.h5', run['CHANNEL']) for run in selectruns])
pulseratioResults = pd.concat([loadRatio(args.dir.format(run['RUNNO']) + '/pulseRatio.h5', run['CHANNEL'], run['RUNNO']) for run in selectruns])
promptwindow, delay1window, delay10window = promptB - promptE, delay1E - delay1B, delay10E - delay10B
pulseratioResults['promptWODCR'] = (pulseratioResults['prompt'] - darkresult['DCR'] * promptwindow * 1e-6)/(1 - darkresult['DCR'] * promptwindow * 1e-6)
pulseratioResults['delay1WODCR'] = (pulseratioResults['delay1'] - darkresult['DCR'] * delay1window * 1e-6)/(1 - darkresult['DCR'] * delay1window * 1e-6)
pulseratioResults['delay10WODCR'] = (pulseratioResults['delay10'] - darkresult['DCR'] * delay10window * 1e-6)/(1 - darkresult['DCR'] * delay10window * 1e-6)
mergePulseResults = np.zeros((1,), dtype=[('prompt', '<f4'), ('delay1', '<f4'), ('delay10', '<f4'), ('promptWODCR', '<f4'), ('delay1WODCR', '<f4'), ('delay10WODCR', '<f4')])
totalTriggerNum = np.sum(pulseratioResults['TriggerNum'])
mergePulseResults = (
    np.sum(pulseratioResults['prompt'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['delay1'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['delay10'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['promptWODCR'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['delay1WODCR'] * pulseratioResults['TriggerNum'])/totalTriggerNum,
    np.sum(pulseratioResults['delay10WODCR'] * pulseratioResults['TriggerNum'])/totalTriggerNum)
# 统计结果并合并存储
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('concat', data=results, compression='gzip')
    opt.create_dataset('merge', data=mergeresults, compression='gzip')
    opt.create_dataset('mergepulse', data=mergePulseResults, compression='gzip')
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
    ax.plot(results['Run'], results['TriggerRateWODCR'], marker='o', label='Trigger Rate WO Dark Noise')
    ax.set_xlabel('Run')
    ax.set_ylabel('Trigger Rate')
    ax.legend()
    pdf.savefig(fig)
    plt.close()
    # Afterpulse 变化
    fig, ax = plt.subplots(figsize=(12,6))
    h = ax.hist2d(pulseResults['t'], pulseResults['Q'], bins=[int((delay10E - delay1B)/50), 50], range=[[delay1B, delay10E], [0, 1000]], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('Relative t/ns')
    ax.set_ylabel('Equivalent Charge/ADCns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(12,6))
    binwidth = 50
    h = ax.hist(pulseResults['t'], bins=int(delay10E/binwidth), range=[0, delay10E], histtype='step')
    h = ax.hist(pulseResults['t'], bins=int((delay10E - delay1B)/binwidth), range=[delay1B, delay10E], histtype='step')
    ax.axhline(totalTriggerNum * darkresult['DCR'] * binwidth * 1e-6, label='Expected Dark Noise')
    ax.set_xlabel('Relative t/ns')
    ax.set_ylabel('Entries')
    ax.set_xlim([0, delay10E])
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.legend()
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(pulseResults['t'], pulseResults['Q'], bins=[int((promptB - promptE)/10), 50], range=[[-promptB, -promptE], [0, 1000]], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('Relative t/ns')
    ax.set_ylabel('Equivalent Charge/ADCns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist(pulseResults['t'], bins=int((promptB - promptE)/10), range=[-promptB, -promptE], histtype='step')
    ax.axhline(totalTriggerNum * darkresult['DCR'] * 10 * 1e-6, label='Expected Dark Noise')
    ax.set_xlabel('Relative t/ns')
    ax.set_ylabel('Entries')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)
