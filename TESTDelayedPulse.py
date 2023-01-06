'''
Pre-pulse waveform check.
# python3 TESTDelayedPulse.py --ana ExResult/800/600ns/qe/preanalysisMerge.h5 --pulseana ExResult/800/600ns/laserElastic/analysisMerge.h5 --interval  ExResult/800/600ns/trigger.h5 --datadir /tarski/JNE/JinpingData/Jinping_1ton_Data/pmtTest --runno 800 --channel 2 3 4 5 --entries ExResult/800/entries.txt -o ExResult/800/600ns/TestDelayedPulse.pdf
'''
import argparse
import h5py, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('./journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import config
import subprocess
import uproot
import pandas as pd
from pandas import Series
from entries import EntriesReader
psr = argparse.ArgumentParser()
psr.add_argument('--ana', help='the ana result')
psr.add_argument('--pulseana', help='the ana result')
psr.add_argument('--datadir', help='directory of data')
psr.add_argument('--interval', help='the time interval for each channel')
psr.add_argument('--channel', nargs='+', help='channels')
psr.add_argument('--runno', help='runno')
psr.add_argument('-o', dest='opt', help='name of output file')
psr.add_argument('--entries', dest='entries', help='entries file name')
args = psr.parse_args()
# 使用uproot读入波形数据
runno = args.runno
process = subprocess.Popen("ls {}/{} -v".format(args.datadir, runno), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
path = process.stdout.read().decode('utf-8').split('\n')
path = ['{}/{}/'.format(args.datadir, runno)+i for i in path[:-1]]
ch = uproot.concatenate([i+':Readout' for i in path[:1]], filter_name='ChannelId',library='np')['ChannelId'][0]
# h5py读入分析结果
anainfo = []
info = []
ER = EntriesReader(args.entries)
channels = [int(c) for c in args.channel]
with h5py.File(args.ana, 'r') as ipt:
    for j in range(len(args.channel)):
        anainfo.append(ipt['ch{}'.format(channels[j])][:])
    trigger = ipt['trigger'][:]
with h5py.File(args.pulseana, 'r') as ipt:
    for j in range(len(channels)):
        info.append(ipt['ch{}'.format(channels[j])][:])
    trigger = ipt['trigger'][:]
with h5py.File(args.interval, 'r') as ipt:
    rinterval = ipt['rinterval'][:]
intervalCenters = [int(ti) for ti in rinterval['mean']]
chmap = Series(range(ch.shape[0]), index=ch)
triggerDf = pd.DataFrame(trigger).set_index('EventID')
with PdfPages(args.opt) as pdf:
    for j in range(len(args.channel)):
        anainfoDf = pd.DataFrame(anainfo[j]).set_index('EventID')
        select1 = info[j][:,1]['isTrigger']
        select2 = info[j][:,2]['isTrigger']&((info[j][:,2]['begin10']- trigger['triggerTime'])>(intervalCenters[j] + 10))
        select = select1&select2
        eventIDs = info[j][:,1][select]['EventID']
        # 绘制j channel
        fthtmp = -1
        for eid, ids in zip(eventIDs[:1],info[j][select,:][:1]):
            fig, ax = plt.subplots()
            fth = ER.getFileNum(eid)
            if fth != fthtmp:
                with uproot.open(path[fth]) as ipt:
                    waveforms = ipt['Readout/Waveform'].array(library='np')
                    eids = ipt['Readout/TriggerNo'].array(library='np')
                fthtmp = fth
            referT = ids[1]['begin10']
            start = int(referT)
            wave = waveforms[eid-eids[0]].reshape((ch.shape[0],-1))[chmap.loc[channels[j]]]
            ax.plot(np.arange((start+config.laserB), (start+config.elasticE)) - triggerDf.loc[eid]['triggerTime'], anainfoDf.loc[eid]['baseline'] - wave[(start+config.laserB):(start+config.elasticE)], label='waveform')
            ax.fill_between(np.arange(int(rinterval[j]['start']), int(rinterval[j]['end'])), 0, 5, color='pink', alpha=0.5, label='candidate window')
            ax.axvline(ids[2]['begin10'] - triggerDf.loc[eid]['triggerTime'], color='g', ls='--', label='$t_r^{10}$ of delayed pulse')
            ax.axvline(referT - triggerDf.loc[eid]['triggerTime'], color='r', ls='--', label='$t_r^{10}$ of main pulse')
            ax.xaxis.set_minor_locator(MultipleLocator(2))
            ax.xaxis.set_major_locator(MultipleLocator(10))
            ax.legend()
            ax.set_title('eid:' + str(eid))
            ax.set_xlabel('$t-t_{\mathrm{trig}}$/ns')
            ax.set_ylabel('Voltage/ADC')
            pdf.savefig(fig)
            plt.close()