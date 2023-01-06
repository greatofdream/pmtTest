'''
Pre-pulse waveform check.
# python3 TESTPrePulse.py --ana ExResult/763/600ns/qe/preanalysisMerge.h5 --pulseana ExResult/763/600ns/pulseRatio.h5 --datadir /tarski/JNE/JinpingData/Jinping_1ton_Data/pmtTest --runno 763 --channel 2 3 4 5 --entries ExResult/763/entries.txt -o ExResult/763/600ns/TestPulse.pdf
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
psr.add_argument('--pulseana', help='the ana result directory')
psr.add_argument('--datadir', help='directory of data')
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
chmap = Series(range(ch.shape[0]), index=ch)
triggerDf = pd.DataFrame(trigger).set_index('EventID')
with PdfPages(args.opt) as pdf:
    for j in range(len(args.channel)):
        # infoDf = pd.DataFrame(info[j]).groupby('EventID')
        anainfoDf = pd.DataFrame(anainfo[j]).set_index('EventID')
        select = (info[j]['t']<100)&(info[j]['t']>-50)
        eventIDs, counts = np.unique(info[j][select]['EventID'], return_counts=True)
        # eventIDs = eventIDs[counts>1]
        infoDf = pd.DataFrame(info[j][select]).groupby('EventID')
        # 绘制j channel
        fthtmp = -1
        for eid in eventIDs[:100]:
            fig, ax = plt.subplots()
            fth = ER.getFileNum(eid)
            if fth != fthtmp:
                with uproot.open(path[fth]) as ipt:
                    waveforms = ipt['Readout/Waveform'].array(library='np')
                    eids = ipt['Readout/TriggerNo'].array(library='np')
                fthtmp = fth
            referT = anainfoDf.loc[eid]['begin10'] + triggerDf.loc[eid]['triggerTime']
            start = int(referT)
            wave = waveforms[eid-eids[0]].reshape((ch.shape[0],-1))[chmap.loc[channels[j]]]
            ax.plot(np.arange((start+config.laserB), (start+config.elasticE)), anainfoDf.loc[eid]['baseline'] - wave[(start+config.laserB):(start+config.elasticE)])
            ax.vlines(referT + infoDf.get_group(eid)['t'], ymin=0, ymax=5)
            ax.axvline(referT)
            ax.set_title('eid:' + str(eid))
            ax.set_xlabel('t/ns')
            ax.set_ylabel('Trigger Amplitude/ADC')
            pdf.savefig(fig)
            plt.close()