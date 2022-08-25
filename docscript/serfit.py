'''
挑选100个触发波形，绘制ser拟合示意图
'''
# 添加模块路径
import sys
sys.path.append('..')
import h5py, numpy as np, uproot
np.seterr(divide='raise')
from pandas import Series
from waveana.util import fitSER, SER, peakNum
import config
import argparse
import matplotlib.pyplot as plt
plt.style.use('../journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
spelength, spestart, speend = config.spelength, config.spestart, config.speend

psr = argparse.ArgumentParser()
psr.add_argument('--ana', help='analyze result')
psr.add_argument('--wave', help='wave of data')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-o', dest='opt', help='output figure file')
psr.add_argument('--cid', type=int, help='channel id')
psr.add_argument('--eid', type=int, nargs='+', help='event id')
psr.add_argument('--summary', dest='summary', help='summary result')
args = psr.parse_args()
cid = args.cid
with uproot.open(args.wave) as ipt:
    eventIds = ipt["Readout/TriggerNo"].array(library='np')
    waveforms = ipt["Readout/Waveform"].array(library='np')
    channelIds = ipt["Readout/ChannelId"].array(library='np')
with h5py.File(args.ana, 'r') as ipt:
    info = ipt['ch{}'.format(cid)][:]
    trigger = ipt['trigger'][:]
ch = [int(i) for i in args.channel]
nchannel = len(channelIds[0])

thresholds = []
with h5py.File(args.summary, 'r') as sum_ipt:
    for j in range(len(ch)):
        thresholds.append(
            (sum_ipt['res'][j]['peakC'] * (1 - sum_ipt['res'][j]['GainSigma']/sum_ipt['res'][j]['Gain']), sum_ipt['res'][j]['peakC'] * (1 + sum_ipt['res'][j]['GainSigma']/sum_ipt['res'][j]['Gain']))
        )
# select the trigger wave
chmap = Series(range(len(ch)), index=ch)
j = chmap.loc[cid]
if args.eid:
    eidmap = Series(range(len(eventIds)), index=eventIds)
    index = eidmap.loc[args.eid]
else:
    indexTF = (info['minPeakCharge']>thresholds[j][0])&(info['minPeakCharge']<thresholds[j][1])&(info['FWHM']>5)&(info['minPeak']>3)
    index = np.where(indexTF)[0]
    print(index.shape, info.shape)
    print(index[:3])
chmap = Series(range(nchannel), index=channelIds[0])
with PdfPages(args.opt) as pdf:
    # 绘制原始波形切分范围
    for i in index[:100]:
        trig = int(info[i]['begin10'] + trigger[i]['triggerTime'])
        print(info[i]['minPeak'], info[i]['minPeakCharge'])
        baseline = info[i]['baseline']
        begin = trig - spestart
        end = begin + spelength
        wave = waveforms[i].reshape((nchannel,-1))
        selectwave = baseline - wave[chmap.loc[cid]][begin:end]
        print(peakNum(selectwave, info[i]['std']))
        ## fit the SER
        xs = np.arange(begin, end)
        result = fitSER(xs, selectwave)
        if not result.success:
            print('eid {} failed:{}'.format(i, result.message))
        fig, ax = plt.subplots()
        ax.plot(xs, selectwave, label='waveform')
        ax.plot(xs, SER(result.x, xs), linestyle='--', alpha=0.8, label='fit')
        handles, labels = ax.get_legend_handles_labels()
        handles.append(mpatches.Patch(color='none', label='$\mu$:{:.2f}'.format(result.x[0])))
        handles.append(mpatches.Patch(color='none', label='$\sigma$:{:.2f}'.format(result.x[1])))
        handles.append(mpatches.Patch(color='none', label=r'$\tau$'+':{:.2f}'.format(result.x[2])))
        ax.legend(handles=handles)
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
        ax.set_xlabel('t/ns')
        ax.set_ylabel('Amplitude/ADC')
        pdf.savefig(fig)
        plt.close()