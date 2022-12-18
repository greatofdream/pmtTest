'''
挑选100个触发波形，绘制ser拟合示意图
'''
# 添加模块路径
import subprocess
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
import os
spelength, spestart, speend = config.spelength, config.spestart, config.speend

psr = argparse.ArgumentParser()
psr.add_argument('--ana', help='analyze result')
psr.add_argument('--wave', help='wave of data')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-o', dest='opt', help='output figure file')
psr.add_argument('--cid', type=int, help='channel id')
psr.add_argument('--eid', type=int, nargs='+', help='event id')
psr.add_argument('--summary', dest='summary', help='summary result')
psr.add_argument('--run', type=int, help='run no')
args = psr.parse_args()
cid = args.cid
dir = args.wave
dir = os.path.dirname(args.wave)
entries = np.loadtxt('../ExResult/{}/entries.txt'.format(args.run))
process = subprocess.Popen("ls {} -v".format(dir), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
path = process.stdout.read().decode('utf-8').split('\n')
path = [dir +'/'+i for i in path[:-1]]
run_i = np.where((entries-args.eid)>=0)[0][0]
filename = path[run_i]
with uproot.open(filename) as ipt:
    eventIds = ipt["Readout/TriggerNo"].array(library='np')
    waveforms = ipt["Readout/Waveform"].array(library='np')
    channelIds = ipt["Readout/ChannelId"].array(library='np')
dir = os.path.dirname(args.ana)
process = subprocess.Popen("ls {}/*.h5 -v".format(dir), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
path = process.stdout.read().decode('utf-8').split('\n')
with h5py.File(path[run_i], 'r') as ipt:
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
eidmap = Series(range(len(eventIds)), index=eventIds)
index = eidmap.loc[args.eid].iloc[0]
chmap = Series(range(nchannel), index=channelIds[0])
with PdfPages(args.opt) as pdf:
    # 绘制原始波形切分范围
    trig = int(info[index]['begin10'] + trigger[index]['triggerTime'])
    print(info[index]['minPeak'], info[index]['minPeakCharge'])
    baseline = info[index]['baseline']
    begin = trig - spestart
    end = begin + spelength
    wave = waveforms[index].reshape((nchannel,-1))
    selectwave = baseline - wave[chmap.loc[cid]][begin:end]
    print(peakNum(selectwave, info[index]['std']))
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