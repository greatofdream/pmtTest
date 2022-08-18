'''
绘制trigger计算示意图
'''
# 添加模块路径
import sys
sys.path.append('..')
import h5py, numpy as np, uproot
from pandas import Series
from waveana.triggerana import Triggerana
import config
import subprocess
import argparse
import matplotlib.pyplot as plt
plt.style.use('../journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# 波形长度，此处保持和论文一致，虽然实际采数时为602
wavelength = 600
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('--datadir', help='directory of data')
    psr.add_argument('--ana', help='analyze result')
    psr.add_argument('-o', dest='opt', help='output figure file')
    psr.add_argument('--run', type=int, help='run no')
    psr.add_argument('--eid', type=int, help='event id')
    psr.add_argument('--cid', type=int, help='channel id')
    psr.add_argument('-t', dest='trigger', type=int, help='trigger channel')
    args = psr.parse_args()
    runno = args.run
    eid = args.eid
    cid = args.cid
    triggerch = args.trigger
    waveCut = 600
    # 使用uproot读入波形数据
    process = subprocess.Popen("ls {}/{} -v".format(args.datadir, runno), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path = process.stdout.read().decode('utf-8').split('\n')
    path = ['{}/{}/'.format(args.datadir, runno)+i for i in path[:-1]]
    waveforms = uproot.concatenate([i+':Readout' for i in path[:11]], filter_name='Waveform',library='np')['Waveform']
    ch = uproot.concatenate([i+':Readout' for i in path[:11]], filter_name='ChannelId',library='np')['ChannelId'][0]
    # h5py读入分析结果
    with h5py.File(args.ana, 'r') as ipt:
        info = ipt['ch{}'.format(cid)][:]
        trigger = ipt['trigger'][:]
    chmap = Series(range(ch.shape[0]), index=ch)
    # 获取波形
    wave = waveforms[eid].reshape((ch.shape[0],-1))
    triggerWave = wave[chmap.loc[triggerch]][:waveCut]
    chwave =wave[chmap.loc[cid]][:wavelength]
    ## 重新implement激光触发波形的前后baseline
    baseline1 = np.average(triggerWave[0:50])
    baseline2 = np.average(triggerWave[-150:-50])
    with PdfPages(args.opt) as pdf:
        # 绘制原始波形切分范围
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(chwave, label='PMT waveform')
        ax.plot(triggerWave, label='trigger waveform')
        ax.axhline(baseline1, linestyle='--', alpha=0.5)
        ax.axhline(baseline2, linestyle='--', alpha=0.5)
        ax.scatter(trigger[eid]['triggerTime'], (baseline1 + baseline2)/2, marker='+', s=200, c='g', label='trigger time')
        ax.set_xlabel('t/ns')
        ax.set_ylabel('Amplitude/ADC')
        ax.set_xlim([0, waveCut])
        ## 绘制放大波形
        axins = inset_axes(ax, width="50%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.47, 0.2, 1, 2),
                   bbox_transform=ax.transAxes)
        axins.plot(range(int(trigger[eid]['triggerTime']) + 100, waveCut), chwave[(int(trigger[eid]['triggerTime']) + 100):waveCut])
        axins.axhline(info[eid]['baseline'], linestyle='--')
        axins.axvline(info[eid]['begin10'], linestyle='--', color='r', label='$t^r_{10}$')
        axins.axvline(info[eid]['minPeakPos'], linestyle='--', color='y', label='$t_{p}$')
        axins.set_xlim([int(trigger[eid]['triggerTime']) + 100, waveCut])
        axins.xaxis.set_minor_locator(MultipleLocator(10))
        ## ax上绘制指示框和连接线
        mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec='k', lw=0.5)

        ax.xaxis.set_minor_locator(MultipleLocator(10))
        lines = []
        labels = []
        for ax_i in [ax, axins]:
            Line, Label = ax_i.get_legend_handles_labels()
            # print(Label)
            lines.extend(Line)
            labels.extend(Label)
        ax.legend(lines, labels, loc='center left')
        pdf.savefig(fig)