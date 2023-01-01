'''
绘制afterpulse计算示意图
'''
# 添加模块路径
import sys
sys.path.append('..')
import h5py, numpy as np, uproot
from pandas import Series
from waveana.util import getIntervals
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
promptB, promptE = config.promptB, config.promptE
delay1B, delay1E = config.delay1B, config.delay1E
delay10B, delay10E = config.delay10B, config.delay10E
spelength, spestart, speend = config.spelength, config.spestart, config.speend

if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('--datadir', help='directory of data')
    psr.add_argument('--ana', help='analyze result')
    psr.add_argument('-o', dest='opt', help='output figure file')
    psr.add_argument('--run', type=int, help='run no')
    psr.add_argument('--cid', type=int, help='channel id')
    psr.add_argument('-t', dest='trigger', type=int, help='trigger channel')
    psr.add_argument('--ser', help='ser file')
    args = psr.parse_args()
    runno = args.run
    cid = args.cid
    triggerch = args.trigger
    waveCut = 600
    # 使用uproot读入前11个波形数据
    process = subprocess.Popen("ls {}/{} -v".format(args.datadir, runno), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path = process.stdout.read().decode('utf-8').split('\n')
    path = ['{}/{}/'.format(args.datadir, runno)+i for i in path[:-1]]
    waveforms = uproot.concatenate([i+':Readout' for i in path[:10]], filter_name='Waveform',library='np')['Waveform']
    ch = uproot.concatenate([i+':Readout' for i in path[:1]], filter_name='ChannelId',library='np')['ChannelId'][0]
    # h5py读入分析结果
    with h5py.File(args.ana, 'r') as ipt:
        info = ipt['ch{}'.format(cid)][:]
        trigger = ipt['trigger'][:]
    chmap = Series(range(ch.shape[0]), index=ch)
    with PdfPages(args.opt) as pdf:
        # 获取波形
        fignum = 0
        for eid in np.where((info['isTrigger'] & (info['FWHM']>5) & (info['minPeakCharge']>50))[:len(waveforms)])[0]:
            wave = waveforms[eid].reshape((ch.shape[0],-1))
            chwave = wave[chmap.loc[cid]][:]
            ## 处理触发后的波形切分
            start = int(trigger[eid]['triggerTime'] + info[eid]['begin10'] + delay1B)
            ## 检查前脉冲
            end = int(trigger[eid]['triggerTime'] + info[eid]['begin10']) - config.anapromptE
            baseline, std = info[eid]['baseline'], info[eid]['std']
            if np.max(baseline - chwave[start:])<3 or np.max(baseline - chwave[:end]) < 3:
                continue
            wavelength = chwave.shape[0]
            threshold = np.max([5 * std, 3])
            intervals = getIntervals(np.arange(start, wavelength), baseline - chwave[start:], threshold, spestart, speend)
            print(info[eid]['EventID'], intervals, np.max(baseline - chwave[start:]))
            intervals_pre = getIntervals(np.arange(end), baseline - chwave[:end], threshold, spestart, speend)
            print(info[eid]['EventID'], intervals_pre, np.max(baseline - chwave[:end]))
            print('begin plot')
            # 绘制原始波形切分范围
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6), gridspec_kw={'width_ratios':[intervals[-1][-1]-150, wavelength-10200]})
            fig.subplots_adjust(wspace=0.05)
            ax1.axvline(trigger[eid]['triggerTime'], linestyle='--', linewidth=1, alpha=0.5, color='g', label='trigger time')
            ax1.axvline(trigger[eid]['triggerTime'] + info[eid]['begin10'], linestyle='--', linewidth=1, alpha=0.5, color='r', label='$t_{10}^r$')
            ax1.plot(baseline - chwave, linewidth=0.2, label='PMT waveform')
            ax2.plot(baseline - chwave, linewidth=0.2)
            maxy = np.max(baseline-chwave)
            for interval, label in zip(intervals, np.concatenate([['Pulse interval'], ['_']*(len(intervals)-1)])):
                line = ax1.fill_between(interval, [0, 0], [maxy/2, maxy/2], alpha=0.5, color='violet', label=label)
            for interval, label in zip(intervals_pre, np.concatenate([['Pulse interval'], ['_']*(len(intervals_pre)-1)])):
                ax1.fill_between(interval, [0, 0], [maxy/2, maxy/2], alpha=0.5, color='violet', label=label)
            ax1.fill_between([start, int(trigger[eid]['triggerTime'] + info[eid]['begin10'])+delay10E], [0, 0], [3, 3], alpha=0.5, color='gray', label='After-pulse search window')
            ax2.fill_between([start, int(trigger[eid]['triggerTime'] + info[eid]['begin10'])+delay10E], [0, 0], [3, 3], alpha=0.5, color='gray')
            ax1.fill_between([int(trigger[eid]['triggerTime'] + info[eid]['begin10'])-promptB, int(trigger[eid]['triggerTime'] + info[eid]['begin10'])-promptE], [0, 0], [3, 3], alpha=0.5, color='orange', label='Pre-pulse search window')
            ax1.set_xlabel('t/ns')
            ax1.set_ylabel('Amplitude/ADC')
            ax1.set_xlim([200, intervals[-1][-1]+50])            
            ax2.set_xlim([10200, wavelength])
            # ax.set_xscale('log')
            ax1.xaxis.set_minor_locator(MultipleLocator(100))
            ax2.xaxis.set_minor_locator(MultipleLocator(100))
            ax1.legend()
            # hide the spines between ax and ax2
            ax1.spines.right.set_visible(False)
            ax2.spines.left.set_visible(False)
            ax2.yaxis.tick_right()
            ax2.tick_params(labelright=False)  # don't put tick labels at the top
            # ax2.xaxis.tick_bottom()
            d = .5  # proportion of vertical to horizontal extent of the slanted line
            kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
                        linestyle="none", color='k', mec='k', mew=1, clip_on=False)
            ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
            ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
            pdf.savefig(fig)
            plt.close()
            fignum += 1
            if fignum > 1:
                break