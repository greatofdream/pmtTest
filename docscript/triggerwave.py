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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6), gridspec_kw={'width_ratios':[50, 200]})
        fig.subplots_adjust(wspace=0.05)
        # axins = ax1.twinx()
        # ax.plot(chwave, label='PMT waveform')
        # ax.axhline(baseline1, linestyle='--', color='k', alpha=0.5)
        # ax.axhline(baseline2, linestyle='--', color='k', alpha=0.5)
        starttime = 200
        ax1.plot(range(starttime, waveCut), chwave[starttime:waveCut], color='b')
        ax1.axhline(info[eid]['baseline'], color='b', linestyle='--')
        ax1.plot([starttime, waveCut], [info[eid]['baseline']-min(5,5*info[eid]['std']), info[eid]['baseline']-min(5,5*info[eid]['std'])], linestyle='dotted', color='violet')
        ax1.plot(triggerWave/200 + info[eid]['baseline']-10, color='orange', label='Trigger waveform(scaled)')
        ax1.scatter(trigger[eid]['triggerTime'], (baseline1 + baseline2)/2/200+info[eid]['baseline']-10, marker='x', s=300, c='g', label='$t_{\mathrm{trig}}$')
        ax2.plot(range(starttime, waveCut), chwave[starttime:waveCut], color='b', label='PMT waveform')
        ax2.plot(triggerWave/200 + info[eid]['baseline']-10, color='orange')
        ax2.plot([starttime, waveCut], [info[eid]['baseline']-min(5,5*info[eid]['std']), info[eid]['baseline']-min(5,5*info[eid]['std'])], linestyle='dotted', color='violet', label='voltage threshold')
        ax2.axhline(info[eid]['baseline'], color='b', linestyle='--')
        ax2.axvline(info[eid]['minPeakPos'], linestyle='--', color='y', label='$t_{p}$')
        ax2.axvline(info[eid]['begin10'], linestyle='--', color='r', label='$t^r_{10}$')
        ax2.fill_between(np.arange(info[eid]['minPeakPos']-config.baselength, info[eid]['minPeakPos']+config.afterlength), np.minimum(chwave[(info[eid]['minPeakPos']-config.baselength):(info[eid]['minPeakPos']+config.afterlength)]-0.5,info[eid]['baseline']-1), np.zeros(config.baselength + config.afterlength)+info[eid]['baseline']+0.5, color='pink', alpha=0.5, label='integration window')
        ## 绘制rise time, falling time
        begin10, begin50, begin90 = info[eid]['begin10'], info[eid]['begin50'], info[eid]['begin90']
        end10, end50, end90 = info[eid]['end10'], info[eid]['end50'], info[eid]['end90']
        baseline, minpeak = info[eid]['baseline'], info[eid]['minPeak']
        ax2.plot([begin10, end10], [baseline-0.1*minpeak, baseline-0.1*minpeak], linestyle='dotted', color='k')
        ax2.plot([begin10, end10], [baseline-0.9*minpeak, baseline-0.9*minpeak], linestyle='dotted', color='k')
        ax2.plot([begin10, end10], [baseline-0.5*minpeak, baseline-0.5*minpeak], linestyle='dotted', color='k')
        ax2.annotate("$\mu_b$-0.1$V_p$", (end10+5, baseline-0.15*minpeak), color='k')
        ax2.annotate("$\mu_b$-0.5$V_p$", (end10+5, baseline-0.5*minpeak), color='k')
        ax2.annotate("$\mu_b$-0.9$V_p$", (end10+5, baseline-0.9*minpeak), color='k')
        ### 绘制矩形区域
        ax2.fill_between([begin10, begin90], [baseline-0.1*minpeak, baseline-0.1*minpeak], [baseline-0.9*minpeak, baseline-0.9*minpeak], color='g', alpha=0.5, label='risetime')
        ax2.fill_between([end10, end90], [baseline-0.1*minpeak, baseline-0.1*minpeak], [baseline-0.9*minpeak, baseline-0.9*minpeak], color='c', alpha=0.5, label='falltime')
        ax1.set_xlabel('t/ns')
        ax1.set_ylabel('Voltage/ADC')
        ax1.set_xlim([starttime, 250])
        ax2.set_xlim([400, 550])
        ax1.xaxis.set_minor_locator(MultipleLocator(10))
        ax2.xaxis.set_minor_locator(MultipleLocator(10))
        ax1.xaxis.set_major_locator(MultipleLocator(50))
        ax2.xaxis.set_major_locator(MultipleLocator(50))
        # hide the spines between ax and ax2
        ax1.spines.right.set_visible(False)
        ax2.spines.left.set_visible(False)
        ax2.yaxis.tick_right()
        ax2.tick_params(labelright=False)
        ## 绘制放大波形在第二个坐标轴
        # 
        # axins = inset_axes(ax, width="50%", height="30%", loc='lower left',
        #            bbox_to_anchor=(0.47, 0.2, 1, 2),
        #            bbox_transform=ax.transAxes)
        
        print(info[eid]['begin10'])
        # axins.set_xlim([200, waveCut])
        # axins.set_ylabel('PMT voltage/ADC')
        # axins.set_xlim([int(trigger[eid]['triggerTime']) + 100, waveCut])
        # axins.xaxis.set_minor_locator(MultipleLocator(10))
        ## ax上绘制指示框和连接线
        # mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec='k', lw=0.5)

        d = .5  # proportion of vertical to horizontal extent of the slanted line
        kwargs = dict(marker=[(-d, -1), (d, 1)], markersize=12,
                    linestyle="none", color='k', mec='k', mew=1, clip_on=False)
        ax1.plot([1, 1], [0, 1], transform=ax1.transAxes, **kwargs)
        ax2.plot([0, 0], [0, 1], transform=ax2.transAxes, **kwargs)
        lines = []
        labels = []
        for ax_i in [ax2, ax1]:
            Line, Label = ax_i.get_legend_handles_labels()
            # print(Label)
            lines.extend(Line)
            labels.extend(Label)
        ax2.legend(lines, labels)
        pdf.savefig(fig)
