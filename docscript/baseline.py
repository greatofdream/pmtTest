# 添加模块路径
import sys
sys.path.append('..')
import h5py, numpy as np, uproot
from pandas import Series
from waveana.waveana import Waveana, gausfit
import config
import subprocess
import argparse
import matplotlib.pyplot as plt
plt.style.use('../journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
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
    args = psr.parse_args()
    runno = args.run
    eid = args.eid
    cid = args.cid
    # 使用uproot读入波形数据
    process = subprocess.Popen("ls {}/{} -v".format(args.datadir, runno), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path = process.stdout.read().decode('utf-8').split('\n')
    path = ['{}/{}/'.format(args.datadir, runno)+i for i in path[:-1]]
    waveforms = uproot.concatenate([i+':Readout' for i in path], filter_name='Waveform',library='np')['Waveform']
    ch = uproot.concatenate([i+':Readout' for i in path], filter_name='ChannelId',library='np')['ChannelId'][0]
    # h5py读入分析结果
    with h5py.File(args.ana, 'r') as ipt:
        info = ipt['ch{}'.format(cid)][:]
    chmap = Series(range(ch.shape[0]), index=ch)
    # 获取波形
    wave = waveforms[eid].reshape((ch.shape[0],-1))[chmap.loc[cid]][:wavelength]
    waveana = Waveana()
    waveana.setWave(wave)
    waveana.getBaselineFine(waveana.minIndex)
    waveana.integrateWave()
    waveana.integrateMinPeakWave(config.baselength, config.afterlength)
    baseline, minpeak, minIndex = waveana.minPeakBaseline, waveana.minPeak, waveana.minIndex
    begin10, begin50, begin90= waveana.begin10, waveana.begin50, waveana.begin90
    end10, end50, end90 = waveana.end10, waveana.end50, waveana.end90
    # 由于拟合没有在类里存中间数据，重新implement在waveana里的拟合
    extractWave = waveana.wave[np.max([waveana.minIndex - 200, 0]):(waveana.minIndex - 10)]
    roughbaseline, roughstd = np.mean(extractWave), np.clip(np.std(extractWave), 1, 3)
    print(roughstd)
    nsigma = 5
    threshold = roughbaseline - np.min([5, nsigma*roughstd])
    x = gausfit(
            x0=[roughbaseline, roughstd],
            args=extractWave[extractWave>threshold],
            bounds=[
                (roughbaseline - nsigma * roughstd, roughbaseline + nsigma * roughstd),
                (0.001, nsigma * roughstd),
            ],
        )
    print(x.x)
    with PdfPages(args.opt) as pdf:
        # 绘制原始波形切分范围
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(wave, label='waveform')
        ax.axhline(baseline, linestyle='--', label='baseline')
        ## 绘制baseline计算区间
        ax.vlines(minIndex, baseline - minpeak, baseline,  linestyle='--', color='g')
        ax.annotate("$t_p$", (minIndex + 5, baseline-10), color='g')
        ax.annotate("baseline-$V_p$", (minIndex + 10, baseline-minpeak), color='g')
        x1, x2 = np.max([minIndex - 200, 0]), minIndex - 10
        y1, y2 = waveana.minPeakBaseline - 5, waveana.minPeakBaseline-5
        ax.annotate("", xy=(x1, y1), xycoords='data', xytext=(x2, y2),
                arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5",
                connectionstyle="arc3"),) 
        ax.annotate("-$t_s$~-10", (x1 + 10, y1 - 15))
        ### 绘制阈值
        ax.axhline(threshold, linestyle='dotted', color='violet', label='baseline threshold')
        x1, x2 = waveana.minIndex + 100, np.min([waveana.minIndex + 200, wavelength])
        y1, y2 = waveana.minPeakBaseline - 5, waveana.minPeakBaseline-5
        ax.annotate("", xy=(x1, y1), xycoords='data', xytext=(x2, y2),
                arrowprops=dict(arrowstyle="|-|, widthA=0.5, widthB=0.5",
                connectionstyle="arc3"),)
        ax.annotate("+100~+200", (x1 + 10, y1 - 15))
        
        ## 绘制rise time, falling time
        ax.plot([begin10, end10], [baseline-0.1*minpeak, baseline-0.1*minpeak], linestyle='dotted', color='r')
        ax.plot([begin90, end90], [baseline-0.9*minpeak, baseline-0.9*minpeak], linestyle='dotted', color='r')
        ax.plot([begin50, end50], [baseline-0.5*minpeak, baseline-0.5*minpeak], linestyle='dotted', color='r')
        ax.annotate("baseline-0.1$V_p$", (end10+5, baseline-0.1*minpeak), color='r')
        ax.annotate("baseline-0.5$V_p$", (end10+5, baseline-0.5*minpeak), color='r')
        ax.annotate("baseline-0.9$V_p$", (end10+5, baseline-0.9*minpeak), color='r')
        ### 绘制矩形区域
        ax.fill_between([begin10, begin90], [baseline-0.1*minpeak, baseline-0.1*minpeak], [baseline-0.9*minpeak, baseline-0.9*minpeak], color='pink', alpha=0.5, label='risetime')
        ax.fill_between([end10, end90], [baseline-0.1*minpeak, baseline-0.1*minpeak], [baseline-0.9*minpeak, baseline-0.9*minpeak], color='c', alpha=0.5, label='falltime')
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlabel('t/ns')
        ax.set_ylabel('Amplitude/ADC')
        ax.set_xlim([0, (x2//100 + 1) * 100])
        ax.legend(loc='best')
        pdf.savefig(fig)
        # 绘制baseline对应的histogram
        fig, ax = plt.subplots()
        ax.hist(extractWave, range=[930,980], bins=50, density=True, histtype='step', label='sample')
        ax.plot(np.arange(930, 980, 0.1), np.exp(-(np.arange(930, 980, 0.1)-x.x[0])**2/2/x.x[1]**2)/np.sqrt(2*np.pi), label='fit')
        ax.set_xlabel('Amplitude/ADC')
        ax.set_ylabel('PDF')
        ax.set_xlim([930, 980])
        ax.legend(loc='best')
        pdf.savefig(fig)
        # 绘制remove baseline的结果
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(wave - baseline, label='waveform w/o baseline')
        ax.axhline(0, linestyle='--')
        ax.fill_between([minIndex - config.baselength, minIndex + config.afterlength], [0, 0], [-5, -5], color='pink', alpha=0.5, label='integration window')
        ax.annotate("baseline-0.1$V_p$", (end10+5, baseline-0.1*minpeak), color='r')
        ax.set_xlim([np.max([minIndex - 200, 0]), np.min([minIndex + 300, wavelength])])
        ax.set_ylim([-10, 5])
        ax.yaxis.set_minor_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlabel('t/ns')
        ax.set_ylabel('Amplitude/ADC')
        ax.legend(loc='best')
        pdf.savefig(fig)