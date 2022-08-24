'''
获取并拟合ser, 平均ser在10%处对齐
'''
import h5py, argparse, numpy as np, uproot
from waveana.util import fitSER, SER
from pandas import Series
import matplotlib.pyplot as plt
plt.style.use('./journal.mplstyle')
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import config
import time
psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output png file dir')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1], help='channel used in DAQ')
psr.add_argument('-t', dest='trigger', help='trigger h5 file')
psr.add_argument('--wave', dest='wave', help='input root wave files')
psr.add_argument('--ana', dest="ana", help='reference for peak')
psr.add_argument('--summary', dest='summary', help='summary result')
psr.add_argument('--merge', default=False, action='store_true')
psr.add_argument('--serfiles', nargs='+', help='ser files to merge')
args = psr.parse_args()
fitdtype = [('eid', '<i2'), ('mu', '<f4'), ('sigma', '<f4'), ('tau', '<f4'), ('A', '<f4'), ('fun', '<f4'), ('suc', '?')]
spelength, spestart, speend = config.spelength, config.spestart, config.speend
ch = [int(i) for i in args.channel]
if not args.merge:
    info=[]
    thresholds = []
    with h5py.File(args.ana, 'r') as ana_ipt, h5py.File(args.summary, 'r') as sum_ipt:
        for j in range(len(args.channel)):
            info.append(ana_ipt['ch{}'.format(args.channel[j])][:])
            thresholds.append(
                (sum_ipt['res'][j]['peakC'] * (1 - sum_ipt['res'][j]['GainSigma']/sum_ipt['res'][j]['Gain']), sum_ipt['res'][j]['peakC'] * (1 + sum_ipt['res'][j]['GainSigma']/sum_ipt['res'][j]['Gain']))
                )
        trigger = ana_ipt['trigger'][:]
    # print(thresholds)
    t_b = time.time()
    with uproot.open(args.wave) as ipt:
        eventIds = ipt["Readout/TriggerNo"].array(library='np')
        waveforms = ipt["Readout/Waveform"].array(library='np')
        channelIds = ipt["Readout/ChannelId"].array(library='np')
    print('load root consume:{:.2f}s'.format(time.time()-t_b))
    nchannel = len(channelIds[0])
    waveformLength = int(len(waveforms[0])/nchannel)
    chmap = Series(range(channelIds[0].shape[0]), index=channelIds[0])

    storeWave = np.zeros((len(args.channel), spelength))
    nums = np.zeros(len(args.channel), dtype=int)
    fitResult = []
    for j in range(len(args.channel)):
        # 筛选出对应的波形
        indexTF = (info[j]['minPeakCharge']>thresholds[j][0])&(info[j]['minPeakCharge']<thresholds[j][1])&(info[j]['FWHM']>5)&(info[j]['minPeak']>3)
        index = np.where(indexTF)[0]
        nums[j] = np.sum(indexTF)
        fitResult.append(np.zeros((nums[j]), dtype=fitdtype))
        # print('select number is {}'.format(nums[j]))
        if(index.shape[0]>0):
            for idx, i in enumerate(index):
                trig = int(info[j][i]['begin10'] + trigger[i]['triggerTime'])
                baseline = info[j][i]['baseline']
                begin = trig - spestart
                end = begin + spelength
                wave = waveforms[i].reshape((nchannel,-1))
                ## 此处转为正向
                storeWave[j] += baseline - wave[chmap.loc[ch[j]]][begin:end]
                ## fit the SER
                xs = np.arange(begin, end)
                result = fitSER(xs, baseline - wave[chmap.loc[ch[j]]][begin:end])
                if not result.success:
                    print('eid-cid {}-{} failed:{}'.format(i, ch[j], result.message))
                fitResult[j][idx] = (i, *result.x, result.fun, result.success)
    with h5py.File(args.opt,'w') as opt:
        opt.create_dataset('spe', data=storeWave, compression='gzip')
        opt.create_dataset('nums', data=nums, compression='gzip')
        for j in range(len(args.channel)):
            opt.create_dataset('ch{}'.format(ch[j]), data=fitResult[j], compression='gzip')
else:
    storeWave = np.zeros((len(args.channel), spelength))
    nums = np.zeros(len(args.channel), dtype=int)
    fitResult = []
    # 合并ser及拟合参数
    for f in args.serfiles:
        with h5py.File(f, 'r') as ipt:
            storeWave += ipt['spe'][:]
            nums += ipt['nums'][:]
    print(nums)
    for j in range(len(args.channel)):
        fitResult.append(np.zeros(nums[j], dtype=fitdtype))
    cursor = np.zeros(len(args.channel), dtype=int)
    for f in args.serfiles:
        with h5py.File(f, 'r') as ipt:
            num = ipt['nums'][:]
            for j in range(len(args.channel)):
                fitResult[j][cursor[j]:(cursor[j]+num[j])] = ipt['ch{}'.format(ch[j])][:]
            cursor += num
    with h5py.File(args.opt,'w') as opt:
        opt.create_dataset('spe', data=storeWave/nums[:, None], compression='gzip')
        opt.create_dataset('nums', data=nums, compression='gzip')
        for j in range(len(args.channel)):
            opt.create_dataset('ch{}'.format(ch[j]), data=fitResult[j], compression='gzip')
    # 绘制图像
    jet = plt.cm.jet
    newcolors = jet(np.linspace(0, 1, 32768))
    white = np.array([1, 1, 1, 0.5])
    newcolors[0, :] = white
    cmap = ListedColormap(newcolors)
    with PdfPages(args.opt + '.pdf') as pdf:
        for j in range(len(args.channel)):
            fig, ax = plt.subplots()
            ax.plot(storeWave[j]/nums[j], label='SER')
            xs = range(spelength)
            result = fitSER(xs, storeWave[j]/nums[j])
            ax.plot(xs, SER(result.x, xs), linestyle='--', alpha=0.8, label='fit')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(mpatches.Patch(color='none', label='$\mu$:{:.2f}'.format(result.x[0])))
            handles.append(mpatches.Patch(color='none', label='$\sigma$:{:.2f}'.format(result.x[1])))
            handles.append(mpatches.Patch(color='none', label=r'$\tau$'+':{:.2f}'.format(result.x[2])))
            ax.legend(handles=handles)
            ax.set_xlabel('t/ns')
            ax.set_ylabel('Peak/ADC')
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots()
            h = ax.hist2d(fitResult[j]['sigma'], fitResult[j]['tau'], bins=[100,100], cmap=cmap)
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel('$\sigma$/ADCns')
            ax.set_ylabel(r'$\tau$/ns')
            pdf.savefig(fig)
            plt.close()