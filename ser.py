'''
获取并拟合ser, 平均ser在10%处对齐
'''
import h5py, argparse, numpy as np, uproot
from waveana.util import fitSER, SER, peakNum
from pandas import Series
import matplotlib.pyplot as plt
plt.style.use('./journal.mplstyle')
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
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
fitdtype = [('eid', '<i4'), ('mu', '<f4'), ('sigma', '<f4'), ('tau', '<f4'), ('A', '<f4'), ('fun', '<f4'), ('suc', '?'), ('maxerr', '<f4'), ('std', '<f4'), ('peakNum', '<i2')]
spelength, spestart, speend = config.spelength, config.spestart, config.speend
ch = [int(i) for i in args.channel]
thresholds = []
info=[]
with h5py.File(args.ana, 'r') as ana_ipt, h5py.File(args.summary, 'r') as sum_ipt:
    for j in range(len(args.channel)):
        info.append(ana_ipt['ch{}'.format(args.channel[j])][:])
        thresholds.append(
            # (sum_ipt['res'][j]['peakC'] * (1 - sum_ipt['res'][j]['GainSigma']/sum_ipt['res'][j]['Gain']), sum_ipt['res'][j]['peakC'] * (1 + sum_ipt['res'][j]['GainSigma']/sum_ipt['res'][j]['Gain']))
            (0.5 * sum_ipt['res'][j]['peakC'], 1000)
            )
    trigger = ana_ipt['trigger'][:]
    peakCs = sum_ipt['res'][:]
if not args.merge:
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
        indexTF = (info[j]['minPeakCharge']>thresholds[j][0])&(info[j]['minPeakCharge']<thresholds[j][1])&(info[j]['FWHM']>2)&(info[j]['FWHM']<15)#&(info[j]['minPeak']>5)
        index = np.where(indexTF)[0]
        fitResult.append(np.zeros((np.sum(indexTF)), dtype=fitdtype))
        # print('select number is {}'.format(nums[j]))
        if(index.shape[0]>0):
            for idx, i in enumerate(index):
                trig = int(info[j][i]['begin10'] + trigger[i]['triggerTime'])
                baseline = info[j][i]['baseline']
                begin = trig - spestart
                end = begin + spelength
                wave = waveforms[i].reshape((nchannel,-1))
                ## 此处转为正向
                peakn = peakNum(baseline - wave[chmap.loc[ch[j]]][begin:end], info[j][i]['std'])
                ## fit the SER
                xs = np.arange(begin, end)
                result = fitSER(xs, baseline - wave[chmap.loc[ch[j]]][begin:end])
                eys = SER(result.x, xs)
                if not result.success:
                    print('eid-cid {}-{} failed:{}'.format(i, ch[j], result.message))
                fitResult[j][idx] = (
                    eventIds[i], *result.x, result.fun, result.success,
                    np.max(np.abs(baseline - wave[chmap.loc[ch[j]]][begin:end] - eys)),
                    info[j][i]['std'], peakn
                    )
                # 排除识别的单峰和拟合偏差大的波形
                if peakn==1 and fitResult[j][idx]['maxerr']<3*info[j][i]['std']:
                    storeWave[j] += baseline - wave[chmap.loc[ch[j]]][begin:end]
                    nums[j] += 1
    with h5py.File(args.opt,'w') as opt:
        opt.create_dataset('spe', data=storeWave, compression='gzip')
        opt.create_dataset('nums', data=nums, compression='gzip')
        for j in range(len(args.channel)):
            opt.create_dataset('ch{}'.format(ch[j]), data=fitResult[j], compression='gzip')
else:
    storeWave = np.zeros((len(args.channel), spelength))
    nums = np.zeros(len(args.channel), dtype=int)
    totalnums = np.zeros(len(args.channel), dtype=int)
    fitResult = []
    # 合并ser及拟合参数
    for f in args.serfiles:
        with h5py.File(f, 'r') as ipt:
            storeWave += ipt['spe'][:]
            nums += ipt['nums'][:]
            for j in range(len(args.channel)):
                totalnums[j] += ipt['ch{}'.format(ch[j])][:].shape[0]
    print(nums)
    for j in range(len(args.channel)):
        fitResult.append(np.zeros(totalnums[j], dtype=fitdtype))
    cursor = np.zeros(len(args.channel), dtype=int)
    for f in args.serfiles:
        with h5py.File(f, 'r') as ipt:
            for j in range(len(args.channel)):
                num = ipt['ch{}'.format(ch[j])][:].shape[0]
                fitResult[j][cursor[j]:(cursor[j]+num)] = ipt['ch{}'.format(ch[j])][:]
                cursor[j] += num
    # 绘制图像
    jet = plt.cm.jet
    newcolors = jet(np.linspace(0, 1, 32768))
    white = np.array([1, 1, 1, 0.5])
    newcolors[0, :] = white
    cmap = ListedColormap(newcolors)
    fitSummary = np.zeros(len(args.channel), dtype=[
        ('Channel', '<i2'), ('tau', '<f4'), ('sigma', '<f4'), ('tau_sigma', '<f4'), ('sigma_sigma', '<f4'),
        ('tau_total', '<f4'), ('sigma_total', '<f4'), ('SelectNum', '<i4')
        ])
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
            h = ax.hist(fitResult[j]['A'], bins=100, histtype='step')
            ax.set_xlabel('A')
            ax.set_ylabel(r'Entries')
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots()
            h = ax.hist2d(fitResult[j]['sigma'], fitResult[j]['tau'], bins=[100,100], cmap=cmap)
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel('$\sigma$/ns')
            ax.set_ylabel(r'$\tau$/ns')
            pdf.savefig(fig)
            plt.close()

            fig, ax = plt.subplots()
            # 剔除拟合非常坏的波形
            select = (fitResult[j]['maxerr']<3*fitResult[j]['std']) & (fitResult[j]['peakNum']==1)
            print('{}/{}'.format(np.sum(select), fitResult[j].shape[0]))
            h = ax.hist2d(fitResult[j]['sigma'][select], fitResult[j]['tau'][select], bins=[100,100], cmap=cmap)
            ax.scatter(np.mean(fitResult[j]['sigma'][select]), np.std(fitResult[j]['tau'][select]), marker='*', s=20)
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel('$\sigma$/ns')
            ax.set_ylabel(r'$\tau$/ns')
            pdf.savefig(fig)
            plt.close()
            fitSummary[j] = (args.channel[j], np.mean(fitResult[j]['tau'][select]), np.mean(fitResult[j]['sigma'][select]), np.std(fitResult[j]['tau'][select]), np.std(fitResult[j]['sigma'][select]), result.x[2], result.x[1], np.sum(select))
            # 检查tau, sigma和A的关联性
            fig, ax = plt.subplots()
            h = ax.hist2d(fitResult[j]['A'][select], fitResult[j]['tau'][select], bins=[100,100], cmap=cmap)
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel('A')
            ax.set_ylabel(r'$\tau$/ns')
            pdf.savefig(fig)
            plt.close()
            fig, ax = plt.subplots()
            h = ax.hist2d(fitResult[j]['A'][select], fitResult[j]['sigma'][select], bins=[100,100], cmap=cmap)
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel('A')
            ax.set_ylabel(r'$\sigma$/ns')
            pdf.savefig(fig)
            plt.close()
            # 检查chi2值
            fig, ax = plt.subplots()
            h = ax.hist2d(fitResult[j]['sigma'][select] + fitResult[j]['tau'][select], fitResult[j]['fun'][select], bins=[100,100], cmap=cmap)
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel(r'$\sigma+\tau$/ns')
            ax.set_ylabel('$\chi^2$')
            pdf.savefig(fig)
            plt.close()
            # peak分布
            fig, ax = plt.subplots()
            h = ax.hist(info[j]['minPeak'],histtype='step', bins=1000, range=[0,1000], label='peak')
            ax.hist(info[j]['minPeak'][(info[j]['minPeakCharge']>thresholds[j][0])&(info[j]['minPeakCharge']<thresholds[j][1])], histtype='step', bins=1000, range=[0, 1000], alpha=0.8, label='peak($C\in [C_1-\sigma_{C_1},C_1+\sigma_{C_1}]$)')
            ax.set_xlabel('Peak/ADC')
            ax.set_ylabel('Entries')
            ## zoom in
            ax.axvline(3, linestyle='--', color='g', label='3ADC')
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.set_yscale('linear')
            ax.set_xlim([0, 50])
            ax.set_ylim([0, 2*np.max(h[0][5:30])])
            ax.legend()
            pdf.savefig(fig)
    
    with h5py.File(args.opt,'w') as opt:
        opt.create_dataset('spe', data=storeWave/nums[:, None], compression='gzip')
        opt.create_dataset('nums', data=nums, compression='gzip')
        for j in range(len(args.channel)):
            opt.create_dataset('ch{}'.format(ch[j]), data=fitResult[j], compression='gzip')
        opt.create_dataset('mu', data=peakCs, compression='gzip')
        opt.create_dataset('res', data=fitSummary, compression='gzip')
