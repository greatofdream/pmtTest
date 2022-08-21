'''
获取并拟合ser, 平均ser在10%处对齐
'''
import h5py, argparse, numpy as np, uproot
from pandas import Series
import matplotlib.pyplot as plt
plt.style.use('./journal.mplstyle')
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import config
psr = argparse.ArgumentParser()
psr.add_argument('-o', dest='opt', help='output png file dir')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-t', dest='trigger', help='trigger h5 file')
psr.add_argument('--wave', dest='wave', nargs='+', help='input root wave files')
psr.add_argument('--ana', dest="ana", help='reference for peak')
psr.add_argument('--summary', dest='summary', help='summary result')
args = psr.parse_args()
info=[]
thresholds = []
spelength, spestart, speend = config.spelength, config.spestart, config.speend
with h5py.File(args.ana, 'r') as ana_ipt, h5py.File(args.summary, 'r') as sum_ipt:
    for j in range(len(args.channel)):
        info.append(ana_ipt['ch{}'.format(args.channel[j])][:])
    thresholds.append(
        (sum_ipt['res']['peakC'] * (1 + sum_ipt['res']['GainSigma']/sum_ipt['res']['Gain']), sum_ipt['res']['peakC'] * (1 + sum_ipt['res']['GainSigma']/sum_ipt['res']['Gain']))
        )

ch = [int(i) for i in args.channel]
pdf = PdfPages(args.opt+'.pdf')
waveforms = uproot.concatenate([i+':Readout' for i in args.wave], filter_name='Waveform',library='np')['Waveform']

with uproot.open(args.root[0]) as ipt:
    channelIds = ipt["Readout/ChannelId"].array(library='np')
nchannel = len(channelIds[0])
waveformLength = int(len(waveforms[0])/nchannel)
chmap = Series(range(channelIds[0].shape[0]), index=channelIds[0])

indexs = []
storeWave = np.zeros((len(args.channel), spelength))
for j in range(len(args.channel)):
    # 筛选出对应的波形
    indexTF = (info[j]['minPeakCharge']>thresholds[j][0])&(info[j]['minPeakCharge']<thresholds[j][1])&(info[j]['FWHM']>5)&(info[j]['minPeak']>3)
    index = np.where(indexTF)[0]
    print('select number is {}'.format(np.sum(indexTF)))
    if(index.shape[0]>0):
        for i in index:
            pos0 = info[j][i]['minPeakPos']
            trig = int(info[j][i]['begin10'])
            baseline = info[j][i]['baseline']
            begin = trig - spestart
            end = begin + spelength
            wave = waveforms[i].reshape((nchannel,-1))
            ## 此处转为正向
            storeWave[j] += baseline - wave[chmap.loc[ch[j]]][begin:end]

    fig, ax = plt.subplots()
    ax.plot(storeWave[j]/index.shape[0])
    ax.set_xlabel('t/ns')
    ax.set_ylabel('Peak/ADC')
    pdf.savefig(fig)
    plt.close()

pdf.close()
with h5py.File(args.opt,'w') as opt:
    opt.create_dataset('spe', data=storeWave,compression='gzip')