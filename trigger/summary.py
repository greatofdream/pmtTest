#!/usr/bin/python3
import matplotlib.pyplot as plt
import h5py, argparse
from scipy.optimize import minimize
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
NSIGMA = 5
# 分析触发时间的大致窗口，选择触发前后3sigma范围作为interval存储
def likelihood(x, *args):
    mu, sigma = x
    times = args[0]
    N = times.shape[0]
    return np.sum((times-mu)**2/sigma**2/2) + N * np.log(sigma)
def fitInterval(x0, timeselect):
    result = minimize(likelihood, x0, args=(timeselect))
    return result.x
def getInterval(minpeakpos, minpeak, threshold=5):
    # TODO: 使用histogram处理
    hist = np.histogram(minpeakpos[minpeak>threshold], bins=400, range=[0,400])
    # 计算拟合初值和均方差
    mean = int(hist[1][np.argmax(hist[0])])
    sigma = np.std(minpeakpos[(minpeak>threshold)&(minpeakpos>(mean-10))&(minpeakpos<(mean+10))])
    # 启动拟合
    timeselect = minpeakpos[(minpeak>threshold)&(minpeakpos>(mean-5*sigma))&(minpeakpos<(mean+5*sigma))]
    mean, sigma = fitInterval((mean, sigma), timeselect)
    begin, end = int(mean-NSIGMA*sigma), int(mean+NSIGMA*sigma)+1
    return (begin, end, mean, sigma), timeselect.shape[0]
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input h5 file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
    args = psr.parse_args()
    info = []
    with h5py.File(args.ipt, 'r') as ipt:
        waveformLength = ipt.attrs['waveformLength']
        for j in range(len(args.channel)):
            info.append(ipt['ch{}'.format(args.channel[j])][:])
        trigger = ipt['trigger'][:]
    interval = np.zeros((len(args.channel),), dtype=[('start', np.float64), ('end', np.float64), ('mean', np.float64), ('sigma', np.float64)])
    relative_interval = np.zeros((len(args.channel),), dtype=[('start', np.float64), ('end', np.float64), ('mean', np.float64), ('sigma', np.float64)])
    pdf = PdfPages(args.opt+'.pdf')
    for j in range(len(args.channel)):
        interval[j], N = getInterval(info[j]['minPeakPos'], info[j]['minPeak'])
        relative_interval[j], N = getInterval(info[j]['minPeakPos'] - trigger['triggerTime'], info[j]['minPeak'])
        print(interval[j], relative_interval[j])
        fig, ax = plt.subplots()
        ax.hist((info[j]['minPeakPos'] - trigger['triggerTime'])[info[j]['minPeak']>5], bins=100, range=(relative_interval[j]['mean']-10*relative_interval[j]['sigma'],relative_interval[j]['mean']+10*relative_interval[j]['sigma']), histtype='step', label='TT {:.2f}$\pm${:.2f}'.format(*relative_interval[j][['mean','sigma']]))
        xs = np.arange(relative_interval[j]['mean']-10*relative_interval[j]['sigma'],relative_interval[j]['mean']+10*relative_interval[j]['sigma'], relative_interval[j]['sigma']/5)
        ax.plot(xs, N*relative_interval[j]['sigma']/5*np.exp(-(xs-relative_interval[j]['mean'])**2/2/relative_interval[j]['sigma']**2)/np.sqrt(2*np.pi)/relative_interval[j]['sigma'], label='fit result')
        ax.axvline(relative_interval[j]['start'], color='r')
        ax.axvline(relative_interval[j]['end'], color='r')
        ax.set_xlabel('TT/ns')
        ax.set_ylabel('entries')
        ax.legend()
        pdf.savefig(fig)
    pdf.close()
    with h5py.File(args.opt, 'w') as opt:
        opt.create_dataset('interval', data=interval, compression='gzip')
        opt.create_dataset('rinterval', data=relative_interval, compression='gzip')