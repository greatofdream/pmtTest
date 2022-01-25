#!/usr/bin/python3
import matplotlib.pyplot as plt
import h5py, argparse
from scipy.optimize import minimize
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
# 分析触发时间的大致窗口，选择触发前后3sigma范围作为interval存储
def likelihood(x, *args):
    mu, sigma = x
    times = args
    N = times.shape[0]
    return np.sum((times-mu)**2/sigma**2/2) + N * np.log(sigma)
def fitInterval(minpeakpos, minpeak, threshold=5):
    timeselect = minpeakpos[minpeak>threshold]
    x0 = [np.mean(timeselect), np.std(timeselect)]
    result = minmize(likelihood, x0, args=(timeselect))
def getInterval(minpeakpos, minpeak, threshold=5):
    # TODO: 使用histogram处理
    hist = np.histogram(minpeakpos[minpeak>threshold], bins=200, range=[200,400])
    mean, sigma = int(hist[1][np.argmax(hist[0])]), np.std(minpeakpos[minpeak>threshold])
    begin, end = int(mean-3*sigma), int(mean+3*sigma)
    return (begin, end, mean, sigma)
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input h5 file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
    args = psr.parse_args()
    info = []
    with h5py.File(args.ipt, 'r') as ipt:
        waveformLength = ipt.attrs['waveformLength']
        #waveformLength = 1500
        for j in range(len(args.channel)):
            info.append(ipt['ch{}'.format(args.channel[j])][:])
    interval = np.zeros((len(args.channel),), dtype=[('start', np.float64), ('end', np.float64), ('mean', np.float64), ('sigma', np.float64)])
    for j in range(len(args.channel)):
        interval[j] = getInterval(info[j]['minPeakPos'], info[j]['minPeak'])
    with h5py.File(args.opt, 'w') as opt:
        opt.create_dataset('interval', data=interval, compression='gzip')