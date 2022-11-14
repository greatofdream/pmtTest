'''
根据trigger分析结果，选择trigger的波形，分析前后脉冲的时间和电荷分布
'''
#!/usr/bin/python3
import uproot, numpy as np, h5py
from pandas import Series
from waveana.util import getIntervals, getTQ
import argparse
import matplotlib.pyplot as plt
import config
promptB, promptE = config.promptB, config.promptE
delay1B, delay1E = config.delay1B, config.delay1E
delay10B, delay10E = config.delay10B, config.delay10E
spelength, spestart, speend = config.spelength, config.spestart, config.speend

if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input root file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
    psr.add_argument('-t', dest='trigger', default=0, type=int, help='trigger channel')
    psr.add_argument('-s', dest='nsigma', default=5, type=float, help='nsimga')
    psr.add_argument('--result', help='result of preanalysis')
    psr.add_argument('--summary', dest='summary', help='summary result')
    args = psr.parse_args()
    triggerch = args.trigger
    channels = [int(c) for c in args.channel]
    # read the waveform
    with uproot.open(args.ipt) as ipt:
        eventIds = ipt["Readout/TriggerNo"].array(library='np')
        waveforms = ipt["Readout/Waveform"].array(library='np')
        channelIds = ipt["Readout/ChannelId"].array(library='np')
        entries = waveforms.shape[0]
    # read the analysis result
    info = []
    with h5py.File(args.result, 'r') as ipt:
        for j in range(len(args.channel)):
            info.append(ipt['ch{}'.format(args.channel[j])][:])
        trigger = ipt['trigger'][:]
    # read the threshold
    with h5py.File(args.summary, 'r') as sum_ipt:
        peakCs = sum_ipt['res']['peakC']

    # initialize the storerage
    storedtype = [('EventID', '<i4'), ('t', '<f4'), ('Q', '<f4')]
    ## suppose the ratio of trigger is 0.1 and average pulse number not exceed 10.
    pulse = np.zeros((entries, len(args.channel)), dtype=storedtype)
    nums = np.zeros((len(args.channel)), dtype=int)
    for i, (wave, eid, ch) in enumerate(zip(waveforms, eventIds, channelIds)):
        wave = wave.reshape((ch.shape[0],-1))
        chmap = Series(range(ch.shape[0]), index=ch)
        waveformLength = wave.shape[1]
        for j in range(len(args.channel)):
            w = wave[chmap.loc[channels[j]]]
            anar = info[j][i]
            ## 判定有没有触发,判定条件为Ampitude,Charge, FWHM
            if not anar['isTrigger'] or anar['minPeakCharge']< 0.25 * peakCs[j] or anar['FWHM']<5:
                continue
            baseline, std = anar['baseline'], anar['std']
            triggerPulseT = int(trigger[i]['triggerTime'] + anar['begin10'])
            threshold = np.max([args.nsigma * std, 3])
            # 避免在delay1B的位置出现奇怪的异常峰，~将搜寻范围扩大至delayB-speend~,直接在delayB中调整数值为anadelay1B
            start = triggerPulseT + config.anadelay1B
            ## 检查后脉冲
            if np.max(baseline - w[start:]) > threshold:
                intervals = getIntervals(np.arange(start, waveformLength), baseline - w[start:], threshold, spestart, speend)
                for interval in intervals:
                    t, Q = getTQ(np.arange(interval[0], interval[1]), baseline - w[interval[0]:interval[1]], [])
                    # store the relative time ot begin10
                    pulse[nums[j],j] = (eid, t - triggerPulseT, Q)
                    nums[j] += 1
            ## 检查前脉冲
            end = triggerPulseT - config.anapromptE
            if np.max(baseline - w[:end]) > threshold:
                intervals = getIntervals(np.arange(end), baseline - w[:end], threshold, spestart, speend)
                for interval in intervals:
                    t, Q = getTQ(np.arange(interval[0], interval[1]), baseline - w[interval[0]:interval[1]], [])
                    # store the relative time ot begin10
                    pulse[nums[j],j] = (eid, t - triggerPulseT, Q)
                    nums[j] += 1
    totalNums = np.zeros(len(args.channel))
    for j in range(len(args.channel)):
        totalNums[j] = np.sum(info[j]['isTrigger'] & (info[j]['minPeakCharge']> 0.25 * peakCs[j]) & (info[j]['FWHM']>5))
    with h5py.File(args.opt, 'w') as opt:
        for j in range(len(args.channel)):
            opt.create_dataset('ch{}'.format(args.channel[j]), data=pulse[:nums[j], j], compression='gzip')
        opt.create_dataset('totalNum' , data=totalNums, compression='gzip')
