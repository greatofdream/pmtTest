'''
根据trigger分析结果，选择trigger的波形，分析前后脉冲的时间和电荷分布
'''
#!/usr/bin/python3
import uproot, numpy as np, h5py
from pandas import Series
from waveana.util import getIntervals, mergeIntervals, getTQ
from waveana.waveana import Qb, Qe
import argparse
import matplotlib.pyplot as plt
import config
promptB, promptE = config.promptB, config.promptE
delay1B, delay1E = config.delay1B, config.delay1E
delay10B, delay10E = config.delay10B, config.delay10E
spelength, spestart, speend = config.spelength, config.baselength, config.afterlength

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
        peakTs = sum_ipt['res']['TT']

    # initialize the storerage
    storedtype = [('EventID', '<i4'), ('t', '<f4'), ('Q', '<f4'), ('adjustQ', '<f4'), ('peak', '<f4'), ('begin10', '<f4'), ('begin50', '<f4'), ('begin90', '<f4'), ('down10', '<f4'), ('down50', '<f4'), ('down90', '<f4'), ('integrateL', '<f4'), ('integrateR', '<f4'), ('isTrigger', bool), ('mainBegin10', '<f4'), ('minPeakCharge', '<f4'), ('FWHM', '<f4'), ('peakC', '<f4'), ('peakT', '<f4'), ('isAfter', bool)]
    ## suppose the ratio of trigger is 0.1 and average pulse number not exceed 10.
    pulse = np.zeros((entries*2, len(args.channel)), dtype=storedtype)
    nums = np.zeros((len(args.channel)), dtype=int)
    for i, (wave, eid, ch) in enumerate(zip(waveforms, eventIds, channelIds)):
        wave = wave.reshape((ch.shape[0],-1))
        chmap = Series(range(ch.shape[0]), index=ch)
        waveformLength = wave.shape[1]
        for j in range(len(args.channel)):
            w = wave[chmap.loc[channels[j]]]
            anar = info[j][i]
            ## 判定有没有触发,判定条件为Ampitude,Charge, FWHM
            # if not anar['isTrigger'] or anar['minPeakCharge']< 0.25 * peakCs[j] or anar['FWHM']<2:
            #     continue
            baseline, std = anar['baseline'], anar['std']
            triggerExpectT = int(trigger[i]['triggerTime'] + peakTs[j])
            triggerPulseT = int(trigger[i]['triggerTime'] + anar['begin10'])
            if anar['FWHM'] == 0 and anar['riseTime']==0:
                # 确定没有脉冲
                triggerPulseT = triggerExpectT
            threshold = np.max([args.nsigma * std, 3])
            # 避免在delay1B的位置出现奇怪的异常峰，~将搜寻范围扩大至delayB-speend~,直接在delayB中调整数值为anadelay1B
            start = triggerPulseT + config.anadelay1B
            ## 检查后脉冲
            if np.max(baseline - w[start:]) > threshold:
                intervals = getIntervals(np.arange(start, waveformLength), baseline - w[start:], threshold, int(spestart/2), spestart)
                newIntervals, time_paras = mergeIntervals(np.arange(0, waveformLength), w-baseline, intervals, threshold)
                if len(newIntervals)>0:
                    interval_limits = np.append(np.array([int(inter[0]) for inter in newIntervals[1:]], dtype=int), int(newIntervals[-1][1])+speend)
                else:
                    interval_limits = []
                for interval, interval_limit, time_p in zip(intervals, interval_limits, time_paras):
                    t, pv = time_p[0], time_p[1]
                    if pv>threshold:
                        up10, up50, up90 = time_p[2], time_p[3], time_p[4]
                        down10, down50, down90 = time_p[5], time_p[6], time_p[7]
                        # 调整Q积分区间
                        integrateL = max(triggerPulseT+speend, min(t-spestart, int(up10)))
                        integrateR = min(interval_limit, max(t+speend, int(down10)+1))
                        Q = np.sum(baseline-w[integrateL:integrateR])
                        adjustQ = np.sum(baseline-w[int(up10):(int(down10)+1)])
                        # store the relative time ot begin10
                        pulse[nums[j],j] = (eid, t - triggerPulseT, Q, adjustQ, pv, up10 - triggerPulseT, up50 - triggerPulseT, up90 - triggerPulseT, down10 - triggerPulseT, down50 - triggerPulseT, down90 - triggerPulseT, integrateL, integrateR, anar['isTrigger'], anar['begin10'], anar['minPeakCharge'], anar['FWHM'], peakCs[j], peakTs[j], True)
                        nums[j] += 1
            ## 检查前脉冲
            end = triggerExpectT - config.anapromptE + 100
            if np.max(baseline - w[:end]) > threshold:
                # intervals = getIntervals(np.arange(end), baseline - w[:end], threshold, spestart, speend)
                intervals = getIntervals(np.arange(end), baseline - w[:end], threshold, int(spestart/2), spestart)
                newIntervals, time_paras = mergeIntervals(np.arange(0, waveformLength), w-baseline, intervals, threshold)
                if len(newIntervals)>0:
                    interval_limits = np.append(np.array([int(inter[0]) for inter in newIntervals[1:]], dtype=int), int(newIntervals[-1][1])+speend)
                else:
                    interval_limits = []
                for interval, interval_limit, time_p in zip(intervals, interval_limits, time_paras):
                    t, pv = time_p[0], time_p[1]
                    # remove the wrong region
                    if pv>threshold:
                        up10, up50, up90 = time_p[2], time_p[3], time_p[4]
                        down10, down50, down90 = time_p[5], time_p[6], time_p[7]
                        # 调整Q积分区间
                        integrateL = max(0, min(t-spestart,int(up10)))
                        integrateR = min(interval_limit, max(t+speend, int(down10)+1))
                        Q = np.sum(baseline-w[integrateL:integrateR])
                        adjustQ = np.sum(baseline-w[int(up10):(int(down10)+1)])
                        # if up10 < triggerPulseT:
                        # store the relative time ot begin10
                        pulse[nums[j],j] = (eid, t - triggerExpectT, Q, adjustQ, pv, up10 - triggerExpectT, up50 - triggerExpectT, up90 - triggerExpectT, down10 - triggerExpectT, down50 - triggerExpectT, down90 - triggerExpectT, integrateL, integrateR, anar['isTrigger'], anar['begin10'], anar['minPeakCharge'], anar['FWHM'], peakCs[j], peakTs[j], False)
                        nums[j] += 1
    totalNums = np.zeros(len(args.channel), dtype=[('HitNum', '<i4'), ('TrigNum', '<i4')])
    for j in range(len(args.channel)):
        totalNums[j] = (
            np.sum(info[j]['isTrigger'] & (info[j]['minPeakCharge']> 0.25 * peakCs[j]) & (info[j]['FWHM']>2)),
            info[j].shape[0]
        )
    with h5py.File(args.opt, 'w') as opt:
        for j in range(len(args.channel)):
            opt.create_dataset('ch{}'.format(args.channel[j]), data=pulse[:nums[j], j], compression='gzip')
        opt.create_dataset('totalNum' , data=totalNums, compression='gzip')
