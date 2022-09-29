#!/usr/bin/python3
import uproot, numpy as np, h5py
from pandas import Series
from waveana.waveana import Qb, Qe
from waveana.triggerana import Triggerana
import argparse
import config
import matplotlib.pyplot as plt

promptB = 250
promptE = 50
delay1B = 300
delay1E = 1000
delay10B = 1000
delay10E = 10000
def Pulse(wave, base, std, up10, down10, promptB,promptE,delay1B,delay1E,delay10B,delay10E):
    before = wave[max(0,up10-promptB):(up10-promptE)]
    after_1 = wave[(down10+delay1B):min(down10+delay1E,wave.shape[0])]
    after_10 = wave[(down10+delay10B):min(down10+delay10E,wave.shape[0])]
    before_pulse = -1
    after_1_pulse = -1
    after_10_pulse = -1
    before_pulse_H,after_1_pulse_H, after_10_pulse_H = 0, 0, 0
    threshold = 5*std
    if threshold<3:
        threshold = 3
    if min(before)<=(base-threshold):
        before_pulse = np.argmin(before)+max(0,up10-promptB)
        before_pulse_H = -np.min(before)
    if min(after_1)<=(base-threshold):
        after_1_pulse = np.argmin(after_1)+(down10+delay1B)
        after_1_pulse_H = -np.min(after_1)
    if min(after_10)<=(base-threshold):
        after_10_pulse = np.argmin(after_10)+(down10+delay10B)
        after_10_pulse_H = -np.min(after_10)
    return before_pulse, after_1_pulse, after_10_pulse, before_pulse_H,after_1_pulse_H, after_10_pulse_H
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input root file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
    psr.add_argument('-t', dest='trigger', default=0, type=int, help='trigger channel')
    psr.add_argument('-s', dest='nsigma', default=5, type=float, help='nsimga')
    psr.add_argument('-n', dest='nchannel', type=int, default=8, help='number of channel')
    psr.add_argument('--interval', help='the time interval for each channel')
    psr.add_argument('--result', help='result of preanalysis')
    psr.add_argument('--QE', default=False, action='store_true')
    args = psr.parse_args()

    triggerch = args.trigger
    with uproot.open(args.ipt) as ipt:
        eventIds = ipt["Readout/TriggerNo"].array(library='np')
        waveforms = ipt["Readout/Waveform"].array(library='np')
        channelIds = ipt["Readout/ChannelId"].array(library='np')

    entries = waveforms.shape[0]
    channels = [int(c) for c in args.channel]
    # read the time interval
    with h5py.File(args.interval, 'r') as ipt:
        interval = ipt['interval'][:]
        if args.QE:
            rinterval = ipt['rinterval'][:]
    # read the analysis result
    info = []
    with h5py.File(args.result, 'r') as ipt:
        waveformLength = ipt.attrs['waveformLength']
        for j in range(len(args.channel)):
            info.append(ipt['ch{}'.format(args.channel[j])][:])
        trigger = ipt['trigger'][:]
    if not args.QE:
        # 计算前后脉冲:initial the result
        pulse = []
        storedtype = [('EventID', '<i4'), ('isTrigger', bool), ('promptPulse','<f4'), ('DelayPulse1','<f4'), ('DelayPulse10','<f4'), ('promptPulseH','<f4'), ('DelayPulse1H','<f4'), ('DelayPulse10H','<f4'), ('up10', '<f4'), ('down10', '<f4')]
        for i in range(len(args.channel)):
            pulse.append(np.zeros((entries,),dtype=storedtype))

        for i, (wave, eid, ch) in enumerate(zip(waveforms, eventIds, channelIds)):
            wave = wave.reshape((ch.shape[0],-1))
            chmap = Series(range(ch.shape[0]), index=ch)
            for j in range(len(args.channel)):
                w = wave[chmap.loc[channels[j]]]
                anar = info[j][i]
                baseline = anar['baseline']
                std = anar['std']
                threshold = args.nsigma*std
                if threshold<3:
                    threshold = 3
                interval_j = [int(ti) for ti in interval[j][['start', 'end']]]
                promptPulse, delayPulse1, delayPulse10 = -1, -1, -1
                promptPulseH, delayPulse1H, delayPulse10H = 0, 0, 0
                # Baseline inteval_j
                if np.max(baseline - w[interval_j[0]:interval_j[1]])>threshold:
                    minIndex = interval_j[0] + np.argmin(w[interval_j[0]:interval_j[1]])
                    # rising edge 0.1 down edge 0.1
                    up10, _, _ = Qb(w-baseline, minIndex, 0)
                    down10, _, _ = Qe(w-baseline, minIndex, 0)
                    # TODO:baseline-3*std
                    # TODO:0.3-1us. 1-10us.
                    #check if pos_histmax_'ch{}'.format(args.channel[j]) is right.it means an int number 
                    promptPulse, delayPulse1, delayPulse10, promptPulseH, delayPulse1H, delayPulse10H = Pulse(w-baseline, 0, std, int(up10), int(down10), promptB,promptE,delay1B,delay1E,delay10B,delay10E)
                    pulse[j][i] = (eid, True, promptPulse, delayPulse1, delayPulse10, promptPulseH, delayPulse1H, delayPulse10H, up10, down10)
                else:
                    pulse[j][i] = (eid, False, promptPulse, delayPulse1, delayPulse10, promptPulseH, delayPulse1H, delayPulse10H, 0, 0)
    else:
        # 计算
        waveana = Triggerana()
        pulse = []
        storedtype = [('EventID', '<i4'), ('isTrigger', bool), ('baseline','<f4'), ('std','<f4'), ('riseTime','<f4'), ('downTime','<f4'),('FWHM','<f4'),
            ('begin10', '<f4'), ('down10', '<f4'), ('begin50', '<f4'), ('down50', '<f4'), ('begin90', '<f4'), ('down90', '<f4'),
            ('minPeak', '<f4'), ('minPeakPos', '<i4'), ('minPeakCharge', '<f4'), ('begin5mV','<f4'), ('end5mV','<f4'), ('nearPosMax','<f4')]
        for i in range(len(args.channel)):
            pulse.append(np.zeros((entries,), dtype=storedtype))
        for i, (wave, eid, ch) in enumerate(zip(waveforms, eventIds, channelIds)):
            wave = wave.reshape((ch.shape[0],-1))
            chmap = Series(range(ch.shape[0]), index=ch)
            for j in range(len(args.channel)):
                w = wave[chmap.loc[channels[j]]]
                # 检查之前预分析的baseline结果是否对应
                anar = info[j][i]
                # 时间区间前后沿往前和后推risetime长度（近似为积分前沿），保证begintime均匀性，后面统计时按照精确的区间
                interval_j = [int(rinterval[j]['start']+trigger[i]['triggerTime']-config.spestart), int(rinterval[j]['end']+trigger[i]['triggerTime']+config.spestart)]
                ## 左右延长范围3ns, r_min相对于选择时间窗，rminIndex相对于激光上升沿
                r_min = np.argmin(w[(interval_j[0]-3):(interval_j[1]+3)]) - 3
                rminIndex = interval_j[0] + r_min - int(trigger[i]['triggerTime'])
                waveana.setWave(wave[chmap.loc[channels[j]]])
                if anar['begin10']>(interval_j[0]-3) and anar['begin10']<(interval_j[1]+3):
                    # 使用之前分析结果加速
                    baseline = anar['baseline']
                    std = anar['std']
                    waveana.minPeakBaseline = baseline
                else:
                    waveana.getBaselineFine(rminIndex + int(trigger[i]['triggerTime']))
                    baseline, std = waveana.minPeakBaseline, waveana.minPeakStd
                threshold = args.nsigma*std
                if threshold<3:
                    threshold = 3
                # Baseline inteval_j
                if np.max(baseline - w[interval_j[0]:interval_j[1]])>threshold:
                    isTrigger = True
                else:
                    isTrigger = False
                # 判断最小值是否在区间内，或baseline下方，如果不是，说明这部分是过冲信号
                if r_min<=0 or r_min>=(interval_j[1]-interval_j[0]) or w[rminIndex + int(trigger[i]['triggerTime'])]>=baseline:
                    isTrigger = False
                    pulse[j][i] = (eid, isTrigger, baseline, std, 0, 0, 0,
                        0, 0, 0, 0, 0, 0,
                        baseline - min(w[interval_j[0]:interval_j[1]]), rminIndex, np.sum(baseline - w[interval_j[0]:interval_j[1]]),
                        anar['begin5mV'], anar['end5mV'], anar['nearPosMax'])
                else:
                    # rising edge 0.1 down edge 0.1
                    waveana.integrateMinPeak(rminIndex + int(trigger[i]['triggerTime']), config.baselength, config.afterlength)
                    up10, up50, up90 = waveana.begin10, waveana.begin50, waveana.begin90
                    down10, down50, down90 = waveana.end10, waveana.end50, waveana.end90
                    ## TT在此处扣除trigger的时间;**此处修正积分区间**
                    pulse[j][i] = (eid, isTrigger, baseline, std, up90-up10, down10-down90, down50-up50, up10-trigger[i]['triggerTime'], down10-trigger[i]['triggerTime'], up50-trigger[i]['triggerTime'], down50-trigger[i]['triggerTime'], up90-trigger[i]['triggerTime'], down90-trigger[i]['triggerTime'],
                                baseline - min(w[interval_j[0]:interval_j[1]]), rminIndex, waveana.minPeakCharge
                                    , anar['begin5mV'], anar['end5mV'], anar['nearPosMax'])
                
    with h5py.File(args.opt, 'w') as opt:
        opt.attrs['waveformLength'] = waveformLength
        for j in range(len(args.channel)):
            opt.create_dataset('ch{}'.format(args.channel[j]), data=pulse[j], compression='gzip')
        opt.create_dataset('trigger', data=trigger, compression='gzip')