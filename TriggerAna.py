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
def Pulse(wave, start, end, threshold, intLimit):
    minIndex = start + np.argmin(wave[start:end])
    minpeak = -wave[minIndex]
    integral_b = minIndex-config.baselength
    integral_e = min((minIndex+config.afterlength), intLimit)
    minpeakcharge = -np.sum(wave[integral_b:integral_e])
    isTrigger = True
    if minpeak>threshold:
        # rising edge 0.1 down edge 0.1
        up10, up50, up90 = Qb(wave, minIndex, 0)
        if np.min(wave[int(up10):minIndex]) < -minpeak:
            # The minindex is the falling edge, omit
            isTrigger = False
            up10, up50, up90 = minIndex, minIndex, minIndex
            down10, down50, down90 = minIndex, minIndex, minIndex
        else:
            down10, down50, down90 = Qe(wave, minIndex, 0)
    else:
        isTrigger = False
        up10, up50, up90 = minIndex, minIndex, minIndex
        down10, down50, down90 = minIndex, minIndex, minIndex
    return isTrigger, up10, up50, up90, down10, down50, down90, minpeak, minIndex, minpeakcharge
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
    psr.add_argument('--QE', default=False, action='store_true', help='consider the laser pulse and late pulse')
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
        storedtype = [('EventID', '<i4'), ('isTrigger', bool), ('begin10', '<f4'), ('down10', '<f4'), ('begin50', '<f4'), ('down50', '<f4'), ('begin90', '<f4'), ('down90', '<f4'),
        ('minPeak', '<f4'), ('minPeakPos', '<i4'), ('minPeakCharge', '<f4')]
        for i in range(len(args.channel)):
            # 1,2,3 columns are laser, main, late pulse. [laserB, laserE], [laserE, -laserE], [elasticB, elasticE]
            pulse.append(np.zeros((entries, 3),dtype=storedtype))
        intervalCenters = rinterval['mean']
        for i, (wave, eid, ch) in enumerate(zip(waveforms, eventIds, channelIds)):
            wave = wave.reshape((ch.shape[0],-1))
            chmap = Series(range(ch.shape[0]), index=ch)
            triggerTime = trigger['triggerTime'][i]
            for j in range(len(args.channel)):
                w = wave[chmap.loc[channels[j]]]
                anar = info[j][i]
                baseline = anar['baseline']
                std = anar['std']
                threshold = args.nsigma*std
                if threshold<3:
                    threshold = 3
                # late pulse
                pulse[j][i, 2] = (eid, *Pulse(w-baseline, int(triggerTime + intervalCenters[j] + config.elasticB), int(triggerTime + intervalCenters[j] + config.elasticE), threshold, w.shape[0]))
                # main pulse
                if pulse[j][i, 2]['isTrigger']:
                    intLimit = pulse[j][i, 2]['minPeakPos']
                else:
                    intLimit = w.shape[0]
                pulse[j][i, 1] = (eid, *Pulse(w-baseline, int(triggerTime + intervalCenters[j] + config.laserE/2), int(triggerTime + intervalCenters[j] - config.laserE/2), threshold, intLimit))
                # laser pulse
                if pulse[j][i, 1]['isTrigger']:
                    intLimit = pulse[j][i, 1]['minPeakPos']
                pulse[j][i, 0] = (eid, *Pulse(w-baseline, int(triggerTime + intervalCenters[j] + config.laserB), int(triggerTime + intervalCenters[j] + config.laserE), threshold, intLimit))
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
                # calculate inteval for each waveform
                interval_j = [int(rinterval[j]['start']+trigger[i]['triggerTime']), int(rinterval[j]['end']+trigger[i]['triggerTime'])]
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