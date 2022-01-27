#!/usr/bin/python3
import uproot, numpy as np, h5py
from pandas import Series
from triggerAnalysis import Qb, Qe
import argparse
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
    after_2 = wave[(down10+delay10B):min(down10+delay10E,wave.shape[0])]
    before_pulse = -1
    after_1_pulse = -1
    after_2_pulse = -1
    threshold = 5*std
    if threshold<3:
        threshold = 3
    if min(before)<=(base-threshold):
        before_pulse = np.argmin(before)+max(0,up10-promptB)
    if min(after_1)<=(base-threshold):
        after_1_pulse = np.argmin(after_1)+(down10+delay1B)
    if min(after_2)<=(base-threshold):
        after_2_pulse = np.argmin(after_2)+(down10+delay10B)
    return before_pulse,after_1_pulse,after_2_pulse
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
        rinterval = ipt['rinterval'][:]
    # read the analysis result
    info = []
    with h5py.File(args.result, 'r') as ipt:
        waveformLength = ipt.attrs['waveformLength']
        for j in range(len(args.channel)):
            info.append(ipt['ch{}'.format(args.channel[j])][:])
        trigger = ipt['trigger'][:]
    if not args.QE:
        # inital the result
        pulse = []
        storedtype = [('EventID', '<i4'), ('isTrigger', bool), ('promptPulse','<f4'), ('DelayPulse1','<f4'),('DelayPulse10','<f4'), ('up10', '<f4'), ('down10', '<f4')]
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
                # Baseline inteval_j
                if np.max(baseline - w[interval_j[0]:interval_j[1]])>threshold:
                    minIndex = interval_j[0] + np.argmin(w[interval_j[0]:interval_j[1]])
                    # rising edge 0.1 down edge 0.1
                    up10, _, _ = Qb(w-baseline, minIndex, 0)
                    down10, _, _ = Qe(w-baseline, minIndex, 0)
                    # TODO:baseline-3*std
                    # TODO:0.3-1us. 1-10us.
                    #check if pos_histmax_'ch{}'.format(args.channel[j]) is right.it means an int number 
                    promptPulse, delayPulse1, delayPulse10 = Pulse(w-baseline, 0, std, int(up10), int(down10), promptB,promptE,delay1B,delay1E,delay10B,delay10E)
                    pulse[j][i] = (eid, True, promptPulse, delayPulse1, delayPulse10, up10, down10)
                else:
                    pulse[j][i] = (eid, False, promptPulse, delayPulse1, delayPulse10, 0, 0)
    else:
        pulse = []
        storedtype = [('EventID', '<i4'), ('isTrigger', bool), ('up10', '<f4'), ('down10', '<f4'), ('minPeak', '<f4'), ('minPeakPos', '<i4'), ('charge', '<f4')]
        for i in range(len(args.channel)):
            pulse.append(np.zeros((entries,), dtype=storedtype))
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
                interval_j = [int(ti+trigger[i]['triggerTime']) for ti in rinterval[j][['start', 'end']]]
                # Baseline inteval_j
                if np.max(baseline - w[interval_j[0]:interval_j[1]])>threshold:
                    isTrigger = True
                    rminIndex = interval_j[0] + np.argmin(w[interval_j[0]:interval_j[1]]) - int(trigger[i]['triggerTime'])
                    # rising edge 0.1 down edge 0.1
                    up10, _, _ = Qb(w-baseline, rminIndex + int(trigger[i]['triggerTime']), 0)
                    down10, _, _ = Qe(w-baseline, rminIndex + int(trigger[i]['triggerTime']), 0)
                    pulse[j][i] = (eid, isTrigger, up10, down10, baseline - min(w[interval_j[0]:interval_j[1]]), rminIndex, np.sum(baseline - w[int(up10):int(down10)]))
                else:
                    isTrigger = False
                    rminIndex = interval_j[0] + np.argmin(w[interval_j[0]:interval_j[1]]) - int(trigger[i]['triggerTime'])
                    pulse[j][i] = (eid, isTrigger, -1, -1, baseline - min(w[interval_j[0]:interval_j[1]]), rminIndex, 0)
    with h5py.File(args.opt, 'w') as opt:
        for j in range(len(args.channel)):
            opt.create_dataset('ch{}'.format(args.channel[j]), data=pulse[j], compression='gzip')
