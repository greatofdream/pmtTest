#!/usr/bin/python3
import uproot, numpy as np, h5py
from pandas import Series
import argparse
from triggerAnalysis import waveAna
import tqdm
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input root file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    #psr.add_argument('-k', dest='tpl',help='template')
    psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
    psr.add_argument('-t', dest='trigger', default=0, type=int, help='trigger channel')
    psr.add_argument('-s', dest='nsigma', default=5, type=float, help='nsimga')
    psr.add_argument('-n', dest='nchannel', type=int, default=8, help='number of channel')
    psr.add_argument('-l', dest='waveCut', type=int, default=0, help='cut of wave length')
    args = psr.parse_args()

    triggerch = args.trigger
    storedtype = [('EventID', '<i4'), ('allCharge','<f4'),('minPeakCharge','<f4'),('minPeak','<f4'),('minPeakPos','<i4'),('baseline','<f4'),('std','<f4'),('riseTime','<f4'),('downTime','<f4'),('FWHM','<f4'),('begin10','<f4'),('begin50','<f4'),('begin90','<f4'),('end10','<f4'),('end50','<f4'),('end90','<f4'),('begin5mV','<f4'),('end5mV','<f4'),('nearPosMax','<f4'),('nearPosMean', '<f4'),('nearPosStd','<f4'),('fitTime','<f4')]# ,('peakCharge','<f4'),('peak','<f4')
    with uproot.open(args.ipt) as ipt:
        eventIds = ipt["Readout/TriggerNo"].array(library='np')
        waveforms = ipt["Readout/Waveform"].array(library='np')
        channelIds = ipt["Readout/ChannelId"].array(library='np')
    baselength = 50

    entries = waveforms.shape[0]
    channels = [int(c) for c in args.channel]
    waveformsLength = int(waveforms[0].shape[0]/args.nchannel)
    info = []
    if args.waveCut==0:
        waveCut = waveformsLength
        print('wave length {}'.format(waveCut))
    else:
        waveCut = args.waveCut
    with h5py.File('Ex1/411/400ns/tpl.h5','r') as ipt:
        tpl = ipt['tpl'][:]
    waveana = waveAna()
    waveana.tpl = np.array(tpl)
    for i in range(len(args.channel)):
        info.append(np.zeros((entries,),dtype=storedtype))
    triggerInfo = np.zeros((entries,), dtype=[('EventID','<i4'),('triggerTime','<f4')])
    print('begin loop')
    # channelid在采数的过程中没有发生变化，而且所有通道均被采集，其实长度一致
    for i, (wave, eid, ch) in enumerate(zip(waveforms, eventIds, channelIds)):
        wave = wave.reshape((ch.shape[0],-1))
        chmap = Series(range(ch.shape[0]), index=ch)
        waveana.setTriggerWave(wave[chmap.loc[triggerch]][:waveCut])

        for j in range(len(args.channel)):
            waveana.setWave(wave[chmap.loc[channels[j]]][:waveCut])
            waveana.getBaselineFine(int(waveana.triggerTime)+200, 5, 250,100)
            waveana.integrateWave()
            waveana.integrateMinPeakWave()
            fittime = 0
            # if waveana.minPeak>5 and waveana.minIndex>220 and waveana.minIndex<280 and waveana.nearMax<2:
            #     fitresult = waveana.fit()
            #     if fitresult.success:
            #         print('fit eid {}'.format(eid))
            #         fittime = fitresult.x[0]+int(waveana.begin10)-50
            info[j][i] = (eid, waveana.allCharge, waveana.minPeakCharge, waveana.minPeak, waveana.minIndex, waveana.minPeakBaseline, waveana.minPeakStd,waveana.begin90-waveana.begin10,waveana.end10-waveana.end90,waveana.end50-waveana.begin50,waveana.begin10,waveana.begin50,waveana.begin90,waveana.end10,waveana.end50,waveana.end90,waveana.begin5mV,waveana.end5mV,waveana.nearMax,waveana.nearPositiveMean,waveana.nearPositiveStd,fittime)
        triggerInfo[i] = (eid, waveana.triggerTime)
    with h5py.File(args.opt, 'w') as opt:
        opt.attrs['waveformLength'] = waveCut
        opt.attrs['triggerChannel'] = args.trigger
        for j in range(len(args.channel)):
            opt.create_dataset('ch{}'.format(args.channel[j]), data=info[j], compression='gzip')
        opt.create_dataset('trigger', data=triggerInfo, compression='gzip')

