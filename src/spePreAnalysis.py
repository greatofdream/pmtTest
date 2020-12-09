import uproot, numpy as np, h5py
import argparse
from analysis import integrateMinPeakWave, integrateWave
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', nargs='+', help='input root file')
psr.add_argument('-o', dest='opt', help='output h5 file')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-s', dest='nsigma', default=5, type=float, help='nsimga')
psr.add_argument('-n', dest='nchannel', type=int, default=8, help='number of channel')
psr.add_argument('-l', dest='waveCut', type=int, default=2002, help='cut of wave length')
args = psr.parse_args()

storedtype = [('allCharge','<f4'),('minPeakCharge','<f4'),('minPeak','<f4')]# ,('peakCharge','<f4'),('peak','<f4')
waveforms = uproot.lazyarray(args.ipt, "Readout","Waveform", basketcache=uproot.cache.ThreadSafeArrayCache("30 MB"))
channelId = uproot.lazyarray(args.ipt, "Readout","ChannelId")
sec = uproot.lazyarray(args.ipt, "Readout", "Sec")
nanosec = uproot.lazyarray(args.ipt, "Readout", "NanoSec")
baselength = 50

entries = waveforms.shape[0]
waveformsLength = int(waveforms[0].shape[0]/args.nchannel)
info = []
for i in range(len(args.channel)):
    info.append(np.zeros((entries,),dtype=storedtype))
for i, (wave, ch, sc, nsc) in enumerate(zip(waveforms, channelId, sec, nanosec)):
    for j in range(len(args.channel)):
        waveAC = integrateWave(wave[(j*waveformsLength):((j+1)*waveformsLength)][:args.waveCut])
        waveMinPC, minP = integrateMinPeakWave(wave[(j*waveformsLength):((j+1)*waveformsLength)][:args.waveCut], baselength)
        info[j][i] = (waveAC, waveMinPC, minP)
with h5py.File(args.opt, 'w') as opt:
    for j in range(len(args.channel)):
        opt.create_dataset('ch{}'.format(args.channel[j]), data=info[j], compression='gzip')

