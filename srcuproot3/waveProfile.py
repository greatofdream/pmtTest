import h5py, argparse, numpy as np, uproot
from pandas import Series
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
def plotWaves(info,l,r,ch,nearMax=20,posl=300, posr=1700,pl=1,pr=3):
    fig, ax = plt.subplots(figsize=(10, 5))
    index = np.where((info['minPeak']>pl)&(info['minPeak']<pr)&(info['minPeakPos']>posl)&(info['minPeakPos']<posr)&(info['minPeakCharge']>l)&(info['minPeakCharge']<r)&(info['nearPosMax']<=nearMax))[0]
    print(index)
    ax.set_title('waveform charge[{},{}] entries:{}'.format(l,r,index.shape[0]))
    for i in index:
        ax.plot(waveforms[i][(ch*waveformLength+(info[i]['minPeakPos']-200)):(ch*waveformLength+(info[i]['minPeakPos']+200))])#-info[i]['baseline'])
    ax.set_xlabel('t/ns')
    ax.set_ylabel('peakHeight/mV')
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file dir')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-r', dest='root', nargs='+', help='input root files')
psr.add_argument('--refer', dest="refer", help='reference for peak')
psr.add_argument('--tl', dest='thresholdL', type=int, default=10, help='threshold left')
psr.add_argument('--tr', dest='thresholdR', type=int, default=30, help='threshold right')

args = psr.parse_args()
info=[]
length = 200
with h5py.File(args.refer, 'r') as ipt:
    thresholdL = ipt['res']['peakC']-5
    thresholdR = ipt['res']['peakC']+5
with h5py.File(args.ipt, 'r') as ipt:
    #waveformLength = 1500
    for j in range(len(args.channel)):
        info.append(ipt['ch{}'.format(args.channel[j])][:])

ch = [int(i) for i in args.channel]
# set the figure appearance
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
pdf = PdfPages(args.opt+'.pdf')
'''
fig, ax = plt.subplots()
ax.set_title('peakHeight {}-{}'.format(ch[0],ch[1]))
h = ax.hist2d(info[ch[0]]['minPeak'],info[ch[1]]['minPeak'],range=[[0,1000],[0,1000]], bins=[1000, 1000], cmax=100,cmap=cmap)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('ch{}/mV'.format(ch[0]))
ax.set_ylabel('ch{}/mV'.format(ch[1]))
plt.savefig('{}/peakCorrelation.png'.format(args.opt))
plt.close()

fig, ax = plt.subplots()
ax.set_title('charge {}-{}'.format(ch[0],ch[1]))
h = ax.hist2d(info[ch[0]]['minPeakCharge'],info[ch[1]]['minPeakCharge'],range=[[0,2000],[0,2000]], bins=[2000, 2000], cmax=100, cmap=cmap)
fig.colorbar(h[3], ax=ax)
ax.set_xlabel('ch{}/mVns'.format(ch[0]))
ax.set_ylabel('ch{}/mVns'.format(ch[1]))
plt.savefig('{}/chargeCorrelation.png'.format(args.opt))
plt.close()
'''

waveforms = uproot.concatenate([i+':Readout' for i in args.root], filter_name='Waveform',library='np')['Waveform']

with uproot.open(args.root[0]) as ipt:
    channelIds = ipt["Readout/ChannelId"].array(library='np')
nchannel = len(channelIds[0])
waveformLength = int(len(waveforms[0])/nchannel)
chmap = Series(range(channelIds[0].shape[0]), index=channelIds[0])
#eid = uproot.lazyarray(args.root,"Readout","TriggerNo")
#print(np.array(eid[index]))
#exit(0)
indexs = []
storeWave = np.zeros((len(args.channel),length*2))
for j in range(len(args.channel)):
    # indexTF = info[j]['minPeak']>threshold
    indexTF = (info[j]['minPeakCharge']>thresholdL[j])&(info[j]['minPeakCharge']<thresholdR[j])&(info[j]['minPeakCharge']>30)#避开全为噪声部分
    index = np.where(indexTF)[0]
    print(np.sum(indexTF))
 
    fig, ax = plt.subplots()
    ax.set_title('ch{}, entries:{}'.format(ch[j],index.shape[0]))
    if(index.shape[0]>0):
        for i in index:
            pos0 = info[j][i]['minPeakPos']
            trig = int(info[j][i]['begin5mV'])
            baseline = info[j][i]['baseline']
            pos = int(pos0)
            begin = pos - length
            end = pos + length
            if begin<0:
                begin = 0
            if end>waveformLength:
                end = waveformLength
            trigBegin = trig-length
            trigEnd = trig+length
            if trigBegin<0:
                trigBegin = 0
            if trigEnd>waveformLength:
                trigEnd = waveformLength
                if (trigEnd-trigBegin)>2*length:
                    trigEnd = trigBegin+2*length
            wave = waveforms[i].reshape((nchannel,-1))
            storeWave[j,(length+trigBegin-trig):(length+trigEnd-trig)] += wave[chmap.loc[ch[j]]][trigBegin:trigEnd]-baseline
            ax.plot(range(begin-pos+length,(end-pos+length)),wave[chmap.loc[ch[j]]][begin:end],rasterized=True)
    ax.set_xlabel('t/ns')
    ax.set_ylabel('V/mV')
    # plt.savefig('{}/profile{}tr{}.png'.format(args.opt,ch[j], threshold))
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title('ch{}, {}<charge<{}mVns'.format(ch[j],thresholdL[j],thresholdR[j]))
    ax.plot(storeWave[j]/index.shape[0])
    ax.set_xlabel('t/ns')
    ax.set_ylabel('peakHeight/mV')
    # plt.savefig('{}/profileSum{}tr{}-{}.png'.format(args.opt,ch[j], thresholdL, thresholdR))
    pdf.savefig(fig)
    plt.close()

pdf.close()
with h5py.File(args.opt,'w') as opt:
    opt.create_dataset('spe', data=storeWave,compression='gzip')