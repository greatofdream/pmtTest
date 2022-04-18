import  numpy as np, h5py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import argparse
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
args = psr.parse_args()
info = []
ch2pmt,ch3pmt = args.channel
with h5py.File(args.ipt, 'r') as ipt:
    waveformLength = ipt.attrs['waveformLength']
    #waveformLength = 1500
    for j in range(len(args.channel)):
        info.append(ipt['ch{}'.format(args.channel[j])][:])
def peakLinearPos(infos, chs, l,r,  nearMax=3, rangel=3,ranger=1000,peakPos=[]):
    fig, ax = plt.subplots(dpi=150,figsize=(8,6))
    xminorLocator = MultipleLocator(10)
    yminorLocator = MultipleLocator(10)
    for info,ch in zip(infos,chs):
        h = ax.hist(info['minPeak'][(info['minPeakPos']>=l)&(info['minPeakPos']<r)&(info['nearPosMax']<=nearMax)],bins=(ranger-rangel), range=[rangel,ranger], histtype='step',label='ch{}'.format(ch))
    for p in peakPos:
        print(h[0][p])
        ax.axvline(p,color='k',alpha=0.2)
    ax.set_title('peak pos[{},{}] range[{},{}] nearMax<={}'.format(l,r,rangel,ranger,nearMax))
    ax.set_xlabel('peak/mV')
    ax.set_ylabel('entries')
    ax.set_ylim([0,1000])
    #ax.set_yscale('log')
    ax.xaxis.set_minor_locator(xminorLocator)
    #ax.yaxis.set_minor_locator(yminorLocator)
    ax.legend()
    return h, fig
def peakPos(infos, chs, l,r,  nearMax=3, rangel=3,ranger=1000,peakPos=[]):
    fig, ax = plt.subplots(dpi=150,figsize=(8,6))
    xminorLocator = MultipleLocator(10)
    yminorLocator = MultipleLocator(10)
    h = [[]]*2
    for i,(info,ch) in enumerate(zip(infos,chs)):
        h[i] = ax.hist(info['minPeak'][(info['minPeakPos']>=l)&(info['minPeakPos']<r)&(info['nearPosMax']<=nearMax)],bins=(ranger-rangel), range=[rangel,ranger], histtype='step',label='ch{}'.format(ch))
        cut = findCut(h[i][0])
        print(cut, np.sum(info['minPeak']>cut)/info.shape[0]/waveformLength*1e6)#kHz
        ax.axvline(cut,color='r',linestyle='--')
    for p in peakPos:
        print(h[0][p])
        ax.axvline(p,color='k',alpha=0.2)
    ax.set_title('peak pos[{},{}] range[{},{}] nearMax<={}'.format(l,r,rangel,ranger,nearMax))
    ax.set_xlabel('peak/mV')
    ax.set_ylabel('entries')
    #ax.set_ylim([0,1000])
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(xminorLocator)
    #ax.yaxis.set_minor_locator(yminorLocator)
    ax.legend()
    return h,fig
def findCut(counts):
    # index = np.argmin(counts[2:10])
    i = 0
    for i in range(2,10):
        if counts[i+1]<counts[i]:
            continue
        else:
            break
    # return i
    return 3
pdf = PdfPages(args.opt)
h, fig = peakPos([info[0],info[1]],[ch2pmt,ch3pmt],0,10000,rangel=0,ranger=50,nearMax=10)
pdf.savefig(fig)
h, fig = peakLinearPos([info[0],info[1]],[ch2pmt,ch3pmt],0,10000,rangel=0,ranger=50,nearMax=10)
pdf.savefig(fig)
pdf.close()