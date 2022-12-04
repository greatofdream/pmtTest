'''
Calculate the laser pulse and elastic pulse
'''
import matplotlib.pyplot as plt
plt.style.use('./journal.mplstyle')
import h5py, argparse
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import config
from waveana.util import peakResidual, vallyResidual, fitGausB, centralMoment, RootFit
import ROOT
dtype = [
        ('Channel', '<i2'),
        ('laserA', '<f4'), ('laserT', '<f4'), ('laserSigma', '<f4'),
        ('mainA', '<f4'), ('mainT', '<f4'), ('mainSigma', '<f4'),
        ('elasticA', '<f4'), ('elasticT', '<f4'), ('elasticSigma', '<f4'),
    ]
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('--interval', help='the time interval for each channel')
args = psr.parse_args()
NChannels = len(args.channel)
info = []
with h5py.File(args.ipt, 'r') as ipt:
    waveformLength = ipt.attrs['waveformLength']
    for j in range(NChannels):
        info.append(ipt['ch{}'.format(args.channel[j])][:])
    trigger = ipt['trigger'][:]
N = info[0].shape[0]
with h5py.File(args.interval, 'r') as ipt:
    rinterval = ipt['rinterval'][:]
intervalCenters = [int(ti) for ti in rinterval['mean']]
results = np.zeros(NChannels, dtype=dtype)
paraSigma2 = np.zeros(NChannels, dtype=dtype)
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
print('begin plot')
pdf = PdfPages(args.opt+'.pdf')
for j in range(NChannels):
    start, end = intervalCenters[j] + config.laserB -10, intervalCenters[j] + config.elasticE
    select0 = info[j][:,0]['isTrigger']
    select1 = info[j][:,1]['isTrigger']
    select2 = info[j][:,2]['isTrigger']
    print('Num:{},{},{}'.format(np.sum(select0), np.sum(select1), np.sum(select2)))
    print('laser and main:{}'.format(np.sum(select0&select1)))
    print('elastic and main:{}'.format(np.sum(select2&select1)))
    # Time distribution
    fig, ax = plt.subplots()
    ax.hist(info[j][:,0]['begin10'][select0] - trigger['triggerTime'][select0], range=[start, end], bins=end-start, histtype='step', color='r')
    ax.hist(info[j][:,1]['begin10'][select1] - trigger['triggerTime'][select1], range=[start, end], bins=end-start, histtype='step', color='g')
    ax.hist(info[j][:,2]['begin10'][select2] - trigger['triggerTime'][select2], range=[start, end], bins=end-start, histtype='step', color='b')
    ax.hist(info[j][:,0]['begin10'][select0&select1] - trigger['triggerTime'][select0&select1], range=[start, end], bins=end-start, histtype='step', color='r')
    # ax.hist(info[j][:,1]['begin10'][select0&select1] - trigger['triggerTime'][select0&select1], range=[start, end], bins=end-start, histtype='step', color='g', label='laser & main')
    ax.hist(info[j][:,2]['begin10'][select2&select1] - trigger['triggerTime'][select2&select1], range=[start, end], bins=end-start, histtype='step', color='b')
    # ax.hist(info[j][:,1]['begin10'][select2&select1] - trigger['triggerTime'][select2&select1], range=[start, end], bins=end-start, histtype='step', color='g', label='elastic & main')
    ax.axhline(N*1e-5, ls='--', label='10kHz')
    ax.set_xlim([start, end])
    ax.set_yscale('log')
    ax.set_xlabel('TT/ns')
    ax.legend()
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    pdf.savefig(fig)
    plt.close()
    
    fig, ax = plt.subplots()
    ax.hist(info[j][:,0]['begin10'][select0&select1] - info[j][:,1]['begin10'][select0&select1], range=(config.laserB, config.laserE), bins=config.laserE-config.laserB, histtype='step')
    ax.hist(info[j][:,2]['begin10'][select1&select2] - info[j][:,1]['begin10'][select1&select2], range=(config.elasticB, config.elasticE), bins=config.elasticE-config.elasticB, histtype='step')
    ax.set_xlabel('TT/ns')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    pdf.savefig(fig)
    plt.close()
    # Time-peakHeight distribution
    fig, ax = plt.subplots()
    h = ax.hist2d(np.concatenate([info[j][:,0]['begin10'][select0] - trigger['triggerTime'][select0], info[j][:,1]['begin10'] - trigger['triggerTime'], info[j][:,2]['begin10'] - trigger['triggerTime']]), np.concatenate([info[j][:,0]['minPeak'][select0], info[j][:,1]['minPeak'], info[j][:,2]['minPeak']]), range=[[start, end], [0, 20]], bins=[end-start, 20], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('TT/ns')
    ax.set_ylabel('$V_p$/mV')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    pdf.savefig(fig)
    plt.close()

    # Time-charge distribution
    fig, ax = plt.subplots()
    h = ax.hist2d(np.concatenate([info[j][:,0]['begin10'][select0] - trigger['triggerTime'][select0], info[j][:,1]['begin10'] - trigger['triggerTime'], info[j][:,2]['begin10'] - trigger['triggerTime']]), np.concatenate([info[j][:,0]['minPeakCharge'][select0], info[j][:,1]['minPeakCharge'], info[j][:,2]['minPeakCharge']]), range=[[start, end], [0, 100]], bins=[end-start, 200], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('TT/ns')
    ax.set_ylabel('Equvilent charge/mV$\cdot$ns')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    pdf.savefig(fig)
    plt.close()
pdf.close()
# with h5py.File(args.opt, 'w') as opt:
#     opt.create_dataset('res',data=results, compression='gzip')
#     opt.create_dataset('resSigma2', data=paraSigma2, compression='gzip')