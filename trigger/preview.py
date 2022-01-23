import uproot, numpy as np, h5py
import argparse
from pandas import Series
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input root file')
psr.add_argument('-o', dest='opt', help='output pdf')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-s', dest='nsigma', default=5, type=float, help='nsimga')
psr.add_argument('-n', dest='nchannel', type=int, default=8, help='number of channel')
psr.add_argument('-l', dest='waveCut', type=int, default=0, help='cut of wave length')
psr.add_argument('-e', dest='eventid', type=int, default=-1, help='eventid')
psr.add_argument('-t', dest='trigger', default=1, help="channel id of trigger")
args = psr.parse_args()

chs = [int(i) for i in args.channel]
with uproot.open(args.ipt) as ipt:
    waveforms = ipt["Readout/Waveform"].array(library='np')
    eventid = ipt["Readout/TriggerNo"].array(library='np')
    channelIds = ipt["Readout/ChannelId"].array(library='np')
eventNum = waveforms.shape[0]
chmap = Series(range(channelIds[0].shape[0]), index = channelIds[0])
chindex = chmap.loc[chs]
assert(1)
nch = len(channelIds[0])
if args.eventid<0:
    with PdfPages('{}'.format(args.opt)) as pdf:
        for eid in range(0,np.min([100,eventNum])):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title('eid {}'.format(eventid[eid]))
            ax.set_xlabel('t/ns')
            ax.set_ylabel('Amplitude/mV')
            ax.plot(waveforms[eid].reshape((nch,-1))[chindex, :].T, label=['ch{}'.format(i) for i in chs])
            axins = inset_axes(ax, width="40%", height="30%",loc='lower left',
                   bbox_to_anchor=(0.2, 0.1, 1, 2),
                   bbox_transform=ax.transAxes)
            axins.plot(waveforms[eid].reshape((nch,-1))[chindex,:1000].T,label=['ch{}'.format(i) for i in chs])
            axins.set_ylim([900,1000])
            mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec='k', lw=1)
            ax.legend()
            pdf.savefig()
            plt.close(fig)


