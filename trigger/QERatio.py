import numpy as np, h5py
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input h5 file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    psr.add_argument('--interval', help='the time interval for each channel')
    psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
    args = psr.parse_args()
    with h5py.File(args.interval, 'r') as ipt:
        interval = ipt['interval'][:]
        rinterval = ipt['rinterval'][:]
    info = []
    with h5py.File(args.ipt, 'r') as ipt:
        for j in range(len(args.channel)):
            info.append(ipt['ch{}'.format(args.channel[j])][:])
    QEinfo = np.zeros((len(args.channel),), dtype=[('ch', '<i2'), ('triggernum', '<i4'), ('totalnum', '<i4'), ('ratio', '<f4')])
    
    for i in range(len(args.channel)):
        QEinfo[i]['ch'] = args.channel[i]
        QEinfo[i]['totalnum'] = info[i].shape[0]
        QEinfo[i]['triggernum'] = np.sum(info[i]['isTrigger'])
        QEinfo[i]['ratio'] = QEinfo[i]['triggernum']/QEinfo[i]['totalnum']
    with h5py.File(args.opt, 'w') as opt:
        opt.create_dataset('QEinfo', data=QEinfo, compression='gzip')
    print(QEinfo)
    with PdfPages(args.opt+'.pdf') as pdf:
        fig, ax = plt.subplots()
        for i in range(len(args.channel)):
            ax.hist(info[i]['minPeak'], range=[0, 50], bins=50, histtype='step', label='ch{}'.format(args.channel[i]))
        ax.set_yscale('log')
        ax.set_xlabel('peakHeight/mV')
        ax.set_ylabel('entries')
        pdf.savefig(fig)

        for i in range(len(args.channel)):
            fig, ax = plt.subplots()
            selectindex = info[i]['isTrigger']
            h=ax.hist2d(info[i]['minPeakPos'][selectindex], info[i]['minPeak'][selectindex],bins=[100,100],range=[rinterval[i][['start', 'end']],[0,50]],cmap=cmap)
            fig.colorbar(h[3], ax=ax)
            ax.set_xlabel('t/ns')
            ax.set_ylabel('peakHeight/mV')
            ax.set_title(str(args.channel[i])+'trigger time - min peak height')
            pdf.savefig()