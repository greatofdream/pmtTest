'''
该文件计算激光触发后脉冲对应比例
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
from mergeH5 import h5Merger
promptB, promptE = config.promptB, config.promptE
delay1B, delay1E = config.delay1B, config.delay1E
delay10B, delay10E = config.delay10B, config.delay10E
spelength, spestart, speend = config.spelength, config.spestart, config.speend

psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', nargs='+', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
psr.add_argument('-c', dest='channel', nargs='+', default=[0, 1], help='channel used in DAQ')
args = psr.parse_args()
reader = h5Merger(args.ipt)
info = reader.read()
totalNums = np.sum(info[-1].reshape((-1,len(args.channel))), axis=0).astype(int)
print(totalNums)
result = np.zeros(len(args.channel), dtype=[('Channel', '<i2'), ('TriggerNum', '<i4'), ('prompt', '<f4'), ('delay1', '<f4'), ('delay10', '<f4')])
resultSigma2 = np.zeros(len(args.channel), dtype=[('Channel', '<i2'), ('TriggerNum', '<i4'), ('prompt', '<f4'), ('delay1', '<f4'), ('delay10', '<f4')])
for j in range(len(args.channel)):
    delay1c = info[j]['EventID'][(info[j]['t']>delay1B) & (info[j]['t']<delay1E)]
    delay10c = info[j]['EventID'][(info[j]['t']>delay10B) & (info[j]['t']<delay10E)]
    # c1 = np.unique(delay1c)
    # c10 = np.unique(delay10c)
    c1, c10 = delay1c, delay10c
    promptc = info[j]['EventID'][(info[j]['t']>-promptB) & (info[j]['t']<-promptE)]
    # c_p = np.unique(promptc)
    c_p = promptc
    result[j] = (args.channel[j], totalNums[j], len(c_p)/totalNums[j], len(c1)/totalNums[j], len(c10)/totalNums[j])
    resultSigma2[j] = (args.channel[j], 0, len(c_p)/totalNums[j]**2, len(c1)/totalNums[j]**2, len(c10)/totalNums[j]**2)
# store the pulse ratio

with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('ratio', data=result, compression='gzip')
    opt.create_dataset('resSigma2', data=resultSigma2, compression='gzip')
    for j in range(len(args.channel)):
        opt.create_dataset('ch{}'.format(args.channel[j]), data=info[j], compression='gzip')
# set the figure appearance
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
with PdfPages(args.opt + '.pdf') as pdf:
    for j in range(len(args.channel)):
        fig, ax = plt.subplots(figsize=(12,6))
        h = ax.hist2d(info[j]['t'], info[j]['Q'], bins=[int((delay10E - delay1B)/50), 50], range=[[delay1B, delay10E], [0, 1000]], cmap=cmap)
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel('Relative t/ns')
        ax.set_ylabel('Equivalent Charge/ADCns')
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        pdf.savefig(fig)

        fig, ax = plt.subplots(figsize=(12,6))
        h = ax.hist(info[j]['t'], bins=int(delay10E/50), range=[0, delay10E], histtype='step')
        ax.set_xlabel('Relative t/ns')
        ax.set_ylabel('Entries')
        ax.set_xlim([0, delay10E])
        ax.set_yscale('log')
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        pdf.savefig(fig)

        fig, ax = plt.subplots()
        h = ax.hist2d(info[j]['t'], info[j]['Q'], bins=[int((delay1B - promptE)/2), 50], range=[[-promptB, delay1B], [0, 1000]], cmap=cmap)
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel('Relative t/ns')
        ax.set_ylabel('Equivalent Charge/ADCns')
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        pdf.savefig(fig)

        fig, ax = plt.subplots()
        h = ax.hist(info[j]['t'], bins=int((delay1B -config.anadelay1B)/2), range=[config.anadelay1B, delay1B], histtype='step')
        h = ax.hist(info[j]['t'], bins=int((promptB - promptE)/2), range=[-promptB, -promptE], histtype='step')
        ax.set_xlabel('Relative t/ns')
        ax.set_ylabel('Entries')
        ax.set_yscale('log')
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        pdf.savefig(fig)