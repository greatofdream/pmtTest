'''
绘制PDE分布示意图
'''
# 添加模块路径
import sys
sys.path.append('..')
import h5py, numpy as np, pandas as pd
import argparse
import matplotlib.pyplot as plt
plt.style.use('../journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from csvDatabase import TESTINFO
import config
psr = argparse.ArgumentParser()
psr.add_argument('--res', nargs='+', help='analyze hdf5 results')
psr.add_argument('-o', dest='opt', help='output figure file')
psr.add_argument('--ref', default='CR365', help='reference pmt')
args = psr.parse_args()
refpmt = args.ref
testinfo = TESTINFO(config.databaseDir + '/TESTINFO.csv')

with PdfPages(args.opt) as pdf:
    # without correlation
    fig1, ax1 = plt.subplots()
    # with correlation
    fig2, ax2 = plt.subplots()
    for res in args.res:
        with h5py.File(res, 'r') as ipt:
            ratio = ipt['splitter'][:]
            pdes = ipt['QE'][:]
            measuredRates = pd.DataFrame(ipt['measuredRates'][:])
            I_t = ipt['I'][:]
            refRates = ipt['refRates'][:]
        run = int(res.split('/')[-1].split('_')[0])
        pmts = np.unique(testinfo.csv.groupby('RUNNO').get_group(run)['PMT'].values)
        refindex = np.where(pmts==refpmt)[0][0]
        pmtsID = np.arange(len(pmts))
        if refindex!=0:
            pmtsID[refindex] = 0
            pmtsID[0:refindex] = pmtsID[0:refindex] + 1
        pmtmap = pd.Series(pmts, index = pmtsID)
        groupRates = measuredRates.groupby('pmtno')
        refDf = groupRates.get_group(0).sort_values('splitter')
        for i in range(1, len(pmts)):
            # Coreelation PDE
            tmpDf = groupRates.get_group(i).sort_values('splitter')
            ax1.plot(tmpDf['splitter'], tmpDf['TriggerRate_adj'].values/refRates[tmpDf['splitter'].values]*I_t[0], marker='o', label=pmtmap.loc[i])
            # Direct PDE
            x_splitter = []
            y_pde = []
            for j, i_s in enumerate(tmpDf['splitter'].values):
                i_f = np.argwhere(i_s == refDf['splitter'].values)
                if len(i_f)>0:
                    x_splitter.append(tmpDf.iloc[j]['splitter'])
                    y_pde.append(tmpDf.iloc[j]['TriggerRate']/refDf.iloc[i_f[0]]['TriggerRate'])
            ax2.plot(x_splitter, y_pde, marker='o', label=pmtmap.loc[i])
    ax1.set_xlabel('splitter')
    ax2.set_xlabel('splitter')
    ax1.set_ylabel('Adjusted Relative PDE')
    ax2.set_ylabel('Direct Relative PDE')
    ax1.legend()
    ax2.legend()
    ax2.set_yscale('log')
    pdf.savefig(fig1)
    pdf.savefig(fig2)
