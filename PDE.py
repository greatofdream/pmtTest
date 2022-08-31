import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy import dmatrices
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import pandas as pd
import h5py
import argparse
'''
used Ordinary-Least-Square(OLS) for calculate the QE
N: Test pmt number and run number. Unknown variable: 2N+(N-1)=3N-1. Total equation: 4N
测试761-764
python3 PDE.py -i ExResult/{}/600ns/qe/ratio.h5 -o testQE2.h5 --merge --runs 761 762 763 764 --pmts PM2112-2010 PM2112-9089F PM2107-9029F HAMAMATSU --pos 1 4 2 3
python3 PDE.py -i ExResult/{}/600ns/qe/ratio.h5 -o testQE1.h5 --merge --runs 720 721 722 723 --pmts PM2107-9017F PM2107-9012 PM2106-9145 HAMAMATSU --pos 1 4 2 3
python3 PDE.py -i ExResult/{}/600ns/qe/ratio.h5 -o testQE3.h5 --merge --runs 768 769 --pmts PM2106-9103F PM2112-2002 PM2112-9041F PM2112-2005 --pos 1 4 2 3
# 合并多测测试
python3 PDE.py -i ExResult/{}/600ns/qe/ratio.h5 -o testQE.h5 --mergerun --runs testQE1.h5 testQE2.h5 testQE3.h5
# GLM分析
python3 PDE.py -i testQE.h5 -o QEResult.h5
'''
def loadh5(f, pmts, pos, i):
    # 此处channel跟随pmt位置，激光位置编号不动
    # pos 为
    with h5py.File(f, 'r') as ipt:
        ratesDF = pd.DataFrame(ipt['QEinfo'][:])
        ratesDF['pmt'] = pmts
        ratesDF['pmt'] = ratesDF['pmt'].astype('|S20')
        chmap = pd.Series(np.roll(np.sort(pos), -i), index=np.sort(pos))
        ratesDF['splitter'] = chmap.loc[pos].values
        ratesDF['batch'] = i
    return ratesDF
def merge(runs, iptdir, opt, pmts, pos):
    rates = pd.concat([loadh5(iptdir.format(run), pmts, pos, i) for i, run in enumerate(runs)])
    with h5py.File(opt, 'w') as h5opt:
        h5opt.create_dataset('rates', data=rates.to_records(), compression='gzip')
def loadRun(f, runno):
    with h5py.File(f, 'r') as ipt:
        rates = pd.DataFrame(ipt['rates'][:])
    rates['runno'] = runno*4 + rates['batch'].values
    return rates
def mergeRun(files, opt):
    rates = pd.concat([loadRun(f, i) for i, f in enumerate(files)])
    rates = rates.set_index('index')
    pmts = np.unique(rates['pmt'])
    pmtmap = pd.Series(np.arange(len(pmts)), index = pmts.astype('|S20'))
    rates['pmt'] = rates['pmt'].astype('|S20')
    rates['pmtno'] = pmtmap.loc[rates['pmt']].values
    # with h5py.File(opt, 'w') as h5opt:
    h5opt = h5py.File(opt, 'w')
    h5opt.create_dataset('rates', data=rates.to_records(), compression='gzip')
    h5opt.create_dataset('pmtmap', data=pmts, compression='gzip')
    h5opt.close()
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input h5 file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    psr.add_argument('--merge', default=False, action='store_true')
    psr.add_argument('--mergerun', default=False, action='store_true')
    psr.add_argument('--runs', nargs='+')
    psr.add_argument('--pmts', nargs='+')
    psr.add_argument('--pos', nargs='+')
    args = psr.parse_args()
    if args.merge:
        merge(args.runs, args.ipt, args.opt, args.pmts, [int(i) for i in args.pos])
        print('finish merge')
        exit()
    elif args.mergerun:
        mergeRun(args.runs, args.opt)
        print('finish merge runs')
        exit()
    with h5py.File(args.ipt, 'r') as ipt:
        measuredRates = pd.DataFrame(ipt['rates'][:])
        pmts = ipt['pmtmap'][:]
        pmtmap = pd.Series(np.arange(len(pmts)), index = pmts)
    # 对数化
    measuredRates['logR'] = np.log(measuredRates['ratio'].values)
    # design matrix
    y, X = dmatrices('logR ~ 0 + C(runno, Treatment)+ C(splitter, Treatment) + C(pmtno, Treatment)', measuredRates)
    mod = sm.OLS(y, X)
    res = mod.fit()
    print(res.summary())
    print('esitimate sigma^2 {:.3f}'.format(res.mse_resid))
    # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
    for r in zip(pmts[1:], np.exp(res.params[-(len(pmts)-1):]), res.bse[-(len(pmts)-1):], res.bse[-(len(pmts)-1):]*np.exp(res.params[-(len(pmts)-1):])):
        print(str(r[0], 'UTF-8'), r[1], r[2], r[3])
    print(res.outlier_test())
    num_run = np.unique(measuredRates['runno'].values).shape[0]
    with h5py.File(args.opt, 'w') as opt:
        opt.create_dataset('QE', data=np.exp(res.params[-(len(pmts)-1):]), compression='gzip')
        opt.create_dataset('logerr', data=res.bse[-(len(pmts)-1):], compression='gzip')
        opt.create_dataset('err', data=res.bse[-(len(pmts)-1):]*np.exp(res.params[-(len(pmts)-1):]))
        opt.create_dataset('I', data=np.exp(res.params[:num_run]), compression='gzip')
    with PdfPages(args.opt+'.pdf') as pdf:
        fig, ax = plt.subplots()
        ax.errorbar(x=range(num_run), y=np.exp(res.params[:num_run]), yerr=np.exp(res.params[:num_run])*res.bse[:num_run])
        ax.set_xlabel('runno')
        ax.set_ylabel('Intensity(A.U)')
        pdf.savefig(fig)

        fig, ax = plt.subplots()
        num_splitter = np.unique(measuredRates['splitter'].values).shape[0]
        ax.errorbar(x=range(1,num_splitter), y=np.exp(res.params[num_run:(num_run+num_splitter-1)]), yerr=np.exp(res.params[num_run:(num_run+num_splitter-1)])*res.bse[num_run:(num_run+num_splitter-1)])
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.set_xticks(range(1,num_splitter), range(1,num_splitter))
        ax.set_xlabel('splitter id')
        ax.set_ylabel('relative ratio')
        pdf.savefig(fig)
    for r in zip(np.exp(res.params[num_run:(num_run+num_splitter-1)]), np.exp(res.params[num_run:(num_run+num_splitter-1)])*res.bse[num_run:(num_run+num_splitter-1)]):
        print('{:.2f}+-{:.2f}'.format(r[0], r[1]))
