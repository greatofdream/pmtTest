'''
used Weighted-Least-Square(WLS) for calculate the PDE
N: Test pmt number and run number. Unknown variable: 2N+(N-1)=3N-1. Total equation: 4N
测试720 721 722 723
python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o ExPMT/QE.h5 --runs 720 721 722 723 --ref CR365
'''
import statsmodels.api as sm
from patsy import dmatrices
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
plt.style.use('./journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import argparse
from csvDatabase import OriginINFO
import config
def loadh5(f, n):
    # 从meta数据库中读取配置
    metainfo = OriginINFO(config.databaseDir + '/{}.csv'.format(n))
    pmts, splitter = metainfo.csv['PMT'], metainfo.csv['BOXID']
    # splitter为分束器编号，n指示run编号
    with h5py.File(f, 'r') as ipt:
        ratesDF = pd.DataFrame(ipt['res'][:][['Channel', 'TriggerRate', 'TotalNum']])
        ratesDF['pmt'] = pmts
        ratesDF['splitter'] = splitter
        ratesDF['runno'] = n
        # 加入variance
        ratesDF['sigma2'] = (1 - ratesDF['TriggerRate']) * ratesDF['TriggerRate'] / ratesDF['TotalNum']
        ratesDF['logsigma2'] = (1 - ratesDF['TriggerRate']) / ratesDF['TriggerRate'] / ratesDF['TotalNum']
    return ratesDF
def merge(runs, iptdir, refpmt):
    rates = pd.concat([loadh5(iptdir.format(run), run) for run in runs])
    # 为PMT进行编号，将参考管编号置为0
    pmts = np.unique(rates['pmt'])
    refindex = np.where(pmts==refpmt)[0]
    pmtsID = np.arange(len(pmts))
    pmtsID[refindex], pmtsID[0] = 0, refindex
    pmtmap = pd.Series(pmtsID, index = pmts)
    rates['pmtno'] = pmtmap.loc[rates['pmt']].values
    return rates, pmtmap
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input h5 file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    psr.add_argument('--runs', nargs='+')
    psr.add_argument('--ref', help='reference PMT')
    args = psr.parse_args()
    measuredRates, pmtmap = merge(args.runs, args.ipt, args.ref)
    pmts = pmtmap.index
    print('finish merge')
    # 对数化
    measuredRates['logR'] = np.log(measuredRates['TriggerRate'].values)
    weights = 1 / measuredRates['logsigma2'].values
    # design matrix
    y, X = dmatrices('logR ~ 0 + C(runno, Treatment)+ C(splitter, Treatment) + C(pmtno, Treatment)', measuredRates)
    mod = sm.WLS(y, X, weights=weights)
    res = mod.fit()
    print(res.summary())
    print('esitimate sigma^2 {:.3f}'.format(res.mse_resid))
    # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
    for r in zip(pmts[1:], np.exp(res.params[-(len(pmts)-1):]), res.bse[-(len(pmts)-1):], res.bse[-(len(pmts)-1):]*np.exp(res.params[-(len(pmts)-1):])):
        print(r[0], r[1], r[2], r[3])
    # print(res.outlier_test())
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
