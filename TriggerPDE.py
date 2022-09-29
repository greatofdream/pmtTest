'''
used Weighted-Least-Square(WLS) for calculate the PDE
N: Test pmt number and run number. Unknown variable: 2N+(N-1)=3N-1. Total equation: 4N
测试720 721 722 723
python3 TriggerPDE.py -i ExResult/{}/600ns/chargeSelect.h5 -o ExPMT/QE.h5 --runs 720 721 722 723 --ref CR365
'''
import statsmodels.api as sm
from patsy import dmatrices, dmatrix
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sympy import rotations
plt.style.use('./journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import argparse
from csvDatabase import OriginINFO, PMTINFO
import config
def loadh5(f, n):
    # 从meta数据库中读取配置
    metainfo = OriginINFO(config.databaseDir + '/{}.csv'.format(n))
    pmts, splitter = metainfo.csv['PMT'], metainfo.csv['BOXID']
    # splitter为分束器编号，n指示run编号
    with h5py.File(f, 'r') as ipt:
        ratesDF = pd.DataFrame(ipt['res'][:][['Channel', 'TriggerRate', 'TotalNum']])
        ratesDF['TriggerNum'] = ratesDF['TriggerRate'] * ratesDF['TotalNum']
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
    psr.add_argument('--glm', default=False, action='store_true')
    args = psr.parse_args()
    measuredRates, pmtmap = merge(args.runs, args.ipt, args.ref)
    pmts = pmtmap.index
    index = pmtmap.values
    pmtmap = pd.Series(pmts, index=index)
    testpmts = pmtmap.loc[np.arange(1, len(pmts))]
    print('finish merge')
    if not args.glm:
        # 对数化
        measuredRates['logR'] = np.log(measuredRates['TriggerRate'].values)
        weights = 1 / measuredRates['logsigma2'].values
        # design matrix
        y, X = dmatrices('logR ~ 0 + C(runno, Treatment)+ C(splitter, Treatment) + C(pmtno, Treatment)', measuredRates)
        # mod = sm.WLS(y, X, weights=weights)
        mod = sm.OLS(y, X)
        res = mod.fit()
        print(res.summary())
        print('esitimate sigma^2 {:.3f}'.format(res.mse_resid))
    else:
        # CLogLog transform link function. 
        # https://www.statsmodels.org/dev/generated/statsmodels.genmod.families.links.cloglog.html#statsmodels.genmod.families.links.cloglog
        # design matrix
        measuredRates['NoTriggerNum'] = measuredRates['TotalNum'] - measuredRates['TriggerNum']
        response = measuredRates[['TriggerNum', 'NoTriggerNum']]
        mod = sm.GLM.from_formula('response ~ 0 + C(runno, Treatment)+ C(splitter, Treatment) + C(pmtno, Treatment)', family=sm.families.Binomial(sm.genmod.families.links.cloglog()), data=measuredRates)
        res = mod.fit()
        print(res.summary())
    # https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
    for r in zip(testpmts, np.exp(res.params[-(len(pmts)-1):]), res.bse[-(len(pmts)-1):], res.bse[-(len(pmts)-1):]*np.exp(res.params[-(len(pmts)-1):])):
        print(r[0], r[1], r[2], r[3])
    # print(res.outlier_test())
    num_run = np.unique(measuredRates['runno'].values).shape[0]
    ## Get the reference value from vendor
    pmtinfo = PMTINFO(config.databaseDir + '/PMTINFO.csv')
    PDE_r = pmtinfo.getPMTInfo(testpmts)['PDE_r']
    print('reference:{}'.format(PDE_r))
    PDE_t = np.exp(res.params[-(len(pmts)-1):])
    sigma_PDE_t = res.bse[-(len(pmts)-1):]*np.exp(res.params[-(len(pmts)-1):])
    I_t = np.exp(res.params[:num_run])
    with h5py.File(args.opt, 'w') as opt:
        opt.create_dataset('QE', data=PDE_t, compression='gzip')
        opt.create_dataset('logerr', data=res.bse[-(len(pmts)-1):], compression='gzip')
        opt.create_dataset('err', data=sigma_PDE_t, compression='gzip')
        opt.create_dataset('I', data=I_t, compression='gzip')
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

        fig, ax = plt.subplots()
        yerr = PDE_t/PDE_t[0]*np.sqrt((sigma_PDE_t/PDE_t)**2+(sigma_PDE_t[0]/PDE_t[0])**2)
        yerr[0] = 0
        ax.errorbar(testpmts, PDE_t/PDE_t[0], yerr=yerr, label='Test')
        ax.scatter(testpmts, PDE_r/PDE_r[0], color='g', label='Vendor')
        ax.set_ylabel('relative PDE')
        ax.legend()
        pdf.savefig(fig)

        I_total = np.zeros(measuredRates.shape[0])
        for i, run in enumerate(args.runs):
            I_total[measuredRates['runno'].values==run] = I_t[i]
        
        # 为了避免测试管没有完全经过所有位置，refRates使用预测值
        # refRates = groupRates.get_group(0).sort_values('splitter')
        
        measuredRates['I_t'] = I_total
        measuredRates['TriggerRate_adj'] = measuredRates['TriggerRate'] / I_total
        groupRates = measuredRates.groupby('pmtno')
        
        if not args.glm:
            predictRates = np.exp(res.predict())
            measuredRates['PredictRate'] = predictRates
            refPredictor = pd.DataFrame({
                'runno': np.zeros(num_splitter),
                'splitter': np.arange(num_splitter),
                'pmtno': np.zeros(num_splitter)
            })
            runnolevel = np.arange(num_run)
            pmtnolevel = np.arange(num_splitter)
            refPredictorX = dmatrix("0 + C(runno, Treatment, levels=runnolevel)+ C(splitter, Treatment) + C(pmtno, Treatment,levels=pmtnolevel)", refPredictor)
            refRates = np.exp(res.predict(refPredictorX))
        else:
            predictRates = res.predict()
            measuredRates['PredictRate'] = predictRates
            # Pearson Chisquare:
            # print(np.sum(
            #     (measuredRates['TriggerNum']-measuredRates['PredictRate']*measuredRates['TotalNum'])**2/(measuredRates['PredictRate']*measuredRates['TotalNum']*(1-measuredRates['PredictRate']))
            #     ))
            refPredictor = pd.DataFrame({
                'runno': np.repeat([args.runs[0]], num_splitter),
                'splitter': np.arange(num_splitter),
                'pmtno': np.zeros(num_splitter)
            })
            refRates = res.predict(refPredictor)
        print(refRates)
        # fig, ax = plt.subplots()
        # for i in range(1, len(pmts)):
        #     tmpDf = groupRates.get_group(i).sort_values('splitter')
        #     ax.plot(tmpDf['splitter'], tmpDf['TriggerRate'].values/refRates[tmpDf['splitter']]/tmpDf['I_t'], label=pmtmap.loc[i])
        # ax.set_xlabel('splitter')
        # ax.set_ylabel('Relative PDE')
        # ax.legend()
        # pdf.savefig(fig)

        fig, ax = plt.subplots()
        for i in range(1, len(pmts)):
            tmpDf = groupRates.get_group(i).sort_values('splitter')
            ax.plot(tmpDf['splitter'], tmpDf['TriggerRate_adj'].values/refRates[tmpDf['splitter'].values]*I_t[0], label=pmtmap.loc[i])
        ax.set_xlabel('splitter')
        ax.set_ylabel('Adjusted Relative PDE')
        ax.legend()
        pdf.savefig(fig)

        fig, ax = plt.subplots()
        xlabels = [i+'-'+str(j)+'-'+str(k) for i,j,k in zip(measuredRates['pmt'].values, measuredRates['splitter'].values, measuredRates['runno'].values)]
        ax.scatter(xlabels, predictRates, label='predict')
        ax.scatter(xlabels, measuredRates['TriggerRate'].values, label='observe')
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_xlabel('pmt-splitter-run')
        ax.set_ylabel('Relative rate')
        ax.legend()
        pdf.savefig(fig)

        fig, ax = plt.subplots()
        ax.scatter(xlabels, predictRates*measuredRates['TotalNum'], label='predict')
        ax.errorbar(xlabels, measuredRates['TriggerNum'].values, yerr=np.sqrt(predictRates*measuredRates['TotalNum']*(1-predictRates)), fmt='.', color='orange', label='observe')
        ax.set_xticklabels(xlabels, rotation=45)
        ax.set_xlabel('pmt-splitter-run')
        ax.set_ylabel('Entries')
        ax.legend()
        pdf.savefig(fig)


    for r in zip(np.exp(res.params[num_run:(num_run+num_splitter-1)]), np.exp(res.params[num_run:(num_run+num_splitter-1)])*res.bse[num_run:(num_run+num_splitter-1)]):
        print('{:.2f}+-{:.2f}'.format(r[0], r[1]))
