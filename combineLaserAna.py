'''
合并多个Laser Run 结果，并扣除Dark Mode run合并的结果
'''
import argparse
import h5py, pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('./journal.mplstyle')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors
from scipy.optimize import minimize
import config
import ROOT
ADC2mV = config.ADC2mV
promptB, promptE = config.promptB, config.promptE
delay1B, delay1E = config.delay1B, config.delay1E
delay10B, delay10E = config.delay10B, config.delay10E
spelength, spestart, speend = config.spelength, config.spestart, config.speend

def loadh5(f, channel, run):
    with h5py.File(f, 'r') as ipt:
        res = ipt['res'][:]
        resSigma2 = ipt['resSigma2'][:]
    result = pd.DataFrame(res[res['Channel']==channel])
    resultSigma2 = pd.DataFrame(resSigma2[resSigma2['Channel']==channel])
    result['Run'] = run
    resultSigma2['Run'] = run
    return result, resultSigma2
def loadPulse(f, channel):
    with h5py.File(f, 'r') as ipt:
        res = ipt['ch{}'.format(channel)][:]
    df = pd.DataFrame(res)
    mainpulseLargeQ = df.loc[df['t']>config.delay1E].sort_values('Q', ascending=False).groupby('EventID').first().reset_index()
    return df, mainpulseLargeQ
def loadMainPulse(fm, fp, channel):
    with h5py.File(fm, 'r') as ipt:
        mainRes = ipt['ch{}'.format(channel)][:]
    with h5py.File(fp, 'r') as ipt:
        pulseRes = ipt['ch{}'.format(channel)][:]
    with h5py.File(args.summary, 'r') as sum_ipt:
        summary = sum_ipt['res'][:]
    peakC = summary[summary['Channel']==int(channel)]['peakC']
    mainDtype = [('EventID', '<i4'), ('isTrigger', bool), ('mainBegin10', '<f4'), ('minPeakCharge', '<f4'), ('FWHM', '<f4'), ('peakC', '<f4')]
def loadRatio(f, channel, run):
    with h5py.File(f, 'r') as ipt:
        res = ipt['ratio'][:]
        resSigma2 = ipt['resSigma2'][:]
    result = pd.DataFrame(res[res['Channel']==channel])
    resultSigma2 = pd.DataFrame(resSigma2[resSigma2['Channel']==channel])
    result['Run'] = run
    resultSigma2['Run'] = run
    return result, resultSigma2
def loadSER(f, channel, run):
    with h5py.File(f, 'r') as ipt:
        res = ipt['res'][:]
    result = pd.DataFrame(res[res['Channel']==channel])
    result['Run'] = run
    return result
def Afterpulse(x, xs):
    x = x.reshape((-1,3))
    # return np.sum(x[:,0]*np.exp(-(xs[:,None]-x[:,1])**2/2/x[:,2]**2)/x[:,2], axis=-1)
    return np.sum(x[:,0]*np.exp(-(xs[:,None]-x[:,1])**2/2/x[:,2]**2), axis=-1)
def likelihood(x, *args):
    xs, ys = args
    eys = Afterpulse(x, xs)
    return np.sum((ys-eys)**2)
psr = argparse.ArgumentParser()
psr.add_argument('--dir', help='the ana result directory')
psr.add_argument('--config', help='test run no csv')
psr.add_argument('--badrun', help='runs excluded')
psr.add_argument('--dark', help='dark run merge result')
psr.add_argument('-o', dest='opt', help='name of output file')
args = psr.parse_args()
# 读取darkrun 合并结果
try:
    with h5py.File(args.dark, 'r') as ipt:
        darkresult = ipt['merge'][:]
    darkExpect = True
except:
    darkExpect = False

# 统计分析结果路径和对应ch号
configcsv = pd.read_csv(args.config)
PMTName = configcsv['PMT'].values[0]
if PMTName.startswith('PM'):
    boundsSigma = config.boundsMCP
else:
    boundsSigma = config.boundsHama

runs = configcsv[configcsv['MODE']==1].to_records()
badruns = np.unique(np.append(np.loadtxt(args.badrun), np.loadtxt('ExPMT/ExcludeRun.csv')))
selectruns = []
for run in runs:
    if not run['RUNNO'] in badruns:
        selectruns.append(run)
if len(selectruns)==0:
    print('Error: No runs contains laser stage run')
    exit(0)
infos = [loadh5(args.dir.format(run['RUNNO']) + '/chargeSelect.h5', run['CHANNEL'], run['RUNNO']) for run in selectruns]
results = pd.concat([i[0] for i in infos])
resultsSigma2 = pd.concat([i[1] for i in infos])
print(results[['Run', 'Channel', 'TTS','TTS_bin']])
if darkExpect:
    # 注意这里在去除DCR时使用了平均DCR，而不是每次laser stage对应的dark stage结果
    results['TriggerRateWODCR'] = (results['TriggerRate'] - darkresult[0]['DCR'] * results['window'] * 1e-6)/(1 - darkresult[0]['DCR'] * results['window'] * 1e-6)
    resultsSigma2['TriggerRateWODCR'] = results['TriggerRateWODCR']**2 * ((resultsSigma2['TriggerRate'] + darkresult[1]['DCR'] * (results['window'] * 1e-6)**2)/(results['TriggerRate'] - darkresult[0]['DCR'] * results['window'] * 1e-6)**2 + (darkresult[1]['DCR'] * (results['window'] * 1e-6)**2)/(1-darkresult[0]['DCR'] * results['window'] * 1e-6)**2)
else:
    # omit the influence of DCR for trigger pulse
    results['TriggerRateWODCR'] = results['TriggerRate']
    resultsSigma2['TriggerRateWODCR'] = resultsSigma2['TriggerRate']
mergeresultsA = np.zeros((2,), dtype=[
    ('peakC', '<f4'), ('vallyC', '<f4'), ('Gain', '<f4'), ('GainSigma', '<f4'), ('PV', '<f4'),
    ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'),
    ('TTS', '<f4'), ('TTA', '<f4'), ('TTS2', '<f4'), ('TTA2', '<f4'), ('TTS_exp', '<f4'), ('TTA_exp', '<f4'), ('DCR_exp', '<f4'), ('TTS_bin', '<f4'), ('Res', '<f4'), ('chargeRes', '<f4'),
    ('TriggerRate', '<f4'), ('TriggerRateWODCR', '<f4'), ('chargeMu', '<f4'), ('chargeSigma', '<f4'), ('PDE', '<f4'),
    ('TT_kToTT', '<f4'), ('TT_expToTT', '<f4'), ('TT_DCRToTT', '<f4'), ('TT2_1', '<f4'),
    ('A0', '<f4')
])
# PDE测量从TestSummary.csv中获取
pdes = pd.read_csv(config.TestSummaryPath).set_index('PMT').loc[PMTName, 'PDE'].values
pdes_sigma = pd.read_csv(config.TestSummaryPath).set_index('PMT').loc[PMTName, 'PDESigma'].values
if np.sum(~(np.isnan(pdes)|(pdes==0)))>0:
    pdes_sigma = pdes_sigma[~(np.isnan(pdes)|(pdes==0))]
    pdes = pdes[~(np.isnan(pdes)|(pdes==0))]
    pde = np.sum(pdes/pdes_sigma**2)/np.sum(1/pdes_sigma**2)
    pde_sigma2 = 1/np.sum(1/pdes_sigma**2)
else:
    pde = np.nan
    pde_sigma2 = np.nan
# 计算去除dark后的触发率,采用合并测量的方式
## 这里peakC,vallyC等拟合参数合并测量时考虑每次测量的方差
triggerNum = results['TriggerNum']
ress = results['GainSigma'] / results['Gain']
res_weights = ress**2 * (resultsSigma2['GainSigma']/results['GainSigma']**2 + resultsSigma2['Gain']/results['Gain']**2)
chargeress = np.sqrt(results['chargeSigma2']) / results['chargeMu']
chargeress_weights = chargeress**2 * (resultsSigma2['chargeSigma2']/results['chargeSigma2']**2/4 + resultsSigma2['chargeMu']/results['chargeMu']**2)
sig_TTS = results['TTS']/np.sqrt(2 * np.log(2))/2
TT_kToTT = results['TTA2']*results['TTS2']/(results['TTA']*results['TTS'])
TT_expToTT = (results['TTA_exp']*results['TTS_exp'])/(results['TTA']*sig_TTS*np.sqrt(2*np.pi))
TT_DCRToTT = 2*results['DCR_exp'] / (results['TTA']*np.sqrt(2*np.pi))#sig_TTS*2*results['DCR_exp']/(results['TTA']*sig_TTS*np.sqrt(2*np.pi))
deltaTT = results['TT'] - results['TT2']
TT_kToTT_Sigma2 = TT_kToTT**2 * (resultsSigma2['TTA2']/results['TTA2']**2 + resultsSigma2['TTS2']/results['TTS2']**2 + resultsSigma2['TTA']/results['TTA']**2 + resultsSigma2['TTS']/results['TTS']**2)
TT_expToTT_Sigma2 = TT_expToTT**2 * (resultsSigma2['TTA_exp']/results['TTA_exp']**2 + resultsSigma2['TTS_exp']/results['TTS_exp']**2 + resultsSigma2['TTA']/results['TTA']**2 + resultsSigma2['TTS']/results['TTS']**2)
TT_DCRToTT_Sigma2 = TT_DCRToTT**2 * (resultsSigma2['DCR_exp']/results['DCR_exp']**2 + resultsSigma2['TTA']/results['TTA']**2)
deltaTT_Sigma2 = resultsSigma2['TT'] + resultsSigma2['TT2']
# the main peak ratio in charge distribution
A0_Ratios = results['peakV'] * np.sqrt(2*np.pi) * (results['GainSigma']*50*1.6/ADC2mV) / results['TriggerNum']
A0_RatioSigma2s = A0_Ratios**2 * (resultsSigma2['peakV']/results['peakV']**2 + resultsSigma2['GainSigma']/results['GainSigma']**2 + (resultsSigma2['TriggerNum']/results['TriggerNum'])/results['TriggerNum'])#results['TriggerNum'] int32, cannot use operater **2 

mergeresultsA[0] = (
    np.sum(results['peakC']/resultsSigma2['peakC']) / np.sum(1/resultsSigma2['peakC']),
    np.sum(results['vallyC']/resultsSigma2['vallyC']) / np.sum(1/resultsSigma2['vallyC']),
    np.sum(results['Gain']/resultsSigma2['Gain']) / np.sum(1/resultsSigma2['Gain']),
    np.sum(results['GainSigma']/resultsSigma2['GainSigma']) / np.sum(1/resultsSigma2['GainSigma']),
    np.sum(results['PV']/resultsSigma2['PV']) / np.sum(1/resultsSigma2['PV']),
    np.sum(results['Rise']/resultsSigma2['Rise']) / np.sum(1/resultsSigma2['Rise']),
    np.sum(results['Fall']/resultsSigma2['Fall'])/np.sum(1/resultsSigma2['Fall']),
    np.sum(results['TH']/resultsSigma2['TH'])/np.sum(1/resultsSigma2['TH']),
    np.sum(results['FWHM']/resultsSigma2['FWHM'])/np.sum(1/resultsSigma2['FWHM']),
    np.sum(results['TTS']/resultsSigma2['TTS'])/np.sum(1/resultsSigma2['TTS']),
    np.sum(results['TTA']/resultsSigma2['TTA'])/np.sum(1/resultsSigma2['TTA']),
    np.sum(results['TTS2']/resultsSigma2['TTS2'])/np.sum(1/resultsSigma2['TTS2']),
    np.sum(results['TTA2']/resultsSigma2['TTA2'])/np.sum(1/resultsSigma2['TTA2']),
    np.sum(results['TTS_exp']/resultsSigma2['TTS_exp'])/np.sum(1/resultsSigma2['TTS_exp']),
    np.sum(results['TTA_exp']/resultsSigma2['TTA_exp'])/np.sum(1/resultsSigma2['TTA_exp']),
    np.sum(results['DCR_exp']/resultsSigma2['DCR_exp'])/np.sum(1/resultsSigma2['DCR_exp']),
    np.sum(results['TTS_bin']/resultsSigma2['TTS_bin'])/np.sum(1/resultsSigma2['TTS_bin']),
    np.sum(ress/res_weights) / np.sum(1/res_weights),
    np.sum(chargeress / chargeress_weights) / np.sum(1/chargeress_weights),
    np.sum(results['TriggerRate'] / resultsSigma2['TriggerRate'])/np.sum(1/resultsSigma2['TriggerRate']),
    np.sum(results['TriggerRateWODCR'] / resultsSigma2['TriggerRateWODCR'])/np.sum(1 / resultsSigma2['TriggerRateWODCR']),
    np.sum(results['chargeMu'] / resultsSigma2['chargeMu']) / np.sum(1 / resultsSigma2['chargeMu']),
    np.sum(results['chargeSigma2']/resultsSigma2['chargeSigma2']) / np.sum(1/resultsSigma2['chargeSigma2']),
    pde,
    np.sum(TT_kToTT / TT_kToTT_Sigma2) / np.sum(1 / TT_kToTT_Sigma2),
    np.sum(TT_expToTT / TT_expToTT_Sigma2) / np.sum(1 / TT_expToTT_Sigma2),
    np.sum(TT_DCRToTT / TT_DCRToTT_Sigma2) / np.sum(1 / TT_DCRToTT_Sigma2),
    np.sum(deltaTT / deltaTT_Sigma2) / np.sum(1 / deltaTT_Sigma2),
    np.sum(A0_Ratios / A0_RatioSigma2s) / np.sum(1 / A0_RatioSigma2s)
    )
# 误差使用最小二乘法
mergeresultsA[1] = (
    1 / np.sum(1/resultsSigma2['peakC']),
    1 / np.sum(1/resultsSigma2['vallyC']),
    1 / np.sum(1/resultsSigma2['Gain']),
    1 / np.sum(1/resultsSigma2['GainSigma']),
    1 / np.sum(1/resultsSigma2['PV']),
    1 / np.sum(1/resultsSigma2['Rise']),
    1 / np.sum(1/resultsSigma2['Fall']),
    1 / np.sum(1/resultsSigma2['TH']),
    1 / np.sum(1/resultsSigma2['FWHM']),
    1 / np.sum(1/resultsSigma2['TTS']),
    1 / np.sum(1/resultsSigma2['TTA']),
    1 / np.sum(1/resultsSigma2['TTS2']),
    1 / np.sum(1/resultsSigma2['TTA2']),
    1 / np.sum(1/resultsSigma2['TTS_exp']),
    1 / np.sum(1/resultsSigma2['TTA_exp']),
    1 / np.sum(1/resultsSigma2['DCR_exp']),
    1 / np.sum(1/resultsSigma2['TTS_bin']),
    1 / np.sum(1/res_weights),
    1 / np.sum(1/chargeress_weights),
    1 / np.sum(1/resultsSigma2['TriggerRate']),
    1 / np.sum(1 / resultsSigma2['TriggerRateWODCR']),
    1 / np.sum(1/resultsSigma2['chargeMu']),
    1 / np.sum(1/resultsSigma2['chargeSigma2']),
    pde_sigma2,
    1 / np.sum(1 / TT_kToTT_Sigma2),
    1 / np.sum(1 / TT_expToTT_Sigma2),
    1 / np.sum(1 / TT_DCRToTT_Sigma2),
    1 / np.sum(1 / deltaTT_Sigma2),
    1 / np.sum(1 / A0_RatioSigma2s)
    )
print('peak ratio in charge distribution: {:.3f}, sigma {:.3f}'.format(mergeresultsA[0]['A0'], np.sqrt(mergeresultsA[1]['A0'])))
# 统计多批数据的afterpulse
pulseResultsData = [loadPulse(args.dir.format(run['RUNNO']) + '/pulseRatio.h5', run['CHANNEL']) for run in selectruns]
pulseResults_Total = pd.concat([pr[0] for pr in pulseResultsData])
mainpulseResults = pd.concat([pr[1] for pr in pulseResultsData])
# keep the select criterial same as TriggerPulseAna.py
selectEvent = pulseResults_Total['isTrigger'] & (pulseResults_Total['minPeakCharge'] > 0.25 * pulseResults_Total['peakC']) & (pulseResults_Total['FWHM']>2)
selectMainEvent = mainpulseResults['isTrigger'] & (mainpulseResults['minPeakCharge'] > 0.25 * mainpulseResults['peakC']) & (mainpulseResults['FWHM']>2)
pulseResults = pulseResults_Total[selectEvent]
print('Num of mainpulse with afterpulse: {}, afterpulse: {}'.format(np.sum(selectMainEvent), np.sum(pulseResults['begin10']>config.delay1E)))

infos = [loadRatio(args.dir.format(run['RUNNO']) + '/pulseRatio.h5', run['CHANNEL'], run['RUNNO']) for run in selectruns]
pulseratioResults = pd.concat([i[0] for i in infos])
pulseratioResultsSigma2 = pd.concat([i[1] for i in infos])
promptwindow, delay1window, delay10window, DCRwindow = promptB - promptE, delay1E - delay1B, delay10E - delay10B, config.DCRE-config.DCRB
promptAll_index = (pulseResults_Total['begin10']>-promptB) & (pulseResults_Total['begin10']<-promptE)
totalTriggerNum = np.sum(pulseratioResults['TriggerNum'])
totalNum = np.sum(pulseratioResults['TotalNum'])
DCR_total = np.sum(pulseratioResults['DCR']*pulseratioResults['TotalNum'])/np.sum(pulseratioResults['TotalNum'])/DCRwindow
DCR_totalSigma2 = DCR_total/np.sum(pulseratioResults['TotalNum'])/DCRwindow
# use pre pulse ratio to estimate the DCR
pulseratioResults['DCR_laser'] = pulseratioResults['DCR']/DCRwindow*1E6
# **Warning**, variance of DCR is different from delay1
pulseratioResults['promptWODCR'] = pulseratioResults['prompt'] - pulseratioResults['DCR']/DCRwindow * promptwindow
pulseratioResults['delay1WODCR'] = pulseratioResults['delay1'] - pulseratioResults['DCR']/DCRwindow * delay1window
pulseratioResults['delay10WODCR'] = pulseratioResults['delay10'] - pulseratioResults['DCR']/DCRwindow * delay10window
promptwodcrSigma2s = pulseratioResultsSigma2['prompt'] + pulseratioResultsSigma2['DCR'] * (promptwindow/DCRwindow)**2
delay1wodcrSigma2s = pulseratioResultsSigma2['delay1'] + pulseratioResultsSigma2['DCR'] * (delay1window / DCRwindow)**2
delay10wodcrSigma2s = pulseratioResultsSigma2['delay10'] + pulseratioResultsSigma2['DCR'] * (delay10window / DCRwindow)**2
# combine the results using LE
promptwodcr = np.sum(pulseratioResults['promptWODCR']/promptwodcrSigma2s) / np.sum(1/promptwodcrSigma2s)
delay1wodcr = np.sum(pulseratioResults['delay1WODCR']/delay1wodcrSigma2s) / np.sum(1/delay1wodcrSigma2s)
delay10wodcr = np.sum(pulseratioResults['delay10WODCR']/delay10wodcrSigma2s) / np.sum(1/delay10wodcrSigma2s)

# promptAllWODCR use all dataset estimate the prompt, all dataset estimate DCR
# delayAllWODCR use main pulse dataset estimate the delay, all dataset estimate DCR
mergePulseResults = np.zeros((2,), dtype=[('prompt', '<f4'), ('delay1', '<f4'), ('delay10', '<f4'), ('promptWODCR', '<f4'), ('delay1WODCR', '<f4'), ('delay10WODCR', '<f4'), ('DCR', '<f4'), ('meanDCR', '<f4'), ('meanprompt', '<f4'), ('meandelay1', '<f4'), ('meandelay10', '<f4'), ('promptAllWODCR', '<f4'), ('delayAllWODCR', '<f4')])

mergePulseResults[0] = (
    np.sum(pulseratioResults['prompt'] / pulseratioResultsSigma2['prompt'])/np.sum(1 / pulseratioResultsSigma2['prompt']),
    np.sum(pulseratioResults['delay1'] / pulseratioResultsSigma2['delay1'])/np.sum(1 / pulseratioResultsSigma2['delay1']),
    np.sum(pulseratioResults['delay10'] / pulseratioResultsSigma2['delay10'])/np.sum(1 / pulseratioResultsSigma2['delay10']),
    promptwodcr,
    delay1wodcr,
    delay10wodcr,
    np.sum(pulseratioResults['DCR']/pulseratioResultsSigma2['DCR']) / np.sum(1/pulseratioResultsSigma2['DCR'])/DCRwindow*1E6,
    DCR_total*1E6,
    np.sum(pulseratioResults['promptWODCR']*pulseratioResults['TriggerNum'])/np.sum(pulseratioResults['TriggerNum']),
    np.sum(pulseratioResults['delay1WODCR']*pulseratioResults['TriggerNum'])/np.sum(pulseratioResults['TriggerNum']),
    np.sum(pulseratioResults['delay10WODCR']*pulseratioResults['TriggerNum'])/np.sum(pulseratioResults['TriggerNum']),
    np.sum(promptAll_index)/np.sum(pulseratioResults['TotalNum']) - DCR_total * promptwindow,
    np.sum((pulseratioResults['delay1'] + pulseratioResults['delay10']) * pulseratioResults['TriggerNum']) / np.sum(pulseratioResults['TriggerNum']) - DCR_total * (delay1window + delay10window)
)
mergePulseResults[1] = (
    1 / np.sum(1 / pulseratioResultsSigma2['prompt']),
    1 / np.sum(1 / pulseratioResultsSigma2['delay1']),
    1 / np.sum(1 / pulseratioResultsSigma2['delay10']),
    1 / np.sum(1/promptwodcrSigma2s),
    1 / np.sum(1/delay1wodcrSigma2s),
    1 / np.sum(1/delay10wodcrSigma2s),
    1 / np.sum(1/pulseratioResultsSigma2['DCR'])*(1E6/DCRwindow)**2,
    DCR_totalSigma2*(1E6)**2,
    np.sum(promptwodcrSigma2s*pulseratioResults['TriggerNum']*pulseratioResults['TriggerNum'])/np.sum(pulseratioResults['TriggerNum'])**2,
    np.sum(delay1wodcrSigma2s*pulseratioResults['TriggerNum']*pulseratioResults['TriggerNum'])/np.sum(pulseratioResults['TriggerNum'])**2,
    np.sum(delay10wodcrSigma2s*pulseratioResults['TriggerNum']*pulseratioResults['TriggerNum'])/np.sum(pulseratioResults['TriggerNum'])**2,
    np.sum(promptAll_index)/np.sum(pulseratioResults['TotalNum'])**2 + DCR_totalSigma2 * promptwindow**2,
    np.sum((pulseratioResults['delay1'] + pulseratioResults['delay10']) * pulseratioResults['TriggerNum']) / np.sum(pulseratioResults['TriggerNum'])**2 + DCR_totalSigma2 * (delay1window + delay10window)
)
print('Total dataset prepulse ratio: {:.5f}$\pm${:.5f} {},{}'.format(mergePulseResults['promptAllWODCR'][0], np.sqrt(mergePulseResults['promptAllWODCR'][1]), np.sum(promptAll_index),  DCR_total * promptwindow * np.sum(pulseratioResults['TotalNum'])))
print('Mainpulse dataset prepulse ratio: {:.5f}$\pm${:.5f}'.format(mergePulseResults['meanprompt'][0], np.sqrt(mergePulseResults['meanprompt'][1])))
print('Mainpulse dataset afterpulse ratio: {:.5f}$\pm${:.5f}'.format(mergePulseResults['delayAllWODCR'][0], np.sqrt(mergePulseResults['delayAllWODCR'][1])))
# 统计ser参数
## tau, sigma考虑的误差为统计误差，未考虑拟合误差, tau_total, sigma_total未考虑拟合误差
serResults = pd.concat([loadSER(args.dir.format(run['RUNNO']) + '/serMerge.h5', run['CHANNEL'], run['RUNNO']) for run in selectruns])
mergeSERResults = np.zeros((2,), dtype=[('tau', '<f4'), ('sigma', '<f4'), ('tau_total', '<f4'), ('sigma_total', '<f4')])
mergeSERResults[0] = (
    np.sum(serResults['tau']/serResults['tau_sigma']**2)/np.sum(1/serResults['tau_sigma']**2),
    np.sum(serResults['sigma']/serResults['sigma_sigma']**2)/np.sum(1/serResults['sigma_sigma']**2),
    np.mean(serResults['tau_total']),
    np.mean(serResults['sigma_total']),
)
mergeSERResults[1] = (
    1/np.sum(1/serResults['tau_sigma']**2),
    1/np.sum(1/serResults['sigma_sigma']**2),
    np.var(serResults['tau_total']),
    np.var(serResults['sigma_total'])
)
# 绘制变化曲线
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)

with PdfPages(args.opt + '.pdf') as pdf:
    # Trigger Rate变化
    fig, ax = plt.subplots()
    ax.errorbar(results['Run'], results['TriggerRate'], yerr=np.sqrt(resultsSigma2['TriggerRate']), marker='o', label='Trigger Rate')
    if darkExpect:
        ax.errorbar(results['Run'], results['TriggerRateWODCR'], yerr=np.sqrt(resultsSigma2['TriggerRateWODCR']), marker='o', label='Trigger Rate WO Dark Noise')
    ax.set_xlabel('Run')
    ax.set_ylabel('Trigger Rate')
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.errorbar(pulseratioResults['Run'], pulseratioResults['DCR_laser'], yerr=np.sqrt(pulseratioResultsSigma2['DCR']), marker='o', label='DCR')
    ax.axhline(mergePulseResults[0]['DCR'], linewidth=1, linestyle='--', color='r', label='Merge DCR')
    ax.axhline(mergePulseResults[0]['meanDCR'], linewidth=1, linestyle='--', label='Average DCR')
    ax.set_xlabel('Run')
    ax.set_ylabel('DCR')
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.errorbar(results['Run'], results['TTS'], yerr=np.sqrt(resultsSigma2['TTS']), marker='o', label='main TTS')
    ax.errorbar(results['Run'], results['TTS2'], yerr=np.sqrt(resultsSigma2['TTS2']), marker='o', label='pre TTS')
    ax.set_xlabel('Run')
    ax.set_ylabel('TTS')
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(results['Run'], results['DCR_exp'], marker='o')
    ax.set_xlabel('Run')
    ax.set_ylabel('Exp components DCR')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.errorbar(pulseratioResults['Run'], pulseratioResults['prompt'], yerr=np.sqrt(pulseratioResultsSigma2['prompt']), marker='o', label='prompt')
    ax.errorbar(pulseratioResults['Run'], pulseratioResults['delay1'], yerr=np.sqrt(pulseratioResultsSigma2['delay1']), marker='o', label='delay1')
    ax.errorbar(pulseratioResults['Run'], pulseratioResults['delay10'], yerr=np.sqrt(pulseratioResultsSigma2['delay10']), marker='o', label='delay10')
    ax.set_xlabel('Run')
    ax.set_ylabel('rate')
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.errorbar(pulseratioResults['Run'], pulseratioResults['promptWODCR'], yerr=np.sqrt(promptwodcrSigma2s), marker='o', label='prompt')
    ax.errorbar(pulseratioResults['Run'], pulseratioResults['delay1WODCR'], yerr=np.sqrt(delay1wodcrSigma2s), marker='o', label='delay1')
    ax.errorbar(pulseratioResults['Run'], pulseratioResults['delay10WODCR'], yerr=np.sqrt(delay10wodcrSigma2s), marker='o', label='delay10')
    ax.set_xlabel('Run')
    ax.set_ylabel('rate')
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.errorbar(results['Run'], results['Gain'], yerr=np.sqrt(resultsSigma2['Gain']), marker='o')
    ax.axhline(mergeresultsA[0]['Gain'])
    ax.fill_betweenx([mergeresultsA[0]['Gain']-np.sqrt(mergeresultsA[1]['Gain']), mergeresultsA[0]['Gain']+np.sqrt(mergeresultsA[1]['Gain'])], results['Run'].values[0], results['Run'].values[-1], alpha=0.5)
    ax.set_xlabel('Run')
    ax.set_ylabel('$G_1$')
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.errorbar(results['Run'], results['chargeMu']/50/1.6*ADC2mV, yerr=np.sqrt(resultsSigma2['chargeMu'])/50/1.6*ADC2mV, marker='o')
    ax.axhline(mergeresultsA[0]['chargeMu']/50/1.6*ADC2mV)
    ax.fill_betweenx([(mergeresultsA[0]['chargeMu']-np.sqrt(mergeresultsA[1]['chargeMu']))/50/1.6*ADC2mV, (mergeresultsA[0]['chargeMu']+np.sqrt(mergeresultsA[1]['chargeMu']))/50/1.6*ADC2mV], results['Run'].values[0], results['Run'].values[-1], alpha=0.5)
    ax.set_xlabel('Run')
    ax.set_ylabel('Gain')
    pdf.savefig(fig)
    # tts对比
    fig, ax = plt.subplots()
    ax.errorbar(results['Run'], results['TTS'], yerr=resultsSigma2['TTS'], marker='o', label='TTS')
    ax.errorbar(results['Run'], results['TTS_bin'], yerr=resultsSigma2['TTS_bin'], marker='o', label='TTS_bin')
    ax.set_xlabel('Run')
    ax.set_ylabel('TTS')
    ax.legend()
    pdf.savefig(fig)
    plt.close()
    # Afterpulse 变化
    if np.sum(pulseResults['begin10']>config.delay1B)/delay10window<1:
        binwidth = 20
    else:
        binwidth = 10
    
    # charge distribution and mean charge line plot
    fig, ax = plt.subplots(figsize=(15,6))
    selectT = (pulseResults['begin10']<config.laserE)|(pulseResults['begin10']>config.delay1B)
    h = ax.hist2d(pulseResults['begin10'][selectT], pulseResults['Q'][selectT], bins=[int((delay10E - config.DCRB)/binwidth/2), 300], range=[[config.DCRB, delay10E], [0, 6000]], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ## 分割区间统计
    chargeThreshold = 700
    pre_data = pulseResults[pulseResults['begin10']<config.laserE]
    after_data = pulseResults[pulseResults['begin10']>config.delay1B]
    pre_bins = np.linspace(config.DCRB, -config.promptE, int((-config.promptE - config.DCRB)/binwidth/2)+1)
    pre_x = (pre_bins[:-1]+pre_bins[1:])/2
    pre_label = np.digitize(pre_data['begin10'], pre_bins)
    pre_df = pd.DataFrame({'t': pre_data['begin10'], 'bin': pre_label, 'Q': pre_data['Q']})
    pre_mean = pre_df.groupby('bin')['Q'].mean().to_numpy()
    pre_std = pre_df.groupby('bin')['Q'].std().to_numpy()
    pre_filter_mean = pre_df.loc[pre_df['Q']<chargeThreshold].groupby('bin')['Q'].mean().to_numpy()
    after_bins = np.linspace(config.delay1B, config.delay10E, int((config.delay10E - config.delay1B)/binwidth/2)+1)
    after_x = (after_bins[:-1]+after_bins[1:])/2
    after_label = np.digitize(after_data['begin10'], after_bins)
    after_df = pd.DataFrame({'t': after_data['begin10'], 'bin': after_label, 'Q': after_data['Q']})
    after_meanDf = after_df.groupby('bin')['Q'].mean()
    after_stdDf = after_df.groupby('bin')['Q'].std()
    after_filter_meanDf = after_df.loc[after_df['Q']<chargeThreshold].groupby('bin')['Q'].mean()
    after_mean = np.zeros(after_bins.shape[0])
    after_mean[after_meanDf.index.to_numpy()-1] = after_meanDf.to_numpy()
    after_std = np.zeros(after_bins.shape[0])
    after_std[after_stdDf.index.to_numpy()-1] = after_stdDf.to_numpy()
    after_filter_mean = np.zeros(after_bins.shape[0])
    after_filter_mean[after_filter_meanDf.index.to_numpy()-1] = after_filter_meanDf.to_numpy()
    if np.min(pre_label)==0:
        start = 1
    else:
        start = 0
    ax.plot(pre_x, pre_mean[start:(start+len(pre_x))], color='k', lw=2, label='Mean')
    ax.plot(after_x, after_mean[:-1], color='k', lw=2)
    ax.plot(pre_x, pre_std[start:(start+len(pre_x))], color='orange', lw=2, label='Std')
    ax.plot(after_x, after_std[:-1], color='orange', lw=1)
    ax.plot(pre_x, pre_filter_mean[start:(start+len(pre_x))], color='k', ls='--', lw=2, label='Mean(Q<{}ADC$\cdot$ns)'.format(chargeThreshold))
    ax.plot(after_x, after_filter_mean[:-1], color='k', ls='--', lw=2)
    ax2 = ax.twinx()
    ax2.hist(pulseResults['begin10'], bins=int(delay10E/20), range=[200, delay10E], color='r', histtype='step', label='After-pulse')
    ax2.hist(pulseResults[pulseResults['Q']<chargeThreshold]['begin10'], bins=int(delay10E/20), range=[200, delay10E], histtype='step', color='g', label='After-pulse(Q<{}ADC$\cdot$ns)'.format(chargeThreshold))
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('Charge/ADC$\cdot$ns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    ax.legend()
    ax2.legend(loc="upper left")
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(15,6))
    selectT = (pulseResults['begin10']<-110)|(pulseResults['begin10']>config.delay1B)
    h = ax.hist2d(pulseResults['begin10'][selectT], pulseResults['Q'][selectT], bins=[int((delay10E - config.DCRB)/binwidth/2), 300], range=[[config.DCRB, delay10E], [0, 6000]], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    deltabins = int(100/binwidth/2)
    ax.plot(pre_x[:-deltabins], pre_filter_mean[start:(start+len(pre_x)-deltabins)], color='k', ls='--', lw=2, label='Mean(Q<{}ADC$\cdot$ns)'.format(chargeThreshold))
    ax.plot(pre_x[:-deltabins], pre_mean[start:(start+len(pre_x)-deltabins)], color='k', lw=2, label='Mean')
    ax.plot(after_x, after_mean[:-1], color='k', lw=2)
    ax.plot(after_x, after_filter_mean[:-1], color='k', ls='--', lw=2)
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('Charge/ADC$\cdot$ns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    ax.legend()
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(15,6))
    selectT = (pulseResults['begin10']<-110)|(pulseResults['begin10']>config.delay1B)
    h = ax.hist2d(pulseResults['begin10'][selectT], pulseResults['adjustQ'][selectT], bins=[int((delay10E - config.DCRB)/binwidth/2), 300], range=[[config.DCRB, delay10E], [0, 6000]], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('Charge/ADC$\cdot$ns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(15,6))
    selectT = (pulseResults['begin10']<-110)|(pulseResults['begin10']>config.delay1B)
    h = ax.hist2d(pulseResults['begin10'][selectT], (pulseResults['down10'][selectT]-pulseResults['begin10'][selectT]), bins=[int((delay10E - config.DCRB)/binwidth/2), 60], range=[[config.DCRB, delay10E], [0, 60]], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('t_over10/ns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)

    select15 = (pulseResults['down50'][selectT]-pulseResults['begin10'][selectT])<15
    fig, ax = plt.subplots(figsize=(15,6))
    selectT = (pulseResults['begin10']<-110)|(pulseResults['begin10']>config.delay1B)
    h = ax.hist2d(pulseResults['begin10'][selectT][select15], pulseResults['Q'][selectT][select15], bins=[int((delay10E - config.DCRB)/binwidth/2), 300], range=[[config.DCRB, delay10E], [0, 6000]], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('Charge/ADC$\cdot$ns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)
    fig, ax = plt.subplots(figsize=(15,6))
    selectT = (pulseResults['begin10']<-110)|(pulseResults['begin10']>config.delay1B)
    h = ax.hist2d(pulseResults['begin10'][selectT][select15], pulseResults['adjustQ'][selectT][select15], bins=[int((delay10E - config.DCRB)/binwidth/2), 300], range=[[config.DCRB, delay10E], [0, 6000]], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('adjust Charge/ADC$\cdot$ns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(pulseResults[pulseResults['begin10']<0]['begin10'], pulseResults[pulseResults['begin10']<0]['t'], range=[[-400,-10], [-400, 100]], bins=[39, 50], cmap=cmap)
    fig.colorbar(h[3])
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.hist(pulseResults[~pulseResults['isAfter']]['begin10'], range=[-400, 100], bins=50*2, histtype='step', label='begin10')
    ax.hist(pulseResults[~pulseResults['isAfter']]['t'], range=[-400, 100], bins=50*2, histtype='step', label='peak')
    ax.hist(pulseResults[pulseResults['isAfter']]['begin10'], range=[30, 200], bins=17*2, histtype='step', label='begin10')
    ax.set_yscale('log')
    ax.legend()
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(pulseResults_Total[pulseResults_Total['begin10']<0]['begin10'], pulseResults_Total[pulseResults_Total['begin10']<0]['t'], range=[[-400,0], [-400, 100]], bins=[40, 50], cmap=cmap)
    fig.colorbar(h[3])
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(pulseResults_Total[~pulseResults_Total['isAfter']]['begin10'], pulseResults_Total[~pulseResults_Total['isAfter']]['Q'], range=[[-300,100], [0, 2000]], bins=[400, 100], norm=colors.LogNorm(), cmap=cmap)
    fig.colorbar(h[3])
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('Charge/ADC$\cdot$ns')
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    # ax.hist(pulseResults_Total[pulseResults_Total['begin10']<0]['begin10'], range=[-400, 0], bins=40*2, histtype='step', label='begin10')
    # ax.hist(pulseResults_Total[pulseResults_Total['begin10']<0]['t'], range=[-400, 50], bins=45*2, histtype='step', label='peak')
    ax.hist(pulseResults_Total[(~pulseResults_Total['isTrigger'])&(~pulseResults_Total['isAfter'])]['begin10'], range=[-400, 100], bins=50*2, histtype='step', label='begin10 no main')
    ax.hist(pulseResults_Total[(~pulseResults_Total['isTrigger'])&(~pulseResults_Total['isAfter'])]['t'], range=[-400, 100], bins=50*2, histtype='step', label='peak no main')
    ax.hist(pulseResults_Total[(~pulseResults_Total['isTrigger'])&(pulseResults_Total['isAfter'])]['t'], range=[30, 200], bins=17*2, histtype='step', label='peak no main')
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('Entries')
    ax.legend()
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.hist(pulseResults_Total[(~pulseResults_Total['isAfter'])]['t'], range=[-400, 100], bins=50*2, histtype='step', label='prepulse peak')
    ax.hist(pulseResults_Total[(pulseResults_Total['isAfter'])]['t'], range=[30, 100], bins=7*2, histtype='step', label='afterpulse peak')
    ax.hist(pulseResults_Total[(~pulseResults_Total['isAfter'])]['begin10'], range=[-400, 100], bins=50*2, histtype='step', label='prepulse begin10')
    ax.hist(pulseResults_Total[(pulseResults_Total['isAfter'])]['begin10'], range=[30, 100], bins=7*2, histtype='step', label='afterpulse begin10')
    ax.set_xlim([-300, 100])
    ax.set_yscale('log')
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('Entries')
    ax.legend()
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(15,6))
    ax.hist(pulseResults_Total[(~pulseResults_Total['isAfter'])]['begin10'], range=[-400, 100], bins=50, histtype='step', label='prepulse begin10')
    ax.hist(pulseResults_Total[(pulseResults_Total['isAfter'])]['begin10'], range=[30, 9800], bins=977, histtype='step', label='afterpulse begin10')
    ax.hist(pulseResults_Total[(~pulseResults_Total['isAfter'])&(~selectEvent)]['begin10'], range=[-400, 100], bins=50, histtype='step', label='prepulse begin10 no main pulse')
    ax.hist(pulseResults_Total[(pulseResults_Total['isAfter'])&(~selectEvent)]['begin10'], range=[30, 9800], bins=977, histtype='step', label='afterpulse begin10 no main pulse')
    ax.axhline(mergePulseResults[0]['meanDCR'] * 10 / 1E6 * totalNum)
    ax.set_ylim([mergePulseResults[0]['meanDCR'] * 10 / 1E6 * totalNum / 1.2, mergePulseResults[0]['meanDCR'] * 10 / 1E6 * totalNum * 2])
    ax.set_xlim([-400, 9800])
    ax.set_yscale('log')
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('Entries')
    ax.legend()
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(15,6))
    h1 = ax.hist(pulseResults['begin10'], bins=int(delay10E/binwidth), range=[0, delay10E], histtype='step', label='After-pulse')
    h2 = ax.hist(pulseResults['begin10'], bins=int((-config.DCRB - promptE)/binwidth), range=[config.DCRB, -promptE], histtype='step', label='Pre-pulse')
    ax.hist(pulseResults[pulseResults['peak']>2]['begin10'], bins=int((-config.DCRB - promptE)/binwidth), range=[config.DCRB, -promptE], histtype='step', label='Pre-pulse >2ADC')
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('Entries')
    ax.set_xlim([config.DCRB, delay10E])
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlim([config.DCRB, delay1B])
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    pdf.savefig(fig)

    # fig, ax = plt.subplots(figsize=(15,6))
    # ax.hist2d(pulseResults['begin10'], bins=int(delay10E/binwidth), range=[0, delay10E])

    # 计算后脉冲比例
    binwidth_large = binwidth * 2
    if PMTName.startswith('PM'):
        searchwindows = config.searchwindowsMCP
        fitwindow = config.fitWindowMCP
    else:
        searchwindows = config.searchwindowsHama
        fitwindow = config.fitWindowHama
    expectPrompt = mergePulseResults[0]['meanDCR'] * binwidth_large / 1E6 * totalTriggerNum
    expectPromptLocal = np.sum((pulseResults['t']>config.DCRB)&(pulseResults['t']<config.DCRE))/(config.DCRE-config.DCRB) * binwidth_large
    MCPPeakNum = np.zeros((len(searchwindows), 2), dtype=[('Group', '<i2'), ('t', '<f4'), ('N', '<i4'), ('pv', '<f4'), ('left', '<f4'), ('right', '<f4'), ('sigma', '<f4'), ('ratio', '<f4'), ('charge', '<f4'), ('chargeSigma', '<f4'), ('chargeSample', '<f4'), ('chargeSigmaSample', '<f4'), ('chargeDirect', '<f4'), ('A', '<f4')])
    
    fig, ax = plt.subplots(figsize=(15,6))
    h_a = ax.hist(pulseResults['begin10'], bins=int(delay10E/binwidth_large), range=[0, delay10E], histtype='step', label='After-pulse')
    ax.hist(pulseResults[pulseResults['Q']<chargeThreshold]['begin10'], bins=int(delay10E/binwidth_large), range=[0, delay10E], histtype='step', label='After-pulse(Q<{}ADC$\cdot$ns)'.format(chargeThreshold))
    counts, edges = h_a[0] - expectPrompt, (h_a[1][:-1] + h_a[1][1:])/2
    h_p = ax.hist(pulseResults['begin10'], bins=int((-config.DCRB - promptB+40)/binwidth_large), range=[config.DCRB-40, -promptB], histtype='step', label='Pre-pulse')
    # 改变搜索算法为拟合算法
    for i, w in enumerate(searchwindows):
        MCPPeakNum['Group'][i, 0] = i
        area = (edges<=w[1]) & (edges>=w[0])
        selectCounts = counts[area]
        pi = np.argmax(selectCounts)
        pv = selectCounts[pi]
        MCPPeakNum['t'][i, 0] = edges[area][pi]
        MCPPeakNum['pv'][i, 0] = pv
        selectArea = selectCounts>(0.5*pv)
        if np.sum(selectArea)>1:
            MCPPeakNum['N'][i, 0] = np.sum(selectCounts[selectArea])
            MCPPeakNum['left'][i, 0] = edges[area][selectArea][0]
            MCPPeakNum['right'][i, 0] = edges[area][selectArea][-1]
        else:
            MCPPeakNum['left'][i, 0] = w[0]
            MCPPeakNum['right'][i, 0] = w[1]
        # ax.fill_between(edges[area][selectArea], selectCounts[selectArea] + expectPrompt, np.ones(np.sum(selectArea)) * expectPrompt)
        print(edges[area][selectArea], selectCounts[selectArea], np.sum(selectArea))
    x0 = np.vstack([MCPPeakNum['pv'][:,0], MCPPeakNum['t'][:,0], (MCPPeakNum['right'][:,0] - MCPPeakNum['left'][:,0])/2+10]).T.reshape(-1).astype('float64')
    print('initial parameters: {}'.format(x0))
    bounds = []
    for idx, sw in enumerate(searchwindows):
        A_expect = max(MCPPeakNum['pv'][idx, 0], 10)
        bounds.append((0.5*A_expect, max(1.2*(MCPPeakNum['pv'][idx, 0]), A_expect)))
        bounds.append(sw)
        if len(boundsSigma)>0:
            bounds.append(boundsSigma[idx])
        else:
            bounds.append((5,100))
        x0[3*idx+2] += bounds[-1][-1]/10
        if bounds[-1][-1]<x0[3*idx+2]:
            x0[3*idx+2] = bounds[-1][-1]-1
        if bounds[-1][0]>x0[3*idx+2]:
            x0[3*idx+2] = (bounds[-1][0]+bounds[-1][1])/2
    startEdges = int((250)/binwidth_large)
    endEdges = int(searchwindows[-1][-1]/binwidth_large)
    f_A = ROOT.TF1("", "gaus(0)+gaus(3)+gaus(6)+gaus(9)+gaus(12)+{}".format(expectPrompt), 250, fitwindow[-1])
    f_A.SetParameters(x0.reshape((-1)))
    for ib,bo in enumerate(bounds):
        print('bounds: {}'.format([bo[0], bo[1]]))
        f_A.SetParLimits(ib, bo[0], bo[1])
    hists_A = ROOT.TH1D("", "", int(delay10E/binwidth_large), 0, delay10E)
    for i in range(1, len(h_a[1])):
        hists_A.SetBinContent(i, h_a[0][i-1])
    hists_A.Fit(f_A, 'R')
    print('finish fit')
    pars, parErrors, par_length = f_A.GetParameters(), f_A.GetParErrors(), len(searchwindows)*3
    pars.reshape((par_length,))
    parErrors.reshape((par_length,))
    aftergroupsX, aftergroupsXErrors = np.frombuffer(pars, dtype=np.float64, count=par_length).reshape((-1,3)), np.frombuffer(parErrors, dtype=np.float64, count=par_length).reshape((-1,3))
    print('finish par get')
    # aftergroups = minimize(
    #     likelihood, x0,
    #     args=(edges[startEdges:endEdges], counts[startEdges:endEdges]),
    #     bounds=bounds,
    #     options={'eps':0.0001}
    #     )
    # aftergroupsX = aftergroups.x.reshape((-1,3))
    MCPPeakNum['t'][:, 0] = aftergroupsX[:, 1]
    MCPPeakNum['pv'][:, 0] = aftergroupsX[:, 0]*np.sqrt(2*np.pi)*aftergroupsX[:, 2]/binwidth_large
    MCPPeakNum['sigma'][:, 0] = aftergroupsX[:, 2]
    MCPPeakNum['ratio'][:, 0] = MCPPeakNum['pv'][:, 0]/totalTriggerNum
    MCPPeakNum['t'][:, 1] = aftergroupsXErrors[:, 1]**2
    MCPPeakNum['pv'][:, 1] = (aftergroupsXErrors[:, 0]**2 + aftergroupsXErrors[:, 2]**2) * 2*np.pi/binwidth_large**2
    MCPPeakNum['sigma'][:, 1] = aftergroupsXErrors[:, 2]**2
    MCPPeakNum['ratio'][:, 1] = MCPPeakNum['pv'][:, 1]/totalTriggerNum**2

    exs = np.arange(fitwindow[0], fitwindow[1])
    eys = Afterpulse(aftergroupsX.reshape(-1), exs)
    ax.plot(exs, eys+expectPrompt, linewidth=1, alpha=0.9, label='fit')
    ax.axhline(expectPrompt, linewidth=1, linestyle='--', label='Average prepulse totalDataset')
    ax.axhline(expectPromptLocal, linewidth=1, linestyle='--', color='r', label='Average prepulse main pulse')
    if darkExpect:
        ax.axhline(totalTriggerNum * darkresult[0]['DCR'] * binwidth_large * 1e-6, label='Expected Dark Noise')
    # for pn in MCPPeakNum:
    #     ax.annotate(pn['N'], (pn['t'], pn['pv'] + expectPrompt), (pn['t'], pn['pv'] + expectPrompt + 5))
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('# per '+'{}ns'.format(binwidth_large))
    ax.set_xlim([config.DCRB, delay10E])
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.legend()
    pdf.savefig(fig)
    ax.set_xlim([config.DCRB, delay1B])
    ax.set_yscale('log')
    pdf.savefig(fig)

    fig, ax = plt.subplots(figsize=(15,6))
    ax.hist(pulseResults['begin10'], bins=int(delay10E/binwidth_large), range=[delay1B, delay10E], histtype='step', label='After-pulse')
    ax.hist(pulseResults[pulseResults['Q']<chargeThreshold]['begin10'], bins=int(delay10E/binwidth_large), range=[delay1B, delay10E], histtype='step', label='After-pulse(Q<{}ADC$\cdot$ns)'.format(chargeThreshold))
    ax.hist(pulseResults['begin10'], bins=int((-config.DCRB +40 - config.promptB)/binwidth_large), range=[config.DCRB-40, -config.promptB], histtype='step', label='Dark Noise')
    ax.plot(exs, eys+expectPrompt, linewidth=1, alpha=0.9, label='Fit')
    ax.axhline(expectPrompt, linewidth=1, linestyle='--', label='DCR')
    ax.set_xlabel('Delay time/ns')
    ax.set_ylabel('# per '+'{}ns'.format(binwidth_large))
    ax.set_xlim([config.DCRB, delay10E])
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    ax.legend()
    pdf.savefig(fig)
    ax.scatter([0], [totalTriggerNum/1000], marker='o', label='$N^{\mathrm{hit}}$/1E3')
    ax.set_yscale('log')
    ax.legend()
    pdf.savefig(fig)
    # ax.set_yscale('function', functions=(lambda x: x/binwidth_large/totalTriggerNum, lambda x: x*binwidth_large*totalTriggerNum))
    ticks = FuncFormatter(lambda x, pos: '{0:.1E}'.format(x/binwidth_large/totalTriggerNum))
    ax.yaxis.set_major_formatter(ticks)
    ax.yaxis.set_minor_formatter(ticks)
    ax.set_ylabel('Entries/$N^{\mathrm{hit}}$/'+'{}ns'.format(binwidth_large))
    pdf.savefig(fig)
    plt.close()

    ## 各个After-pulse峰的charge分布,宽DCRE-DCRB
    bins = np.arange(0, 8000, 40)
    fig, ax = plt.subplots(figsize=(15, 6))
    fig2, ax2 = plt.subplots(figsize=(15, 6))
    colors =['b', 'yellow', 'r', 'g', 'k', 'purple']
    h_dcr = ax.hist(pulseResults['Q'][(pulseResults['begin10']>config.DCRB)&(pulseResults['begin10']<(config.DCRE))], histtype='step', bins=bins, color=colors[0], label='Dark Noise')
    DCRNum = np.sum((pulseResults['begin10']>config.DCRB)&(pulseResults['begin10']<(config.DCRE)))
    im = 0
    for mcpr in MCPPeakNum:
        if mcpr['Group'][0]!=1:
            selectT = (pulseResults['begin10']>(mcpr['t'][0]-3*mcpr['sigma'][0]))&(pulseResults['begin10']<(mcpr['t'][0]+3*mcpr['sigma'][0]))
            selectQ = pulseResults['Q'][selectT]
            scaledDCRNum = DCRNum / (config.DCRE-config.DCRB) * mcpr['sigma'][0]*6
            h = ax.hist(selectQ, weights=np.repeat((config.DCRE-config.DCRB)/mcpr['sigma'][0]/6, np.sum(selectT)), histtype='step', bins=bins, color=colors[im+1], label='{}-th Peak'.format(mcpr['Group'][0]+1))
            ax2.plot(h_dcr[1][:-1], h[0]-h_dcr[0], label='{}-th Peak'.format(mcpr['Group'][0]+1))
            if np.sum(selectQ>700)==0:
                mu_c, sigma_c, mu_4 = 0, 0, 0
            else:
                muN = np.sum(selectQ>700)
                mu_c, sigma_c = np.average(selectQ[selectQ>700]), np.std(selectQ[selectQ>700])
                assert(mu_c<10000)
                mu_4 = np.sum((selectQ[selectQ>700]-mu_c)**4)/(muN-1)
                # Fit
                begin, end = np.max([700,mu_c-2*sigma_c]), np.min([mu_c+2*min(sigma_c,1000), 6000])
                f1 = ROOT.TF1("", "gaus", begin, end)
                f1.SetParameters(np.array([np.sum(selectT), mu_c, sigma_c]))
                f1.SetParLimits(1, mu_c-sigma_c/2, mu_c+sigma_c/2)
                hists = ROOT.TH1D("", "", 200, 0, 8000)
                for i in range(1, len(bins)):
                    hists.SetBinContent(i, h[0][i-1])
                hists.Fit(f1, 'R')
                pars, parErrors = f1.GetParameters(), f1.GetParErrors()
                ax.plot(np.arange(begin, end, 10), pars[0]*np.exp(-0.5*(np.arange(begin, end, 10)-pars[1])**2/pars[2]**2), color=colors[im+1], alpha=0.8)
                MCPPeakNum[['charge', 'chargeSigma', 'A', 'chargeSample', 'chargeSigmaSample']][mcpr['Group'][0], 0] = pars[1], pars[2], pars[0], mu_c, sigma_c
                MCPPeakNum['chargeDirect'][mcpr['Group'][0], 0] = (np.sum(selectQ)-mergeresultsA['chargeMu'][0]*scaledDCRNum)/(np.sum(selectT)-scaledDCRNum)
                MCPPeakNum[['charge', 'chargeSigma', 'A', 'chargeSample', 'chargeSigmaSample']][mcpr['Group'][0], 1] = parErrors[1]**2, parErrors[2]**2, parErrors[0]**2, sigma_c**2/np.sum(selectQ>700), (mu_4-(muN-3)/(muN-1)*sigma_c**4)/muN/sigma_c**2
                print(mcpr['Group'][0], mu_c, sigma_c, pars[1], pars[2], MCPPeakNum['chargeDirect'][mcpr['Group'][0]])
            im += 1

    ax.scatter(MCPPeakNum[:,0]['charge'], MCPPeakNum[:,0]['A'], color='violet', label='Fit')
    ax.scatter(MCPPeakNum[:,0]['chargeDirect'], MCPPeakNum[:,0]['A'], color='cyan', label='Calculation')
    ax.set_xlabel('Charge/ADC$\cdot$ns')
    ax.set_ylabel('Scaled Entries')
    # ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.legend()
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)
    ax2.set_xlabel('Charge/ADC$\cdot$ns')
    ax2.set_ylabel('Scaled Entries')
    ax2.legend()
    ax2.set_yscale('log')
    pdf.savefig(fig2)
    print(MCPPeakNum.dtype)
    print(MCPPeakNum)
    print('total group {:.3f} 0-2000ns {:.3f} afterpulse {:.3f}={:.3f} TriggerNum {} TotalNum {}'.format(np.sum(MCPPeakNum['pv'][:,0])/totalTriggerNum, (np.sum((pulseResults['begin10']<2000)&(pulseResults['begin10']>config.delay1B))-expectPrompt*(2000-config.delay1B)/binwidth_large)/totalTriggerNum,mergePulseResults[0]['delayAllWODCR'], (np.sum((pulseResults['begin10']>config.delay1B)&(pulseResults['begin10']<config.delay10E))-expectPrompt*(config.delay10E-config.delay1B)/binwidth_large)/totalTriggerNum, totalTriggerNum, totalNum))

    # TT 各成分比例
    fig, ax = plt.subplots()
    ax.plot(results['Run'], TT_kToTT, marker='o', label='Early component')
    # TTS分布0.5ns的binwidth
    ax.plot(results['Run'], TT_expToTT, marker='o', label='Exp component')
    ax.plot(results['Run'], TT_kToTT+TT_expToTT, marker='o', label='Total')
    ax.set_xlabel('Run')
    ax.set_ylabel('ratio')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    # main pulse charge和afterpulse之间电荷关系
    fig, ax = plt.subplots()
    selectAfter = pulseResults['begin10']>config.delay1E
    h = ax.hist2d(pulseResults[selectAfter]['Q'], pulseResults[selectAfter]['minPeakCharge'], range=[[0, 6000], [0, 1500]], bins=[300, 75], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('After-pulse Charge/ADC$\cdot$ns')
    ax.set_ylabel('Main-pulse Charge/ADC$\cdot$ns')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(pulseResults[selectAfter&(pulseResults['Q']<chargeThreshold)]['minPeakCharge'], range=[0, 1500], bins=75, density=True, histtype='step', label='After-pulse Q<{}ADC$\cdot$ns'.format(chargeThreshold))
    ax.hist(pulseResults[selectAfter&(pulseResults['Q']>1000)]['minPeakCharge'], range=[0, 1500], bins=75, density=True, histtype='step', label='After-pulse Q>1000ADC$\cdot$ns')
    ax.set_xlabel('Main pulse Charge/ADC$\cdot$ns')
    ax.set_ylabel('Normolized probability')
    ax.legend()
    # ax.xaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.hist(pulseResults_Total[(pulseResults_Total['begin10']<config.DCRE)&(pulseResults_Total['begin10']>config.DCRB)]['Q'], range=[0, 1000], bins=100, density=True, histtype='step', label='DarkNoise')
    ax.set_xlabel('Main pulse Charge/ADC$\cdot$ns')
    ax.set_ylabel('Normolized probability')
    ax.legend()
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    pdf.savefig(fig)
    plt.close()

# 统计结果并合并存储
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('concat', data=results.to_records(), compression='gzip')
    opt.create_dataset('merge', data=mergeresultsA, compression='gzip')
    opt.create_dataset('mergepulse', data=mergePulseResults, compression='gzip')
    opt.create_dataset('mergeSER', data=mergeSERResults, compression='gzip')
    opt.create_dataset('AfterPulse', data=MCPPeakNum, compression='gzip')
