import sys
sys.path.append('..')
import matplotlib.pyplot as plt
plt.style.use('../journal.mplstyle')
import h5py, argparse
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import matplotlib.colors as colors
from scipy.optimize import minimize
import config
from csvDatabase import OriginINFO
from waveana.util import peakResidual, vallyResidual, fitGausB, centralMoment, RootFit
import ROOT
from scipy.special import erf
ADC2mV = config.ADC2mV
rootfit = RootFit()
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
psr.add_argument('--run', type=int, help='run no')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-t', dest='trigger', help='trigger h5 file')
args = psr.parse_args()
info = []
results = np.zeros(len(args.channel),
    dtype=[
        ('Channel', '<i2'),
        ('peakC','<f4'), ('vallyC','<f4'), ('peakV','<f4'), ('vallyV','<f4'), ('peakS','<f4'), ('vallyS','<f4'), ('PV','<f4'),
        ('chargeMu','<f4'), ('chargeSigma2','<f4'),
        ('Gain', '<f4'), ('GainSigma', '<f4'),
        ('TriggerRate', '<f4'), ('TotalNum', '<i4'), ('TriggerNum', '<i4'), ('window', '<f4'),
        ('TTS', '<f4'), ('TTS2', '<f4'), ('TT', '<f4'), ('TT2', '<f4'), ('TTA', '<f4'), ('TTA2', '<f4'), ('TTS_exp', '<f4'), ('TTA_exp', '<f4'), ('DCR', '<f4'), ('DCR_exp', '<f4'), ('TTSinterval', '<f4'), ('TTS_bin', '<f4'),
        ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'),
        ('RiseSigma2', '<f4'), ('FallSigma2', '<f4'), ('THSigma2', '<f4'), ('FWHMSigma2', '<f4')
        ])
paraSigma2 = np.zeros(len(args.channel),
    dtype=[
        ('Channel', '<i2'),
        ('peakC','<f4'), ('vallyC','<f4'), ('peakV','<f4'), ('vallyV','<f4'), ('peakS','<f4'), ('vallyS','<f4'), ('PV','<f4'),
        ('chargeMu','<f4'), ('chargeSigma2','<f4'),
        ('Gain', '<f4'), ('GainSigma', '<f4'),
        ('TriggerRate', '<f4'), ('TotalNum', '<i4'), ('TriggerNum', '<i4'), ('window', '<f4'),
        ('TTS', '<f4'), ('TTS2', '<f4'), ('TT', '<f4'), ('TT2', '<f4'), ('TTA', '<f4'), ('TTA2', '<f4'), ('TTS_exp', '<f4'), ('TTA_exp', '<f4'), ('DCR', '<f4'), ('DCR_exp', '<f4'), ('TTSinterval', '<f4'), ('TTS_bin', '<f4'),
        ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'),
        ('RiseSigma2', '<f4'), ('FallSigma2', '<f4'), ('THSigma2', '<f4'), ('FWHMSigma2', '<f4')
    ])
with h5py.File(args.ipt, 'r') as ipt:
    waveformLength = ipt.attrs['waveformLength']
    for j in range(len(args.channel)):
        info.append(ipt['ch{}'.format(args.channel[j])][:])
with h5py.File(args.trigger, 'r') as ipt:
    rinterval = ipt['rinterval'][:]
# PMT type: isMCP?
runno = args.run
pmtinfo = OriginINFO(config.databaseDir+'/{}.csv'.format(runno))
ismcp = np.array([pmt.startswith('PM') for pmt in pmtinfo.getPMT()])

rangemin =-100
rangemax = 500
bins = rangemax-rangemin

# set the figure appearance
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
jet = plt.cm.jet
newcolors = jet(np.linspace(0, 1, 32768))
white = np.array([1, 1, 1, 0.5])
newcolors[0, :] = white
cmap = ListedColormap(newcolors)
print('begin plot')
pdf = PdfPages(args.opt)
for j in range(len(args.channel)):
    rangemin = int(np.min(info[j]['minPeakCharge'])-1)
    rangemax = int(np.max(info[j]['minPeakCharge'])+1)
    bins = rangemax-rangemin
    if bins < 100:
        print('Warning: no signal charge more than 100')
        continue
    fig, ax = plt.subplots()
    h = ax.hist(info[j]['minPeakCharge'], histtype='step', bins=bins, range=[rangemin, rangemax], lw=3, alpha=0.9, label='charge')
    ax.hist(info[j]['minPeakCharge'][(info[j]['minPeak']>3)], histtype='step', bins=bins, range=[rangemin, rangemax], ls='--', label='charge($V_p$>3ADC)')
    ax.set_xlabel('Charge/ADC$\cdot$ns')
    ax.set_ylabel('Entries')
    ax.legend(loc='best')
    ax.set_xlim([-100, 600])
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_yscale('linear')
    ## 拟合峰值处所需参数
    zeroOffset = np.where(h[1]>=0)[0][0]
    ax.set_ylim([0, 1.5*np.max(h[0][(zeroOffset+70):(zeroOffset+150)])])
    vi_r = np.argmin(h[0][(zeroOffset+15):(zeroOffset+70)])
    vi = int(h[1][(zeroOffset+15):(zeroOffset+70)][vi_r])
    pi_r = np.argmax(h[0][(zeroOffset+15+vi_r):(zeroOffset+15+vi_r+100)])
    pi = h[1][(zeroOffset+15+vi_r):(zeroOffset+15+vi_r+100)][pi_r]
    pv = h[0][(zeroOffset+15+vi_r):(zeroOffset+15+vi_r+100)][pi_r]
    vv = h[0][(zeroOffset+15):(zeroOffset+70)][vi_r]
    print(([pi, vi], [pv, vv]))
    ## 初步估计一个拟合区间
    estimateG = 15 + vi_r + pi_r
    peakspanl, peakspanr = int(config.peakspanl * estimateG), int(config.peakspanr * estimateG)
    vallyspanl, vallyspanr = int(config.vallyspanl * estimateG), int(config.vallyspanr * estimateG)
    # 使用上面值作为初值进行最小二乘拟合
    hi = zeroOffset + 15 + vi_r + pi_r
    result = minimize(peakResidual, [pv, pi, 30], 
        args=(h[0][(hi-peakspanl):(hi+peakspanr)], (h[1][(hi-peakspanl):(hi+peakspanr)]+h[1][(hi-peakspanl+1):(hi+peakspanr+1)])/2),
        bounds=[(0,None), (5, None), (0.1, None)],
        method='SLSQP', options={'eps': 0.1})
    print(result)
    ## 使用拟合值为初值，降低步长再次拟合
    pv, pi = result.x[0], result.x[1]
    hi = zeroOffset + int(pi)
    estimateG = pi
    peakspanl, peakspanr = int(config.peakspanl * estimateG), int(config.peakspanr * estimateG)
    vallyspanl, vallyspanr = int(config.vallyspanl * estimateG), int(config.vallyspanr * estimateG)
    result = minimize(peakResidual, result.x, 
        args=(h[0][(hi-peakspanl):(hi+peakspanr)], (h[1][(hi-peakspanl):(hi+peakspanr)]+h[1][(hi-peakspanl+1):(hi+peakspanr+1)])/2),
        bounds=[(0,None), (5, None), (0, None)],
        method='SLSQP', options={'eps': 1e-5})
    print(result)
    ## ROOT fit
    rootfit.setFunc(ROOT.TF1("", "[0]*exp(-0.5*((x-[1])/[2])^2)", h[1][hi - peakspanl], h[1][hi + peakspanr]), result.x)
    rootfit.setHist(h[1], h[0])
    paraRoot, errorRoot = rootfit.Fit()
    print((paraRoot[0], paraRoot[1], paraRoot[2]), (errorRoot[0], errorRoot[1], errorRoot[2]))
    A, mu, sigma = paraRoot[0], paraRoot[1], paraRoot[2]
    results['peakS'][j] = sigma
    paraSigma2[['Channel', 'peakC', 'peakV', 'peakS']][j] = (args.channel[j], errorRoot[1]**2, errorRoot[0]**2, errorRoot[2]**2)
    ## 绘制拟合结果
    ax.plot(h[1][(hi-peakspanl):(hi+peakspanr)], 
        A*np.exp(-(h[1][(hi-peakspanl):(hi+peakspanr)]-mu)**2/2/sigma**2), color='r', label='peak fit')
    pi = int(mu)
    pv = A
    ax.fill_betweenx([0, pv], h[1][hi-peakspanl], h[1][hi+peakspanr], alpha=0.5, color='lightsalmon', label='peak fit interval')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 2*pv])
    ax.axvline(0.25*mu, ymax=0.5,linestyle='--', label='0.25$Q_0$')
    ## 拟合峰谷处所需参数,smooth不是必须的，polyfit不能进行MLE拟合
    li = zeroOffset + 15 + vi_r
    yy  = h[0][(li-vallyspanl):(li+vallyspanr)]
    ## 对vally进行区间调整，放置左侧padding过长
    while (yy[0] > 1.5 * np.max(yy[-10:])) and vallyspanl > 2:
        vallyspanl = vallyspanl - 1
        yy  = h[0][(li-vallyspanl):(li+vallyspanr)]
    result = minimize(vallyResidual, [0.3, vi, vv + 5], args=(yy, (h[1][(li-vallyspanl):(li+vallyspanr)] + h[1][(li-vallyspanl+1):(li+vallyspanr+1)])/2),
        bounds=[(0.1, None), (vi-5, vi+5), (3, A)],
        method='SLSQP', options={'eps': 0.1, 'maxiter':5000})
    print(result)
    ## ROOT fit
    rootfit.setFunc(ROOT.TF1("", "[0]*(x-[1])^2+[2]", h[1][li-vallyspanl], h[1][li+vallyspanr]), result.x)
    rootfit.func.SetParLimits(0, 0.001, 100000)
    rootfit.func.SetParLimits(1, 5, pi)
    rootfit.func.SetParLimits(2, 1, A)
    rootfit.setHist(h[1], h[0])
    paraRoot, errorRoot = rootfit.Fit()
    print((paraRoot[0], paraRoot[1], paraRoot[2]), (errorRoot[0], errorRoot[1], errorRoot[2]))
    a_v, b_v, c_v = paraRoot[0]*100, paraRoot[1], paraRoot[2]
    results['vallyS'][j] = paraRoot[0]
    paraSigma2[['vallyC', 'vallyV', 'vallyS']][j] = (errorRoot[1]**2, errorRoot[2]**2, errorRoot[0]**2)
    ax.plot(h[1][(li-vallyspanl):(li+vallyspanr)], a_v/100 * (h[1][(li-vallyspanl):(li+vallyspanr)] - b_v)**2 +c_v, color='g', label='vally fit')
    vi, vv = b_v, c_v
    ax.fill_betweenx([0, vv], h[1][li-vallyspanl], h[1][li+vallyspanr], alpha=0.5, color='lightgreen', label='vally fit interval')
    ax.scatter([pi, vi], [pv, vv], color='r')
    ## 将参数放入legend里
    totalselect = (info[j]['minPeak']>3)&(info[j]['minPeakCharge']>0.25*mu)&(info[j]['minPeakCharge']<1000)&info[j]['isTrigger']
    selectinfo = info[j]['minPeakCharge'][totalselect]
    results[['Channel', 'peakC', 'vallyC', 'peakV', 'vallyV', 'PV', 'chargeMu', 'chargeSigma2', 'Gain', 'GainSigma', 'TotalNum', 'TriggerNum', 'window']][j] = (args.channel[j], mu, vi, pv, vv, pv / vv, np.mean(selectinfo), np.var(selectinfo, ddof=1), mu/50/1.6*ADC2mV, sigma/50/1.6*ADC2mV, len(info[j]), np.sum(totalselect), rinterval[j][1] - rinterval[j][0])
    paraSigma2[['PV', 'Gain', 'GainSigma', 'chargeMu', 'chargeSigma2', 'TriggerNum']][j] = (
        results[j]['PV']**2 * (paraSigma2[j]['peakV']/results[j]['peakV']**2 + paraSigma2[j]['vallyV']/results[j]['vallyV']**2),
        paraSigma2[j]['peakC']/(50*1.6/ADC2mV)**2,
        paraSigma2[j]['peakS']/(50*1.6/ADC2mV)**2,
        results[j]['chargeSigma2']/results[j]['TriggerNum'],
        (centralMoment(selectinfo, results[j]['chargeMu'], 4) - ((results[j]['TriggerNum']-3)/(results[j]['TriggerNum']-1)*results[j]['chargeSigma2']**2))/results[j]['TriggerNum'],
        results[j]['TriggerNum']
    )
    ax.annotate("$N_p$", (results[j]['peakC']+10, results[j]['peakV']), color='k')
    ax.annotate("$N_v$", (results[j]['vallyC']+20, results[j]['vallyV']*0.99), color='k')
    ax.annotate("", xy=(results[j]['peakC'] - results[j]['GainSigma']*50*1.6/ADC2mV, results[j]['peakV']*1.1), xycoords='data', xytext=(results[j]['peakC'] + results[j]['GainSigma']*50*1.6/ADC2mV, results[j]['peakV']*1.1),
                arrowprops=dict(arrowstyle="<->, widthA=0.5, widthB=0.5",
                connectionstyle="arc3"),)
    ax.annotate(r"N$\left(Q_0,\sigma_{Q_0}^2=(\nu_0Q_0)^2\right)$", (results[j]['peakC'], results[j]['peakV']*1.15))
    ax.annotate("", xy=(0.25*mu, results[j]['peakV']*0.5), xycoords='data', xytext=(600, results[j]['peakV']*0.5),
                arrowprops=dict(arrowstyle="<|-, widthA=0.5, widthB=0.5",
                connectionstyle="arc3"),)
    ax.annotate("", xy=(results[j]['chargeMu']-np.sqrt(results[j]['chargeSigma2']), results[j]['peakV']*0.55), xycoords='data', xytext=(results[j]['chargeMu']+np.sqrt(results[j]['chargeSigma2']), results[j]['peakV']*0.55),
                arrowprops=dict(arrowstyle="<->, widthA=0.5, widthB=0.5",
                connectionstyle="arc3"),)
    ax.annotate(r"N$\left(\overline{Q},s_Q^2=(\nu\overline{Q})^2\right)$", (results[j]['chargeMu'], results[j]['peakV']*0.6))
    handles, labels = ax.get_legend_handles_labels()
    # handles.append(mpatches.Patch(color='none', label='G/1E7:{:.2f}$\pm${:.2f}'.format(results[j]['Gain'], np.sqrt(paraSigma2[j]['Gain']))))
    # handles.append(mpatches.Patch(color='none', label='$\sigma_G$/1E7:{:.2f}$\pm${:.2f}'.format(results[j]['GainSigma'], np.sqrt(paraSigma2[j]['GainSigma']))))
    # handles.append(mpatches.Patch(color='none', label='P/V:{:.2f}$\pm${:.2f}'.format(results[j]['PV'], np.sqrt(paraSigma2[j]['PV']))))
    # handles.append(mpatches.Patch(color='none', label='$\mu_{select}$:'+'{:.2f}$\pm${:.2f}'.format(results[j]['chargeMu'], np.sqrt(paraSigma2[j]['chargeMu']))))
    # handles.append(mpatches.Patch(color='none', label='$\sigma_{select}$'+':{:.2f}$\pm${:.2f}'.format(np.sqrt(results[j]['chargeSigma2']), np.sqrt(paraSigma2[j]['chargeSigma2'])/2/np.sqrt(results[j]['chargeSigma2']))))
    ax.legend(handles=handles)
    pdf.savefig(fig)
    plt.close()
pdf.close()
