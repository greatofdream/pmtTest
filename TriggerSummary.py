'''
该文件计算激光触发符合区域的事例对应的参量，以及TTS
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
ADC2mV = config.ADC2mV
rootfit = RootFit()
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
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
        ('TTS', '<f4'), ('TTS2', '<f4'), ('TT', '<f4'), ('TT2', '<f4'), ('TTA', '<f4'), ('TTA2', '<f4'), ('DCR', '<f4'), ('TTSinterval', '<f4'), ('TTS_bin', '<f4'),
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
        ('TTS', '<f4'), ('TTS2', '<f4'), ('TT', '<f4'), ('TT2', '<f4'), ('TTA', '<f4'), ('TTA2', '<f4'), ('DCR', '<f4'), ('TTSinterval', '<f4'), ('TTS_bin', '<f4'),
        ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'),
        ('RiseSigma2', '<f4'), ('FallSigma2', '<f4'), ('THSigma2', '<f4'), ('FWHMSigma2', '<f4')
    ])
with h5py.File(args.ipt, 'r') as ipt:
    waveformLength = ipt.attrs['waveformLength']
    for j in range(len(args.channel)):
        info.append(ipt['ch{}'.format(args.channel[j])][:])
with h5py.File(args.trigger, 'r') as ipt:
    rinterval = ipt['rinterval'][:]
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
pdf = PdfPages(args.opt+'.pdf')

# 下面循环绘制每个channel的图像
for j in range(len(args.channel)):
    rangemin = int(np.min(info[j]['minPeakCharge'])-1)
    rangemax = int(np.max(info[j]['minPeakCharge'])+1)
    bins = rangemax-rangemin
    if bins < 100:
        print('Warning: no signal charge more than 100')
        continue
    # charge分布
    fig, ax = plt.subplots()
    h = ax.hist(info[j]['minPeakCharge'], histtype='step', bins=bins, range=[rangemin, rangemax], label='charge')
    ax.hist(info[j]['minPeakCharge'][(info[j]['minPeak']>3)], histtype='step', bins=bins, range=[rangemin, rangemax], alpha=0.8, label='charge($V_p$>3ADC)')
    ax.set_xlabel('Equivalent Charge/ADC$\cdot$ns')
    ax.set_ylabel('Entries')
    ax.legend(loc='best')
    ax.set_yscale('log')
    ax.set_xlim([-100, 600])
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    pdf.savefig(fig)
    ax.set_yscale('linear')
    ## 拟合峰值处所需参数
    zeroOffset = np.where(h[1]>=0)[0][0]
    ax.set_ylim([0, 2*np.max(h[0][(zeroOffset+70):(zeroOffset+150)])])
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
    ax.axvline(0.25*mu, linestyle='--', label='0.25$\mu_{C_1}$')
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
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label='G/1E7:{:.2f}$\pm${:.2f}'.format(results[j]['Gain'], np.sqrt(paraSigma2[j]['Gain']))))
    handles.append(mpatches.Patch(color='none', label='$\sigma_G$/1E7:{:.2f}$\pm${:.2f}'.format(results[j]['GainSigma'], np.sqrt(paraSigma2[j]['GainSigma']))))
    handles.append(mpatches.Patch(color='none', label='P/V:{:.2f}$\pm${:.2f}'.format(results[j]['PV'], np.sqrt(paraSigma2[j]['PV']))))
    handles.append(mpatches.Patch(color='none', label='$\mu_{select}$:'+'{:.2f}$\pm${:.2f}'.format(results[j]['chargeMu'], np.sqrt(paraSigma2[j]['chargeMu']))))
    handles.append(mpatches.Patch(color='none', label='$\sigma_{select}$'+':{:.2f}$\pm${:.2f}'.format(np.sqrt(results[j]['chargeSigma2']), np.sqrt(paraSigma2[j]['chargeSigma2'])/2/np.sqrt(results[j]['chargeSigma2']))))
    ax.legend(handles=handles)
    pdf.savefig(fig)
    plt.close()

    # peak分布
    fig, ax = plt.subplots()
    h = ax.hist(info[j]['minPeak'],histtype='step', bins=1000, range=[0,1000], label='peak')
    ax.hist(info[j]['minPeak'][(info[j]['minPeakCharge']>0.25*mu)], histtype='step', bins=1000, range=[0, 1000], alpha=0.8, label='peak($C_{\mathrm{equ}}$>0.25$\mu_{C_1}$)')
    print('peak height max:{};max index {}; part of peak {}'.format(np.max(h[0]), np.argmax(h[0]), h[0][:(np.argmax(h[0])+5)]))
    ax.set_xlabel('Peak/ADC')
    ax.set_ylabel('Entries')
    ax.legend()
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.set_yscale('log')
    pdf.savefig(fig)
    ## zoom in
    ax.axvline(3, linestyle='--', color='g', label='3ADC')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.set_yscale('linear')
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 2*np.max(h[0][5:30])])
    ax.legend()
    pdf.savefig(fig)
    ## 计算TriggerRate
    TriggerRate = results[j]['TriggerNum']/info[j].shape[0]
    results[j]['TriggerRate'] = TriggerRate
    paraSigma2[j]['TriggerRate'] = results[j]['TriggerNum']/info[j].shape[0]**2
    print('TriggerRate:{:.3f}'.format(TriggerRate))

    # min peak position分布
    fig, ax = plt.subplots()
    ax.set_title('peak position distribution')
    h = ax.hist(info[j]['minPeakPos'],histtype='step', bins=100, label='$t_{peak}-t_{trigger}$')
    print('h shape:{};max index {}'.format(h[0].shape,np.argmax(h[0])))
    ax.set_xlabel('$t_{peak}-t_{trigger}$/ns')
    ax.set_ylabel('entries')
    ax.legend()
    # pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)
    
    fig, ax = plt.subplots()
    ax.set_title('peak($V_p>3$mV) position distribution')
    h = ax.hist(info[j]['minPeakPos'][(info[j]['minPeak']>3)], histtype='step', bins=100, label='$t_{peak}-t_{trigger}$')
    print('h shape:{};max index {}'.format(h[0].shape,np.argmax(h[0])))
    ax.set_xlabel('$t_{peak}-t_{trigger}$/ns')
    ax.set_ylabel('entries')
    ax.legend()
    pdf.savefig(fig)
    ax.set_yscale('log')
    
    # risetime and downtime，里面对于范围做了限制，需要动态考虑
    results[['Rise', 'RiseSigma2', 'Fall', 'FallSigma2', 'FWHM', 'FWHMSigma2', 'TH', 'THSigma2']][j] = (
        np.mean(info[j]['riseTime'][totalselect]), np.var(info[j]['riseTime'][totalselect], ddof=1),
        np.mean(info[j]['downTime'][totalselect]), np.var(info[j]['downTime'][totalselect], ddof=1), 
        np.mean(info[j]['FWHM'][totalselect]), np.var(info[j]['FWHM'][totalselect], ddof=1), 
        np.mean((info[j]['down10']-info[j]['begin10'])[totalselect]), np.var((info[j]['down10']-info[j]['begin10'])[totalselect], ddof=1)
        )
    paraSigma2[['Rise', 'RiseSigma2', 'Fall', 'FallSigma2', 'FWHM', 'FWHMSigma2', 'TH', 'THSigma2']][j] = (
        results[j]['RiseSigma2']/results[j]['TriggerNum'],
        (centralMoment(info[j]['riseTime'][totalselect], results[j]['Rise'], 4) - ((results[j]['TriggerNum']-3)/(results[j]['TriggerNum']-2)*results[j]['RiseSigma2']**2))/results[j]['TriggerNum'],
        results[j]['FallSigma2']/results[j]['TriggerNum'],
        (centralMoment(info[j]['downTime'][totalselect], results[j]['Fall'], 4) - ((results[j]['TriggerNum']-3)/(results[j]['TriggerNum']-2)*results[j]['FallSigma2']**2))/results[j]['TriggerNum'],
        results[j]['FWHMSigma2']/results[j]['TriggerNum'],
        (centralMoment(info[j]['FWHM'][totalselect], results[j]['FWHM'], 4) - ((results[j]['TriggerNum']-3)/(results[j]['TriggerNum']-2)*results[j]['FWHMSigma2']**2))/results[j]['TriggerNum'],
        results[j]['THSigma2']/results[j]['TriggerNum'],
        (centralMoment((info[j]['down10']-info[j]['begin10'])[totalselect], results[j]['TH'], 4) - ((results[j]['TriggerNum']-3)/(results[j]['TriggerNum']-2)*results[j]['THSigma2']**2))/results[j]['TriggerNum'],
    )
    fig, ax = plt.subplots()
    ax.hist(info[j]['riseTime'][(totalselect)], histtype='step', bins=60, range=[0,30], label=r'Rise time:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.sqrt(results[j]['RiseSigma2']), results[j]['Rise']))
    ax.hist(info[j]['downTime'][(totalselect)], histtype='step', bins=60, range=[0,30], label=r'Fall time:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.sqrt(results[j]['FallSigma2']), results[j]['Fall']))
    ax.hist(info[j]['FWHM'][(totalselect)], histtype='step', bins=60, range=[0,30], label=r'FWHM:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.sqrt(results[j]['FWHMSigma2']), results[j]['FWHM']))
    ax.set_xlabel('Time/ns')
    ax.set_ylabel('Entries')
    ax.set_xlim([0, 30])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    # TTS 分布与unbinned拟合
    binwidth = 0.5
    fig, ax = plt.subplots()
    print(np.sum(totalselect&(~info[j]['isTrigger'])))
    ## 限制拟合区间，其中要求sigma不能超过15，从而限制最后的拟合区间最大为2*15
    limits_mu, limits_sigma = np.mean(info[j]['begin10'][(totalselect)]), np.std(info[j]['begin10'][totalselect])
    # 范围不能过大，否则会超过之前预分析的触发区间
    max_sigma = min(limits_mu - np.min(info[j]['begin10'][totalselect]), np.max(info[j]['begin10'][totalselect]) - limits_mu)
    limits_sigma = min(max(10 * limits_sigma, 5), max_sigma)
    limits = [limits_mu - limits_sigma, limits_mu + limits_sigma]
    tts = info[j]['begin10'][totalselect]
    tts_select = tts[(tts<limits[1]) & (tts>limits[0])]
    N = len(tts_select)
    b_u = min(30*2*limits_sigma*results[j]['TotalNum']/1e6/N, 0.5)
    print(limits, b_u)
    result = fitGausB(
        tts_select, 
        limits,
        2 * limits_sigma,
        b_u)# 给定background拟合上限30kHz
    tts_A, tts_mu, tts_sigma = result.x
    print(result)
    estimateDCR = (1-tts_A)*N/results[j]['TotalNum']/2/limits_sigma*1e6
    print('estimate DCR {}'.format(estimateDCR))
    # ROOT Fit for long interval
    h = ax.hist(tts_select, bins=int((limits[1]-limits[0])/binwidth), range=limits, histtype='step', label='$t^r_{10}-t_{\mathrm{trig}}$')
    rootfit.setHist(h[1], h[0])
    tts_edges = (h[1][1:] + h[1][:-1]) / 2
    tts_counts = h[0]
    tts_pi = np.where(tts_mu<=tts_edges)[0][0]
    t_n = int(limits_sigma/binwidth)
    rootfit.setFunc(ROOT.TF1("", "[0]*exp(-0.5*((x-[1])/[2])^2)+[3]+[4]*exp(-0.5*((x-[5])/[6])^2)", tts_edges[tts_pi-t_n], tts_edges[tts_pi+t_n]), np.array([*result.x, 10*results[j]['TotalNum']*binwidth/1e6, result.x[0]/100, result.x[1]-1, result.x[2]*3]))
    paraRoot, errorRoot = rootfit.Fit()
    if paraRoot[2]<paraRoot[6]:
        tts_para1, tts_para2 = (paraRoot[0], paraRoot[1], paraRoot[2]), (paraRoot[4], paraRoot[5], paraRoot[6])
        tts_error1, tts_error2 = (errorRoot[0], errorRoot[1], errorRoot[2]), (errorRoot[4], errorRoot[5], errorRoot[6])
    else:
        tts_para2, tts_para1 = (paraRoot[0], paraRoot[1], paraRoot[2]), (paraRoot[4], paraRoot[5], paraRoot[6])
        tts_error2, tts_error1 = (errorRoot[0], errorRoot[1], errorRoot[2]), (errorRoot[4], errorRoot[5], errorRoot[6])
    results[['TTS', 'TTS2', 'TT', 'TT2', 'TTA', 'TTA2']][j] = (tts_para1[2] * np.sqrt(2 * np.log(2)) * 2, tts_para2[2] * np.sqrt(2 * np.log(2)) * 2, tts_para1[1], tts_para2[1], tts_para1[0], tts_para1[0])
    paraSigma2[['TTS', 'TTS2', 'TT', 'TT2', 'TTA', 'TTA2']][j] = (tts_error1[2]**2 * 2 * np.log(2) * 4, tts_error2[2]**2 * 2 * np.log(2) * 4, tts_error1[1]**2, tts_error2[1]**2, tts_error1[0]**2, tts_error1[0]**2)
    probf1 = paraRoot[0] * np.exp(-(np.arange(limits[0], limits[1], 0.1) - paraRoot[1])**2/2/paraRoot[2]**2)
    probf2 = paraRoot[4] * np.exp(-(np.arange(limits[0], limits[1], 0.1) - paraRoot[5])**2/2/paraRoot[6]**2)
    ax.plot(np.arange(limits[0], limits[1], 0.1), probf1 + probf2 + paraRoot[3], label='Fit:{:.3f}$\pm${:.3f}'.format(paraRoot[2], errorRoot[2]))
    ax.plot(np.arange(limits[0], limits[1], 0.1), probf1 + paraRoot[3], linestyle='--', color='k', alpha=0.8, label='{:.3f} {:.3f}$\pm${:.3f}'.format(paraRoot[0], paraRoot[2], errorRoot[2]))
    ax.plot(np.arange(limits[0], limits[1], 0.1), probf2 + paraRoot[3], linestyle='--', color='k', alpha=0.8, label='{:.3f} {:.3f}$\pm${:.3f}'.format(paraRoot[4], paraRoot[6], errorRoot[6]))
    ax.set_xlabel('TT/ns')
    ax.set_ylabel('Entries')
    ax.set_xlim(limits)
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    ax.set_yscale('log')
    pdf.savefig(fig)
    plt.close()
    # TTS 分布与bin拟合
    fig, ax = plt.subplots()  
    l_range, r_range = int(tts_mu - 5*tts_sigma), int(tts_mu + 5*tts_sigma)+1
    h = ax.hist(info[j]['begin10'][totalselect], bins=int((r_range - l_range)/binwidth), range=[l_range, r_range], histtype='step', label='$t^r_{10}-t_{\mathrm{trig}}$')
    probf = (tts_A * np.exp(-(np.arange(l_range, r_range, 0.1) - tts_mu)**2/2/tts_sigma**2)/np.sqrt(2*np.pi)/tts_sigma + (1-tts_A)/ 2 / limits_sigma)
    ax.plot(np.arange(l_range, r_range, 0.1), N * binwidth * probf, '--')
    probf = (tts_A * np.exp(-(np.arange(limits[0], limits[1], 0.1) - tts_mu)**2/2/tts_sigma**2)/np.sqrt(2*np.pi)/tts_sigma + (1-tts_A)/ 2 / limits_sigma)
    ax.plot(np.arange(limits[0], limits[1], 0.1), N * binwidth * probf, label='fit')
    tts_edges = (h[1][1:] + h[1][:-1]) / 2
    tts_counts = h[0]
    tts_pi = np.where(tts_mu<=tts_edges)[0][0]
    ## 拟合区间缩至2ns
    t_n = int(2/binwidth)
    ttsResults = [minimize(peakResidual, [tts_counts[tts_pi], tts_edges[tts_pi], tts_sigma_init], 
        args=(tts_counts[(tts_pi-t_n):(tts_pi+t_n+1)], tts_edges[(tts_pi-t_n):(tts_pi+t_n+1)]),
        bounds=[(tts_counts[tts_pi]/10, None), (tts_edges[tts_pi]-2, tts_edges[tts_pi]+2), (0.1, 5)],
        method='SLSQP',
        options={'eps': 0.1})
        for tts_sigma_init in np.arange(0.2,1.5,0.05)]
    ttsResult = min(ttsResults, key=lambda x:x.fun)
    # ROOT Fit for short interval
    rootfit.setFunc(ROOT.TF1("", "[0]*exp(-0.5*((x-[1])/[2])^2)", tts_edges[tts_pi-t_n], tts_edges[tts_pi+t_n]), ttsResult.x)
    rootfit.setHist(h[1], h[0])
    paraRoot, errorRoot = rootfit.Fit()
    tts_A_bin, tts_mu_bin, tts_sigma_bin = paraRoot[0], paraRoot[1], paraRoot[2]
    results['TTS_bin'][j] = tts_sigma_bin * np.sqrt(2 * np.log(2)) * 2
    paraSigma2['TTS_bin'][j] = errorRoot[2]**2

    ax.plot(np.arange(tts_edges[tts_pi-t_n], tts_edges[tts_pi+t_n], 0.1), tts_A_bin * np.exp(-(np.arange(tts_edges[tts_pi-t_n], tts_edges[tts_pi+t_n], 0.1)-tts_mu_bin)**2/2/tts_sigma_bin**2), label='binned fit $\sigma$:{:.3f}$\pm{:.3f}$ns '.format(tts_sigma_bin, errorRoot[2]))
    ax.set_xlabel('TT/ns')
    ax.set_ylabel('Entries')
    ax.set_xlim([l_range, r_range])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    results[['DCR', 'TTSinterval']][j] = (estimateDCR, 2*limits_sigma)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label='$\sigma$={:.3f}ns'.format(tts_sigma)))
    handles.append(mpatches.Patch(color='none', label='TTS={:.3f}ns'.format(results[j]['TTS'])))
    ax.legend(handles=handles)
    print('tts:{:.3f}'.format(results[j]['TTS']))
    pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)
    plt.close()    

    searchRange_l, searchRange_r = np.floor(np.min(info[j]['begin10'][totalselect])), np.ceil(np.max(info[j]['begin10'][totalselect]))
    fig, ax = plt.subplots()
    ax.hist(info[j]['begin10'][totalselect], bins=np.arange(searchRange_l, searchRange_r, 0.5), histtype='step', label='$t^r_{10}-t_{\mathrm{trig}}$')
    ax.hist(info[j]['minPeakPos'][totalselect], bins=np.arange(searchRange_l, searchRange_r, 0.5), histtype='step', label='$t^p-t_{\mathrm{trig}}$')
    ax.axhline(0.5 * 10 * results[j]['TotalNum'] / 1e6, label='10kHz expectation')
    ax.axvline(rinterval[j][1])
    ax.axvline(rinterval[j][0])
    ax.set_xlabel('t/ns')
    ax.set_ylabel('Entries')
    ax.legend()
    ax.set_yscale('log')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    h = ax.hist2d(info[j]['minPeakPos'][totalselect], info[j]['riseTime'][totalselect], bins=[50,100], cmap=cmap)
    ax.set_ylabel('Rise time/ns')
    ax.set_xlabel('peak position/ns')
    fig.colorbar(h[3], ax=ax)
    pdf.savefig(fig)
    plt.close()
    # TTS-charge 2d分布
    fig, ax = plt.subplots()
    h = ax.hist2d(info[j]['begin10'][totalselect], info[j]['minPeakCharge'][totalselect], range=[[l_range, r_range], [0, 600]], bins=[(r_range-l_range)*int(1/binwidth), 600], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_ylabel('Equivalent Charge/ADC$\cdot$ns')
    ax.set_xlabel('TT/ns')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(info[j]['begin10'][totalselect], info[j]['minPeakCharge'][totalselect],
        range=[limits, [0, 600]], bins=[int(limits[1]-limits[0])*2, 600], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_ylabel('Equivalent Charge/ADC$\cdot$ns')
    ax.set_xlabel('TT/ns')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    h = ax.hist2d(info[j]['begin10'][totalselect&((info[j]['begin10']<(tts_mu-2*tts_sigma))|(info[j]['begin10']>(tts_mu+2*tts_sigma)))], info[j]['FWHM'][totalselect&((info[j]['begin10']<(tts_mu-2*tts_sigma))|(info[j]['begin10']>(tts_mu+2*tts_sigma)))],
        range=[[searchRange_l, searchRange_r], [0, 20]], bins=[int(searchRange_r - searchRange_l)*2, 200], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_ylabel('FWHM/ns')
    ax.set_xlabel('TT/ns')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    pdf.savefig(fig)

pdf.close()
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('res',data=results, compression='gzip')
    opt.create_dataset('resSigma2', data=paraSigma2, compression='gzip')