'''
Noise stage summary.
Run example:
python3 BasicSummary.py -c 2 3 4 5 -i ExResult/697/0ns/h5/preanalysisMerge.h5 -o ExResult/697/0ns/charge.h5 > ExResult/697/0ns/charge.h5.log 2>&1
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
from waveana.util import peakResidual, vallyResidual, Hessian, centralMoment, RootFit
import ROOT
from array import array
ADC2mV = config.ADC2mV
rootfit = RootFit()
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-t', dest='trigger', default=-1, type=int, help='trigger channel')
args = psr.parse_args()

info = []
results = np.zeros(len(args.channel),
    dtype=[
        ('Channel', '<i2'),
        ('peakC','<f4'), ('vallyC','<f4'), ('peakV','<f4'), ('vallyV','<f4'), ('PV','<f4'),
        ('chargeMu','<f4'), ('chargeSigma2','<f4'),
        ('Gain', '<f4'), ('GainSigma', '<f4'),
        ('DCR', '<f4'), ('TotalNum', '<i4'), ('TriggerNum', '<i4'),
        ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'),
        ('RiseSigma2', '<f4'), ('FallSigma2', '<f4'), ('THSigma2', '<f4'), ('FWHMSigma2', '<f4')
    ])
paraSigma2 = np.zeros(len(args.channel),
    dtype=[
        ('Channel', '<i2'),
        ('peakC','<f4'), ('vallyC','<f4'), ('peakV','<f4'), ('vallyV','<f4'), ('peakS','<f4'), ('vallyS','<f4'), ('PV','<f4'),
        ('chargeMu','<f4'), ('chargeSigma2','<f4'),
        ('Gain', '<f4'), ('GainSigma', '<f4'),
        ('DCR', '<f4'), ('TotalNum', '<i4'), ('TriggerNum', '<i4'),
        ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'),
        ('RiseSigma2', '<f4'), ('FallSigma2', '<f4'), ('THSigma2', '<f4'), ('FWHMSigma2', '<f4')
    ])
with h5py.File(args.ipt, 'r') as ipt:
    waveformLength = ipt.attrs['waveformLength']
    for j in range(len(args.channel)):
        info.append(ipt['ch{}'.format(args.channel[j])][:])
    if args.trigger>=0:
        triggerInfo = ipt['trigger'][:]
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
if args.trigger>=0:
    # 绘制trigger的分布图
    fig, ax = plt.subplots()
    trigger_mu, trigger_sigma = np.average(triggerInfo['triggerTime']), np.std(triggerInfo['triggerTime'])
    ax.hist(triggerInfo['triggerTime'], histtype='step', bins=20*int(trigger_sigma), range=[int(trigger_mu-10*trigger_sigma), int(trigger_mu+10*trigger_sigma)],
        label='trigger time {:.2f}$\pm${:.2f}'.format(trigger_mu, trigger_sigma))
    ax.set_xlabel('time/ns')
    ax.set_ylabel('entries')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close()
# 下面循环绘制每个channel的图像
nearMax = 10

for j in range(len(args.channel)):
    rangemin = int(np.min(info[j]['minPeakCharge'])-1)
    rangemax = int(np.max(info[j]['minPeakCharge'])+1)
    bins = rangemax-rangemin
    if (rangemax - rangemin) < 100:
        print('Warning: no signal charge more than 100')
        continue
    selectNearMax = info[j]['nearPosMax']<=nearMax
    # charge分布
    fig, ax = plt.subplots()
    h = ax.hist(info[j]['minPeakCharge'][selectNearMax], histtype='step', bins=bins, range=[rangemin, rangemax], label='charge')
    ax.hist(info[j]['minPeakCharge'][selectNearMax&(info[j]['minPeak']>3)], histtype='step', bins=bins, range=[rangemin, rangemax], alpha=0.8, label='charge($V_p>3$ADC)')
    ax.set_xlabel('Equivalent Charge/ADC$\cdot$ns')
    ax.set_ylabel('Entries')
    ax.legend(loc='best')
    ax.set_yscale('log')
    pdf.savefig(fig)
    ## zoom in
    ax.set_xlim([-100, 600])
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    pdf.savefig(fig)
    ## 拟合峰值处所需参数
    ax.set_yscale('linear')
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
    ## 使用上面值作为初值进行最小二乘拟合
    hi = zeroOffset + 15 + vi_r + pi_r
    result = minimize(peakResidual, [pv, pi, 30], 
        args=(h[0][(hi-peakspanl):(hi+peakspanr)], (h[1][(hi-peakspanl):(hi+peakspanr)]+h[1][(hi-peakspanl+1):(hi+peakspanr+1)])/2),
        bounds=[(0,None), (5, None), (0, None)],
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
    ## python 求二阶导计算误差
    resultSigma2 = np.linalg.pinv(Hessian(lambda x: peakResidual(x, h[0][(hi-peakspanl):(hi+peakspanr)], (h[1][(hi-peakspanl):(hi+peakspanr)]+h[1][(hi-peakspanl+1):(hi+peakspanr+1)])/2), result.x, step=None))
    print(resultSigma2)
    ## ROOT fit
    rootfit.setFunc(ROOT.TF1("", "[0]*exp(-0.5*((x-[1])/[2])^2)", h[1][hi - peakspanl], h[1][hi + peakspanr]), result.x)
    rootfit.setHist(h[1], h[0])
    paraRoot, errorRoot = rootfit.Fit()
    print((paraRoot[0], paraRoot[1], paraRoot[2]), (errorRoot[0], errorRoot[1], errorRoot[2]))

    # paraSigma2[['Channel', 'peakC', 'peakV', 'peakS']][j] = (args.channel[j], resultSigma2[1,1], resultSigma2[0,0], resultSigma2[2,2])
    # A, mu, sigma = result.x
    A, mu, sigma = paraRoot[0], paraRoot[1], paraRoot[2]
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
    ## 拟合峰谷处所需参数,smooth不是必须的;polyfit包不能进行MLE拟合,所以放弃polyfit包，手动实现MLE
    li = zeroOffset + 15 + vi_r
    yy  = h[0][(li-vallyspanl):(li+vallyspanr)]
    print(yy)
    ## 对vally进行区间调整，防止左侧padding过长
    while (yy[0] > 1.5 * np.max(yy[-10:])) and vallyspanl>2:
        vallyspanl = vallyspanl - 1
        yy  = h[0][(li-vallyspanl):(li+vallyspanr)]
    print(vallyspanl)
    result = minimize(vallyResidual, [0.3, vi, vv+10], args=(yy, (h[1][(li-vallyspanl):(li+vallyspanr)] + h[1][(li-vallyspanl+1):(li+vallyspanr+1)])/2),
        bounds=[(0.1, None), (vi-5, vi+5), (0, A)],
        method='SLSQP', options={'eps': 0.1, 'maxiter':5000})
    print(result)
    ## abandon(使用拟合值为初值，降低步长再次拟合)
    # vv, vi = result.x[2], result.x[1]
    # li = zeroOffset + int(vi)
    # result = minimize(vallyResidual, result.x, args=(yy, (h[1][(li-vallyspanl):(li+vallyspanr)] + h[1][(li-vallyspanl+1):(li+vallyspanr+1)])/2),
    #     method='BFGS')
    # print(result)
    # print(np.linalg.pinv(np.array(result.hess_inv)))
    # resultSigma2 = np.linalg.pinv(Hessian(lambda x: vallyResidual(x, yy, (h[1][(li-vallyspanl):(li+vallyspanr)] + h[1][(li-vallyspanl+1):(li+vallyspanr+1)])/2), result.x, step=None))
    # print(resultSigma2)
    ## ROOT fit
    rootfit.setFunc(ROOT.TF1("", "[0]*(x-[1])^2+[2]", h[1][li-vallyspanl], h[1][li+vallyspanr]), result.x)
    rootfit.func.SetParLimits(0, 0.001, 100000)
    rootfit.func.SetParLimits(1, 5, pi)
    rootfit.func.SetParLimits(2, 1, A)
    rootfit.setHist(h[1], h[0])
    paraRoot, errorRoot = rootfit.Fit()
    print((paraRoot[0], paraRoot[1], paraRoot[2]), (errorRoot[0], errorRoot[1], errorRoot[2]))
    ## 计算误差
    # paraSigma2[['vallyC', 'vallyV', 'vallyS']][j] = (resultSigma2[1,1], resultSigma2[2,2], resultSigma2[0,0])
    # a_v, b_v, c_v = result.x
    a_v, b_v, c_v = paraRoot[0]*100, paraRoot[1], paraRoot[2]
    paraSigma2[['vallyC', 'vallyV', 'vallyS']][j] = (errorRoot[1]**2, errorRoot[2]**2, errorRoot[0]**2)
    ax.plot(h[1][(li-vallyspanl):(li+vallyspanr)], a_v/100 * (h[1][(li-vallyspanl):(li+vallyspanr)] - b_v)**2 +c_v, color='g', label='vally fit')
    vi, vv = b_v, c_v
    ax.fill_betweenx([0, vv], h[1][li-vallyspanl], h[1][li+vallyspanr], alpha=0.5, color='lightgreen', label='vally fit interval')
    ax.scatter([pi,vi], [pv,vv], color='r')
    ## 将参数放入legend里
    totalselect = selectNearMax&(info[j]['minPeak']>3)&(info[j]['minPeakCharge']<1000)&(info[j]['minPeakCharge']>0.25*mu)
    selectinfo = info[j]['minPeakCharge'][totalselect]
    results[j] = (
        args.channel[j], mu, vi, pv, vv, pv / vv, np.mean(selectinfo), np.var(selectinfo, ddof=1), mu/50/1.6*ADC2mV, sigma/50/1.6*ADC2mV, 0, len(info[j]), np.sum(totalselect),
        0, 0, 0, 0, 0, 0, 0, 0)# DCR 一项占位0
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
    
    # peak分布## 绘制筛选后的结果
    fig, ax = plt.subplots()
    h = ax.hist(info[j]['minPeak'][selectNearMax],histtype='step', bins=1000, range=[0, 1000], label='peak')
    ax.hist(info[j]['minPeak'][selectNearMax&(info[j]['minPeakCharge']>0.25*mu)], histtype='step', bins=1000, range=[0, 1000], alpha=0.8, label='peak($C_{\mathrm{equ}}$>0.25$\mu_{C_1}$)')
    print('peak height max:{};max index {}; part of peak {}'.format(np.max(h[0]), np.argmax(h[0]), h[0][:(np.argmax(h[0])+5)]))
    ax.set_xlabel('Peak/ADC')
    ax.set_ylabel('Entries')
    ax.legend()
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.set_yscale('log')
    pdf.savefig(fig)
    ## 计算DCR/kHz
    DCR = results[j]['TriggerNum']/np.sum(selectNearMax)/waveformLength*1e6
    results[j]['DCR'] = DCR
    paraSigma2[j]['DCR'] = results[j]['TriggerNum']/(np.sum(selectNearMax)*waveformLength/1e6)**2
    print('DCR:{:.2f}'.format(DCR))
    ## zoom in
    ax.axvline(3, linestyle='--', color='g', label='3ADC')
    ax.set_xlim([0, 50])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    pdf.savefig(fig)
    ax.set_yscale('linear')
    ax.set_ylim([0, 2*np.max(h[0][5:30])])
    pdf.savefig(fig)
    plt.close()

    # min peak position分布
    fig, ax = plt.subplots()
    ax.set_title('peak position distribution')
    h = ax.hist(info[j]['minPeakPos'][selectNearMax],histtype='step', bins=waveformLength, range=[0,waveformLength], label='the largest peak pos')
    print('h shape:{};max index {}'.format(h[0].shape,np.argmax(h[0])))
    ax.set_xlabel('peak position/ns')
    ax.set_ylabel('entries')
    ax.legend()
    # pdf.savefig(fig)
    ax.set_yscale('log')
    pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title('peak($V_p>5$mV) position distribution')
    h = ax.hist(info[j]['minPeakPos'][(info[j]['minPeak']>5)&(selectNearMax)],histtype='step', bins=waveformLength, range=[0,waveformLength], label='the largest peak pos')
    print('h shape:{};max index {}'.format(h[0].shape,np.argmax(h[0])))
    ax.set_xlabel('peak position/ns')
    ax.set_ylabel('entries')
    ax.legend()
    pdf.savefig(fig)
    ax.set_yscale('log')
    # pdf.savefig(fig)
    plt.close()

    # min peak position and peak height distribution
    fig, ax = plt.subplots()
    ax.set_title('peakPos-peakHeight')
    h = ax.hist2d(info[j]['minPeakPos'][selectNearMax],info[j]['minPeak'][selectNearMax],range=[[0,waveformLength],[0,50]], bins=[int(waveformLength), 100], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('peakPos/ns')
    ax.set_ylabel('peakHeight/mV')
    pdf.savefig(fig)
    plt.close()

    # min peak charge and peak height distribution
    fig, ax = plt.subplots()
    ax.set_title('peakCharge-peakHeight')
    h = ax.hist2d(info[j]['minPeakCharge'][selectNearMax],info[j]['minPeak'][selectNearMax],range=[[0,400],[0,50]], bins=[400, 50], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('peakCharge/mVns')
    ax.set_ylabel('peakHeight/mV')
    pdf.savefig(fig)
    plt.close()
    '''
    fig, ax = plt.subplots()
    ax.set_title('peakCharge-peakHeight')
    h = ax.hist2d(info[j]['minPeakCharge'],info[j]['minPeak'],range=[[0,4500],[0,1000]], bins=[450, 100],norm=colors.LogNorm(), cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('peakCharge/nsmV')
    ax.set_ylabel('peakHeight/mV')
    # plt.savefig('{}/{}peakCharge-peakHeightLog.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    
    fig, ax = plt.subplots()
    ax.set_title('peakCharge-peakHeight')
    h = ax.hist2d(info[j]['minPeakCharge'],info[j]['minPeak'],range=[[35,4000],[3,998]], bins=[793, 199], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('peakCharge/nsmV')
    ax.set_ylabel('peakHeight/mV')
    # plt.savefig('{}/{}peakCharge-peakHeightCut1.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)

    fig, ax = plt.subplots()
    ax.set_title('peakCharge-peakHeight')
    h = ax.hist2d(info[j]['minPeakCharge'],info[j]['minPeak'],range=[[35,1000],[3,75]], bins=[965, 72], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('peakCharge/nsmV')
    ax.set_ylabel('peakHeight/mV')
    # plt.savefig('{}/{}peakCharge-peakHeightCut.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    '''

    # baseline and std distribution
    fig, ax = plt.subplots()
    ax.set_title('baseline-std')
    h = ax.hist2d(info[j]['baseline'][selectNearMax],info[j]['std'][selectNearMax], bins=[100,100], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('baseline/mV')
    ax.set_ylabel('std/mV')
    pdf.savefig(fig)
    plt.close()

    # baseline
    fig, ax = plt.subplots()
    ax.set_title('baseline')
    ax.hist(info[j]['baseline'][selectNearMax],histtype='step',bins=100, label='baseline')
    ax.set_xlabel('baseline/mV')
    ax.set_ylabel('entries')
    ax.set_yscale('log')
    ax.legend()
    pdf.savefig(fig)
    plt.close()

    # std
    fig, ax = plt.subplots()
    ax.set_title('std distribution')
    ax.hist(info[j]['std'][selectNearMax],histtype='step', bins=100, label='std')
    ax.set_xlabel('std/mV')
    ax.set_ylabel('entries')
    ax.legend()
    ax.set_yscale('log')
    pdf.savefig(fig)
    plt.close()
    
    # risetime and falltime，里面对于范围做了限制，需要动态考虑
    position = (info[j]['minPeakPos']>config.baselength)&(info[j]['minPeakPos']<(waveformLength-config.afterlength))
    totalselectAndPos = totalselect & position
    results[['Rise', 'RiseSigma2', 'Fall', 'FallSigma2', 'FWHM', 'FWHMSigma2', 'TH', 'THSigma2']][j] = (
        np.mean(info[j]['riseTime'][totalselectAndPos]), np.var(info[j]['riseTime'][totalselectAndPos], ddof=1),
        np.mean(info[j]['downTime'][totalselectAndPos]), np.var(info[j]['downTime'][totalselectAndPos], ddof=1),
        np.mean(info[j]['FWHM'][totalselectAndPos]), np.var(info[j]['FWHM'][totalselectAndPos], ddof=1),
        np.mean((info[j]['end10']-info[j]['begin10'])[totalselectAndPos]), np.var((info[j]['end10']-info[j]['begin10'])[totalselectAndPos], ddof=1)
        )
    selectNum = np.sum(totalselectAndPos)
    paraSigma2[['Rise', 'RiseSigma2', 'Fall', 'FallSigma2', 'FWHM', 'FWHMSigma2', 'TH', 'THSigma2']][j] = (
        results[j]['RiseSigma2']/selectNum,
        (centralMoment(info[j]['riseTime'][totalselectAndPos], results[j]['Rise'], 4) - ((selectNum-3)/(selectNum-2)*results[j]['RiseSigma2']**2))/selectNum,
        results[j]['FallSigma2']/selectNum,
        (centralMoment(info[j]['downTime'][totalselectAndPos], results[j]['Fall'], 4) - ((selectNum-3)/(selectNum-2)*results[j]['FallSigma2']**2))/selectNum,
        results[j]['FWHMSigma2']/selectNum,
        (centralMoment(info[j]['FWHM'][totalselectAndPos], results[j]['FWHM'], 4) - ((selectNum-3)/(selectNum-2)*results[j]['FWHMSigma2']**2))/selectNum,
        results[j]['THSigma2']/selectNum,
        (centralMoment((info[j]['end10']-info[j]['begin10'])[totalselectAndPos], results[j]['TH'], 4) - ((selectNum-3)/(selectNum-2)*results[j]['THSigma2']**2))/selectNum,
    )
    fig, ax = plt.subplots()
    ax.hist(info[j]['riseTime'][totalselect&position], histtype='step', bins=60, range=[0,30], label=r'Rise time:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.sqrt(results[j]['RiseSigma2']), results[j]['Rise']))
    ax.hist(info[j]['downTime'][totalselect&position], histtype='step', bins=60, range=[0,30], label=r'Fall time:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.sqrt(results[j]['FallSigma2']), results[j]['Fall']))
    ax.hist(info[j]['FWHM'][totalselect&position], histtype='step', bins=60, range=[0,30], label=r'FWHM:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.sqrt(results[j]['FWHMSigma2']), results[j]['FWHM']))
    ax.set_xlabel('Time/ns')
    ax.set_ylabel('Entries')
    ax.set_xlim([0, 30])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    pdf.savefig(fig)
    plt.close()
    
    fig, ax = plt.subplots()
    ax.set_title('$T_R$,$T_d$,FWHM ($5<V_p<40$mV) distribution')
    ax.hist(info[j]['riseTime'][(info[j]['minPeak']>5)&(info[j]['minPeak']<40)&selectNearMax],histtype='step',bins=300, range=[0,30], label='risingtime')
    ax.hist(info[j]['downTime'][(info[j]['minPeak']>5)&(info[j]['minPeak']<40)&selectNearMax],histtype='step',bins=300, range=[0,30], label='downtime')
    ax.hist(info[j]['FWHM'][(info[j]['minPeak']>5)&(info[j]['minPeak']<40)&(selectNearMax)],histtype='step',bins=300, range=[0,30], label='FWHM')
    ax.set_xlabel('Time/ns')
    ax.set_ylabel('entries')
    ax.legend()
    pdf.savefig(fig)
    plt.close()
    

    '''
    # 半高宽分布
    fig, ax = plt.subplots()
    ax.set_title('FWHM ($V_p>5$mV) distribution')
    ax.hist(info[j]['FWHM'][(info[j]['minPeak']>5)&(selectNearMax)],histtype='step',bins=400, range=[0,40], label='FWHM')
    ax.set_xlabel('FWHM/ns')
    ax.set_ylabel('entries')
    ax.legend()
    #ax.set_xlim([1,40])
    # plt.savefig('{}/{}FWHM.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    # ax.set_yscale('log')
    # plt.savefig('{}/{}FWHMLog.png'.format(args.opt,args.channel[j]))
    # pdf.savefig(fig)
    
    fig, ax = plt.subplots()
    ax.set_title('FWHM distribution')
    ax.hist(info[j]['FWHM'][(info[j]['minPeak']>5)&(info[j]['minPeak']<40)],histtype='step',bins=40, range=[0,40], label='FWHM')
    ax.set_xlabel('FWHM/ns')
    ax.set_ylabel('entries')
    ax.legend()
    #ax.set_xlim([1,40])
    # plt.savefig('{}/{}FWHMUlimit.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    ax.set_yscale('log')
    # plt.savefig('{}/{}FWHMUlimitLog.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    '''
    '''
    fig, ax = plt.subplots()
    ax.set_title('FWHM-charge')
    h = ax.hist2d(info[j]['minPeakCharge'][(info[j]['minPeak']>5)],info[j]['FWHM'][(info[j]['minPeak']>5)], bins=[1000,20], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('charge/mVns')
    ax.set_ylabel('FWHM/ns')
    # plt.savefig('{}/{}FWHM-charge.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    
    fig, ax = plt.subplots()
    ax.set_title('FWHM-charge')
    h = ax.hist2d(info[j]['minPeakCharge'][(info[j]['minPeak']>5)&(info[j]['minPeakCharge']<8000)],info[j]['FWHM'][(info[j]['minPeak']>5)&(info[j]['minPeakCharge']<8000)], bins=[1000,20], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('charge/mVns')
    ax.set_ylabel('FWHM/ns')
    # plt.savefig('{}/{}FWHM-chargeCut.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    
    fig, ax = plt.subplots()
    ax.set_title('tot distribution')
    ax.hist(info[j][info[j]['minPeak']>5]['end5mV']-info[j][info[j]['minPeak']>5]['begin5mV'],histtype='step',bins=40, range=[0,40], label='tot')
    ax.set_xlabel('tot/ns')
    ax.set_ylabel('entries')
    ax.legend()
    #ax.set_xlim([1,40])
    # plt.savefig('{}/{}tot.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    ax.set_yscale('log')
    # plt.savefig('{}/{}totLog.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    
    fig, ax = plt.subplots()
    ax.set_title('charge-tot')
    h = ax.hist2d(info[j]['minPeakCharge'][(info[j]['minPeak']>5)],info[j][(info[j]['minPeak']>5)]['end5mV']-info[j][info[j]['minPeak']>5]['begin5mV'], bins=[1000,30], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('charge/mVns')
    ax.set_ylabel('tot/ns')
    # plt.savefig('{}/{}tot-charge.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    
    fig, ax = plt.subplots()
    ax.set_title('charge-tot charge<{}'.format(1000))
    h = ax.hist2d(info[j]['minPeakCharge'][(info[j]['minPeak']>5)&(info[j]['minPeakCharge']<1000)],info[j][(info[j]['minPeak']>5)&(info[j]['minPeakCharge']<1000)]['end5mV']-info[j][(info[j]['minPeak']>5)&(info[j]['minPeakCharge']<1000)]['begin5mV'], bins=[200,30], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('charge/mVns')
    ax.set_ylabel('tot/ns')
    # plt.savefig('{}/{}tot-chargeCut.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    '''
pdf.close()
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('res', data=results, compression='gzip')
    opt.create_dataset('resSigma2', data=paraSigma2, compression='gzip')
