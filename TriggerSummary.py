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
from waveana.util import peakResidual, vallyResidual, fitGaus
ADC2mV = config.ADC2mV
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
        ('peakC','<f4'), ('vallyC','<f4'), ('PV','<f4'),
        ('chargeMu','<f4'), ('chargeSigma','<f4'),
        ('Gain', '<f4'), ('GainSigma', '<f4'),
        ('TriggerRate', '<f4'), ('TotalNum', '<i4'), ('window', '<f4'),
        ('TTS', '<f4'),
        ('Rise', '<f4'), ('Fall', '<f4'), ('TH', '<f4'), ('FWHM', '<f4'),
        ('RiseSigma', '<f4'), ('FallSigma', '<f4'), ('THSigma', '<f4'), ('FWHMSigma', '<f4')
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
peakspanl, peakspanr = config.peakspanl, config.peakspanr
vallyspanl, vallyspanr = config.vallyspanl, config.vallyspanr
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
    ax.hist(info[j]['minPeakCharge'][(info[j]['minPeak']>3)], histtype='step', bins=bins, range=[rangemin, rangemax], alpha=0.8, label='charge($V_p>3ADC$)')
    ax.set_xlabel('Equivalent Charge/ADCns')
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
    # 使用上面值作为初值进行最小二乘拟合
    hi = zeroOffset + 15 + vi_r + pi_r
    result = minimize(peakResidual, [pv, pi, 30], 
        args=(h[0][(hi-peakspanl):(hi+peakspanr)], (h[1][(hi-peakspanl):(hi+peakspanr)]+h[1][(hi-peakspanl+1):(hi+peakspanr+1)])/2),
        bounds=[(0,None), (5, None), (0, None)],
        method='SLSQP', options={'eps': 0.1})
    print(result)
    A, mu, sigma = result.x
    ## 绘制拟合结果
    ax.plot(h[1][(hi-peakspanl):(hi+peakspanr)], 
        A*np.exp(-(h[1][(hi-peakspanl):(hi+peakspanr)]-mu)**2/2/sigma**2), color='r', label='peak fit')
    pi = int(mu)
    pv = A
    ax.fill_betweenx([0, pv], h[1][hi-peakspanl], h[1][hi+peakspanr], alpha=0.5, color='lightsalmon', label='peak fit interval')
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 2*pv])
    ax.axvline(0.25*mu, linestyle='--', label='0.25p.e.')
    ## 拟合峰谷处所需参数,smooth不是必须的，polyfit不能进行MLE拟合
    li = zeroOffset + 15 + vi_r
    yy  = h[0][(li-vallyspanl):(li+vallyspanr)]
    ## 对vally进行区间调整，放置左侧padding过长
    while (yy[0] > 3 * yy[-1]) and vallyspanl>1:
        vallyspanl = vallyspanl // 2
        yy  = h[0][(li-vallyspanl):(li+vallyspanr)]
    result = minimize(vallyResidual, [0.3, vi, vv + 5], args=(yy, (h[1][(li-vallyspanl):(li+vallyspanr)] + h[1][(li-vallyspanl+1):(li+vallyspanr+1)])/2),
        bounds=[(0.1, None), (vi-5, vi+5), (5, A)],
        method='SLSQP', options={'eps': 0.1, 'maxiter':5000})
    print(result)
    a_v, b_v, c_v = result.x
    ax.plot(h[1][(li-vallyspanl):(li+vallyspanr)], a_v/100 * (h[1][(li-vallyspanl):(li+vallyspanr)] - b_v)**2 +c_v, color='g', label='vally fit')
    vi, vv = b_v, c_v
    ax.fill_betweenx([0, vv], h[1][li-vallyspanl], h[1][li+vallyspanr], alpha=0.5, color='lightgreen', label='vally fit interval')
    ax.scatter([pi,vi], [pv,vv], color='r')
    ## 将参数放入legend里
    selectinfo = info[j]['minPeakCharge'][(info[j]['minPeak']>3)&(info[j]['minPeakCharge']<800)&(info[j]['minPeakCharge']>0.25*mu)]
    results[j] = (args.channel[j], mu, vi, pv / vv,np.mean(selectinfo), np.std(selectinfo), mu/50/1.6*ADC2mV, sigma/50/1.6*ADC2mV, 0, len(info[j]), rinterval[j][1] - rinterval[j][0],
        0,
        0, 0, 0, 0, 0, 0, 0, 0)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label='G/1E7:{:.2f}'.format(mu/50/1.6*ADC2mV)))
    handles.append(mpatches.Patch(color='none', label='$\sigma_G$/1E7:{:.2f}'.format(sigma/50/1.6*ADC2mV)))
    handles.append(mpatches.Patch(color='none', label='P/V:{:.2f}'.format(pv/vv)))
    handles.append(mpatches.Patch(color='none', label='$\mu_{select}$:'+'{:.2f}'.format(results[j]['chargeMu'])))
    handles.append(mpatches.Patch(color='none', label='$\sigma_{select}$'+':{:.2f}'.format(results[j]['chargeSigma'])))
    ax.legend(handles=handles)
    pdf.savefig(fig)
    plt.close()

    # peak分布
    fig, ax = plt.subplots()
    h = ax.hist(info[j]['minPeak'],histtype='step', bins=1000, range=[0,1000], label='peak')
    ax.hist(info[j]['minPeak'][(info[j]['minPeakCharge']>0.25*mu)], histtype='step', bins=1000, range=[0, 1000], alpha=0.8, label='peak($C>0.25 p.e.$)')
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
    totalselect = (info[j]['minPeak']>3)&(info[j]['minPeakCharge']>0.25*mu)&info[j]['isTrigger']
    TriggerRate = np.sum(totalselect)/info[j].shape[0]
    results[j]['TriggerRate'] = TriggerRate
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
    fig, ax = plt.subplots()
    ## ax.set_title('$T_R$,$T_d$,FWHM ($V_p>3$mV) distribution')
    ax.hist(info[j]['riseTime'][(totalselect)], histtype='step', bins=300, range=[0,30], label=r'risingtime:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.std(info[j]['riseTime'][totalselect]), np.mean(info[j]['riseTime'][totalselect])))
    ax.hist(info[j]['downTime'][(totalselect)], histtype='step', bins=300, range=[0,30], label=r'falltime:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.std(info[j]['downTime'][totalselect]), np.mean(info[j]['downTime'][totalselect])))
    ax.hist(info[j]['FWHM'][(totalselect)], histtype='step', bins=300, range=[0,30], label=r'FWHM:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.std(info[j]['FWHM'][totalselect]), np.mean(info[j]['FWHM'][totalselect])))
    ax.set_xlabel('Time/ns')
    ax.set_ylabel('Entries')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    pdf.savefig(fig)
    plt.close()
    results[['Rise', 'RiseSigma', 'Fall', 'FallSigma', 'FWHM', 'FWHMSigma', 'TH', 'THSigma']][j] = (
        np.mean(info[j]['riseTime'][totalselect]), np.std(info[j]['riseTime'][totalselect]),
        np.mean(info[j]['downTime'][totalselect]), np.std(info[j]['downTime'][totalselect]), 
        np.mean(info[j]['FWHM'][totalselect]), np.std(info[j]['FWHM'][totalselect]), 
        np.mean((info[j]['down10']-info[j]['begin10'])[totalselect]), np.std((info[j]['down10']-info[j]['begin10'])[totalselect])
        )

    # TTS 分布与拟合
    fig, ax = plt.subplots()
    print(np.sum(totalselect&(~info[j]['isTrigger'])))
    ## 限制拟合区间，其中要求sigma不能超过15，从而限制最后的拟合区间最大为2*15
    limits_mu, limits_sigma = np.mean(info[j]['begin10'][(totalselect)]), np.std(info[j]['begin10'][totalselect])
    limits_sigma = min(limits_sigma, 3)
    limits = [limits_mu - limits_sigma, limits_mu + limits_sigma]
    print(limits)
    result, N = fitGaus(info[j]['begin10'][totalselect], limits)
    tts_A, tts_mu, tts_sigma = result.x
    print(result)
    l_range, r_range = int(tts_mu - 5*tts_sigma), int(tts_mu + 5*tts_sigma)+1
    # l_range, r_range = int(limits_mu - 1*limits_sigma), int(limits_mu + 1*limits_sigma)
    binwidth = 0.1
    ax.hist(info[j]['begin10'][totalselect], bins=int((r_range - l_range)/binwidth), range=[l_range, r_range], histtype='step', label='$t^r_{10}-t_{\mathrm{trig}}$')
    ax.plot(np.arange(l_range, r_range, 0.1), result.x[0] * N * binwidth * np.exp(-(np.arange(l_range, r_range, 0.1) - result.x[1])**2/2/result.x[2]**2)/np.sqrt(2*np.pi)/result.x[2], '--')
    ax.plot(np.arange(limits[0], limits[1], 0.1), result.x[0] * N * binwidth * np.exp(-(np.arange(limits[0], limits[1], 0.1) - result.x[1])**2/2/result.x[2]**2)/np.sqrt(2*np.pi)/result.x[2], label='fit')
    ax.set_xlabel('TT/ns')
    ax.set_ylabel('Entries')
    ax.set_xlim([l_range, r_range])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.legend()
    TTS = result.x[2]*np.sqrt(2*np.log(2))*2
    results[j]['TTS'] = TTS
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label='$\sigma$={:.3f}ns'.format(result.x[2])))
    handles.append(mpatches.Patch(color='none', label='TTS={:.3f}ns'.format(TTS)))
    ax.legend(handles=handles)
    print('tts:{:.3f}'.format(TTS))
    pdf.savefig(fig)
    plt.close()
    # TTS-charge 2d分布
    fig, ax = plt.subplots()
    
    h = ax.hist2d(info[j]['begin10'][totalselect], info[j]['minPeakCharge'][totalselect], range=[[l_range, r_range], [0, 600]], bins=[(r_range-l_range)*10, 600], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_ylabel('Equivalent Charge/ADCns')
    ax.set_xlabel('TT/ns')
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    pdf.savefig(fig)
pdf.close()
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('res',data=results, compression='gzip')
