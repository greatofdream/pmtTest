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
from waveana.util import peakResidual, vallyResidual
ADC2mV = config.ADC2mV

psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-t', dest='trigger', default=-1, type=int, help='trigger channel')
args = psr.parse_args()

info = []
results = np.zeros(len(args.channel),
    dtype=[
        ('peakC','<f4'), ('vallyC','<f4'), ('PV','<f4'),
        ('chargeMu','<f4'), ('chargeSigma','<f4'),
        ('Gain', '<f4'), ('GainSigma', '<f4'),
        ('DCR', '<f4')
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
# 下面循环绘制每个channel的图像
nearMax = 10
peakspanl, peakspanr = config.peakspanl, config.peakspanr
vallyspanl, vallyspanr = config.vallyspanl, config.vallyspanr
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
    ax.hist(info[j]['minPeakCharge'][selectNearMax&(info[j]['minPeak']>3)], histtype='step', bins=bins, range=[rangemin, rangemax], alpha=0.8, label='charge($V_p>3ADC$)')
    ax.set_xlabel('Equivalent Charge/ADCns')
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
    ## 使用上面值作为初值进行最小二乘拟合
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
    result = minimize(vallyResidual, [0.3, vi, vv+10], args=(yy, (h[1][(li-vallyspanl):(li+vallyspanr)] + h[1][(li-vallyspanl+1):(li+vallyspanr+1)])/2),
        bounds=[(0.1, None), (vi-5, vi+5), (16, A)],
        method='SLSQP', options={'eps': 0.1, 'maxiter':5000})
    print(result)
    a_v, b_v, c_v = result.x
    ax.plot(h[1][(li-vallyspanl):(li+vallyspanr)], a_v/100 * (h[1][(li-vallyspanl):(li+vallyspanr)] - b_v)**2 +c_v, color='g', label='vally fit')
    vi, vv = b_v, c_v
    ax.fill_betweenx([0, vv], h[1][li-vallyspanl], h[1][li+vallyspanr], alpha=0.5, color='lightgreen', label='vally fit interval')
    ax.scatter([pi,vi], [pv,vv], color='r')
    ## 将参数放入legend里
    selectinfo = info[j]['minPeakCharge'][(info[j]['nearPosMax']<=nearMax)&(info[j]['minPeak']>3)&(info[j]['minPeakCharge']<800)&(info[j]['minPeakCharge']>0.25*mu)]
    results[j] = (mu, vi, pv / vv, np.mean(selectinfo), np.std(selectinfo), mu/50/1.6, sigma/50/1.6, 0)# DCR 一项占位0
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label='G/1E7:{:.2f}'.format(mu/50/1.6*ADC2mV)))
    handles.append(mpatches.Patch(color='none', label='$\sigma_G$/1E7:{:.2f}'.format(sigma/50/1.6*ADC2mV)))
    handles.append(mpatches.Patch(color='none', label='P/V:{:.2f}'.format(pv/vv)))
    handles.append(mpatches.Patch(color='none', label='$\mu_{select}$:'+'{:.2f}'.format(results[j]['chargeMu'])))
    handles.append(mpatches.Patch(color='none', label='$\sigma_{select}$'+':{:.2f}'.format(results[j]['chargeSigma'])))
    ax.legend(handles=handles)
    pdf.savefig(fig)
    plt.close()

    # peak分布## 绘制筛选后的结果
    fig, ax = plt.subplots()
    h = ax.hist(info[j]['minPeak'][selectNearMax],histtype='step', bins=1000, range=[0, 1000], label='peak')
    ax.hist(info[j]['minPeak'][selectNearMax&(info[j]['minPeakCharge']>0.25*mu)], histtype='step', bins=1000, range=[0, 1000], alpha=0.8, label='peak($C>0.25 p.e.$)')
    print('peak height max:{};max index {}; part of peak {}'.format(np.max(h[0]), np.argmax(h[0]), h[0][:(np.argmax(h[0])+5)]))
    ax.set_xlabel('Peak/ADC')
    ax.set_ylabel('Entries')
    ax.legend()
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.set_yscale('log')
    pdf.savefig(fig)
    ## 计算DCR/kHz
    totalselect = (info[j]['minPeak']>3)&(info[j]['minPeakCharge']>0.25*mu)&selectNearMax
    DCR = np.sum(totalselect)/np.sum(selectNearMax)/waveformLength*1e6
    results[j]['DCR'] = DCR
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
    
    # min peak position and peak height distribution
    fig, ax = plt.subplots()
    ax.set_title('peakPos-peakHeight')
    h = ax.hist2d(info[j]['minPeakPos'][selectNearMax],info[j]['minPeak'][selectNearMax],range=[[0,waveformLength],[0,50]], bins=[int(waveformLength), 100], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('peakPos/ns')
    ax.set_ylabel('peakHeight/mV')
    pdf.savefig(fig)

    # min peak charge and peak height distribution
    fig, ax = plt.subplots()
    ax.set_title('peakCharge-peakHeight')
    h = ax.hist2d(info[j]['minPeakCharge'][selectNearMax],info[j]['minPeak'][selectNearMax],range=[[0,400],[0,50]], bins=[400, 50], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('peakCharge/mVns')
    ax.set_ylabel('peakHeight/mV')
    pdf.savefig(fig)
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

    # baseline
    fig, ax = plt.subplots()
    ax.set_title('baseline')
    ax.hist(info[j]['baseline'][selectNearMax],histtype='step',bins=100, label='baseline')
    ax.set_xlabel('baseline/mV')
    ax.set_ylabel('entries')
    ax.set_yscale('log')
    ax.legend()
    pdf.savefig(fig)

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
    fig, ax = plt.subplots()
    ## ax.set_title('$T_R$,$T_d$,FWHM ($V_p>5$mV,$10<p<${}ns) distribution'.format(waveformLength-30))
    ax.hist(info[j]['riseTime'][totalselect&position], histtype='step', bins=300, range=[0,30], label=r'risingtime:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.std(info[j]['riseTime'][totalselect&position]), np.mean(info[j]['riseTime'][totalselect&position])))
    ax.hist(info[j]['downTime'][totalselect&position], histtype='step', bins=300, range=[0,30], label=r'falltime:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.std(info[j]['downTime'][totalselect&position]), np.mean(info[j]['downTime'][totalselect&position])))
    ax.hist(info[j]['FWHM'][totalselect&position], histtype='step', bins=300, range=[0,30], label=r'FWHM:$\frac{\sigma}{\mu}$'+'={:.2f}/{:.2f}ns'.format(np.std(info[j]['FWHM'][totalselect&position]), np.mean(info[j]['FWHM'][totalselect&position])))
    ax.set_xlabel('Time/ns')
    ax.set_ylabel('Entries')
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
    #ax.set_xlim([1,40])
    # plt.savefig('{}/{}risetimeUlimit.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    ax.set_yscale('log')
    # plt.savefig('{}/{}risetimeUlimitLog.png'.format(args.opt,args.channel[j]))
    # pdf.savefig(fig)
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
    opt.create_dataset('res',data=results, compression='gzip')
