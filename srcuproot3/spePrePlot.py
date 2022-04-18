from unittest import result
import matplotlib.pyplot as plt
import h5py, argparse
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches

psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
args = psr.parse_args()
#plt.style.use('fivethirtyeight')
info = []
results = np.zeros(len(args.channel), dtype=[('peakC','<f4'), ('vallyC','<f4'),('PV','<f4'),('chargeMu','<f4'),('chargeSigma','<f4')])
with h5py.File(args.ipt, 'r') as ipt:
    waveformLength = ipt.attrs['waveformLength']
    #waveformLength = 1500
    for j in range(len(args.channel)):
        print(j)
        info.append(ipt['ch{}'.format(args.channel[j])][:])
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
nearMax = 10
for j in range(len(args.channel)):
    selectNearMax = info[j]['nearPosMax']<=nearMax
    # charge分布
    fig, ax = plt.subplots()
    ax.set_title('charge distribution')
    rangemin = int(np.min(info[j]['minPeakCharge'])-1)
    rangemax = int(np.max(info[j]['minPeakCharge'])+1)
    bins = rangemax-rangemin
    # ax.hist(info[j]['allCharge'],histtype='step', bins=bins, range=[rangemin, rangemax], label='all charge integrate')
    h = ax.hist(info[j]['minPeakCharge'][info[j]['nearPosMax']<=nearMax],histtype='step', bins=bins, range=[rangemin, rangemax], label='charge')
    ax.set_xlabel('charge/mVns')
    ax.set_ylabel('entries')
    ax.legend()
    ax.set_yscale('log')
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    # plt.savefig('{}/{}charge.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    ax.set_xlim([-5, 1000])
    pdf.savefig(fig)
    ax.set_yscale('linear')
    ax.set_ylim([0, 2*np.max(h[0][50:200])])
    pi = h[1][50:200][np.argmax(h[0][50:200])]
    vi = h[1][10:80][np.argmin(h[0][10:80])]
    pv = np.max(h[0][50:200])
    vv = np.min(h[0][10:80])
    plt.scatter([pi,vi],[pv,vv])
    selectinfo = info[j]['minPeakCharge'][(info[j]['nearPosMax']<=nearMax)&(info[j]['minPeak']>3)&(info[j]['minPeakCharge']<800)]
    results[j] = (pi,vi,pv/vv,np.mean(selectinfo), np.std(selectinfo))
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label='Gain:{:.2f}'.format(pi/50/1.6)))
    handles.append(mpatches.Patch(color='none', label='P/V:{:.2f}'.format(pv/vv)))
    handles.append(mpatches.Patch(color='none', label='$\mu_{p>3mV}$:'+'{:.2f}'.format(results[j]['chargeMu'])))
    handles.append(mpatches.Patch(color='none', label='$\sigma_{p>3mV}$'+':{:.2f}'.format(results[j]['chargeSigma'])))
    ax.legend(handles=handles)
    # plt.savefig('{}/{}chargeLinear.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    plt.close()

    # peak分布
    fig, ax = plt.subplots()
    ax.set_title('peak height distribution')
    h = ax.hist(info[j]['minPeak'][info[j]['nearPosMax']<=nearMax],histtype='step', bins=1000, range=[0,1000], label='baseline - peak')
    print('peak height max:{};max index {}; part of peak {}'.format(np.max(h[0]), np.argmax(h[0]), h[0][:(np.argmax(h[0])+5)]))
    ax.set_xlabel('peak height/mV')
    ax.set_ylabel('entries')
    ax.legend()
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    # plt.savefig('{}/{}minpeakLinear.png'.format(args.opt,args.channel[j]))
    # pdf.savefig(fig)
    ax.set_yscale('log')
    # plt.savefig('{}/{}minpeak.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    ax.set_yscale('linear')
    ax.set_xlim([0,100])
    ax.set_ylim([0,2*np.max(h[0][5:30])])
    pdf.savefig(fig)

    # min peak position分布
    fig, ax = plt.subplots()
    ax.set_title('peak position distribution')
    h = ax.hist(info[j]['minPeakPos'][selectNearMax],histtype='step', bins=waveformLength, range=[0,waveformLength], label='the largest peak pos')
    print('h shape:{};max index {}'.format(h[0].shape,np.argmax(h[0])))
    ax.set_xlabel('peak position/ns')
    ax.set_ylabel('entries')
    ax.legend()
    # plt.savefig('{}/{}minpeakposLinear.png'.format(args.opt,args.channel[j]))
    # pdf.savefig(fig)
    ax.set_yscale('log')
    # plt.savefig('{}/{}minpeakpos.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    
    fig, ax = plt.subplots()
    ax.set_title('peak($V_p>5$mV) position distribution')
    h = ax.hist(info[j]['minPeakPos'][(info[j]['minPeak']>5)&(selectNearMax)],histtype='step', bins=waveformLength, range=[0,waveformLength], label='the largest peak pos')
    print('h shape:{};max index {}'.format(h[0].shape,np.argmax(h[0])))
    ax.set_xlabel('peak position/ns')
    ax.set_ylabel('entries')
    ax.legend()
    # plt.savefig('{}/{}minpeakposCutLinear.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    ax.set_yscale('log')
    # plt.savefig('{}/{}minpeakposCut.png'.format(args.opt,args.channel[j]))
    # pdf.savefig(fig)
    
    # min peak position and peak height distribution
    fig, ax = plt.subplots()
    ax.set_title('peakPos-peakHeight')
    h = ax.hist2d(info[j]['minPeakPos'][selectNearMax],info[j]['minPeak'][selectNearMax],range=[[0,waveformLength],[0,50]], bins=[int(waveformLength), 100], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('peakPos/ns')
    ax.set_ylabel('peakHeight/mV')
    # plt.savefig('{}/{}peakPos-peakHeight.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)

    # min peak charge and peak height distribution
    fig, ax = plt.subplots()
    ax.set_title('peakCharge-peakHeight')
    h = ax.hist2d(info[j]['minPeakCharge'][selectNearMax],info[j]['minPeak'][selectNearMax],range=[[0,400],[0,50]], bins=[400, 50], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('peakCharge/mVns')
    ax.set_ylabel('peakHeight/mV')
    # plt.savefig('{}/{}peakCharge-peakHeight.png'.format(args.opt,args.channel[j]))
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
    h = ax.hist2d(info[j]['baseline'][selectNearMax],info[j]['std'], bins=[100,100], cmap=cmap)
    fig.colorbar(h[3], ax=ax)
    ax.set_xlabel('baseline/mV')
    ax.set_ylabel('std/mV')
    # plt.savefig('{}/{}base-std.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)

    # baseline
    fig, ax = plt.subplots()
    ax.set_title('baseline')
    ax.hist(info[j]['baseline'][selectNearMax],histtype='step',bins=100, label='baseline')
    ax.set_xlabel('baseline/mV')
    ax.set_ylabel('entries')
    ax.set_yscale('log')
    ax.legend()
    # plt.savefig('{}/{}baselineLinear.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)

    # std
    fig, ax = plt.subplots()
    ax.set_title('std distribution')
    ax.hist(info[j]['std'][selectNearMax],histtype='step', bins=100, label='std')
    ax.set_xlabel('std/mV')
    ax.set_ylabel('entries')
    ax.legend()
    ax.set_yscale('log')
    # plt.savefig('{}/{}stdLinear.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    plt.close()
    
    # risetime and downtime
    fig, ax = plt.subplots()
    ax.set_title('$T_R$,$T_d$,FWHM ($V_p>5$mV) distribution')
    ax.hist(info[j]['riseTime'][(info[j]['minPeak']>5)&(selectNearMax)],histtype='step',bins=300, range=[0,30], label='risingtime')
    ax.hist(info[j]['downTime'][(info[j]['minPeak']>5)&(selectNearMax)],histtype='step',bins=300, range=[0,30], label='downtime')
    ax.hist(info[j]['FWHM'][(info[j]['minPeak']>5)&(selectNearMax)],histtype='step',bins=300, range=[0,30], label='FWHM')
    ax.set_xlabel('riseTime/ns')
    ax.set_ylabel('entries')
    ax.legend()
    #ax.set_xlim([1,40])
    # plt.savefig('{}/{}risetime.png'.format(args.opt,args.channel[j]))
    pdf.savefig(fig)
    ax.set_yscale('log')
    # plt.savefig('{}/{}risetimeLog.png'.format(args.opt,args.channel[j]))
    # pdf.savefig(fig)
    plt.close()

    fig, ax = plt.subplots()
    ax.set_title('$T_R$,$T_d$,FWHM ($5<V_p<40$mV) distribution')
    ax.hist(info[j]['riseTime'][(info[j]['minPeak']>5)&(info[j]['minPeak']<40)&selectNearMax],histtype='step',bins=300, range=[0,30], label='risingtime')
    ax.hist(info[j]['downTime'][(info[j]['minPeak']>5)&(info[j]['minPeak']<40)&selectNearMax],histtype='step',bins=300, range=[0,30], label='downtime')
    ax.hist(info[j]['FWHM'][(info[j]['minPeak']>5)&(info[j]['minPeak']<40)&(selectNearMax)],histtype='step',bins=300, range=[0,30], label='FWHM')
    ax.set_xlabel('riseTime/ns')
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