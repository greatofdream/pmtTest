import matplotlib.pyplot as plt
import h5py, argparse
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
from scipy.optimize import minimize
'''
该文件计算激光触发符合区域的事例对应的参量，以及TTS
'''
def fitGaus(tts,limits):
    tts_select = tts[(tts<limits[1])&(tts>limits[0])]
    result = minimize(likelihood,[1, np.mean(tts_select),np.std(tts_select)],args=(tts_select, tts_select.shape[0]))
    return result, tts_select.shape[0]
def likelihood(x,*args):
    A,mu,sigma = x
    tts,N = args
    return A*N-tts.shape[0]*np.log(A)+np.sum((tts-mu)**2)/2/sigma**2+tts.shape[0]*np.log(sigma)

psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
psr.add_argument('-t', dest='trigger', help='trigger h5 file')
args = psr.parse_args()
#plt.style.use('fivethirtyeight')
info = []
results = np.zeros(len(args.channel), dtype=[('peakC','<f4'), ('vallyC','<f4'),('PV','<f4'),('chargeMu','<f4'),('chargeSigma','<f4')])
with h5py.File(args.ipt, 'r') as ipt:
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
nearMax = 10
for j in range(len(args.channel)):
    # charge分布
    fig, ax = plt.subplots()
    ax.set_title('charge distribution')
    rangemin = int(np.min(info[j]['charge'])-1)
    rangemax = int(np.max(info[j]['charge'])+1)
    bins = rangemax-rangemin
    h = ax.hist(info[j]['charge'], histtype='step', bins=bins, range=[rangemin, rangemax], label='charge')
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
    if h[0].shape[0]>200:
        ax.set_ylim([0, 2*np.max(h[0][70:150])])
        pi = h[1][70:150][np.argmax(h[0][70:150])]
        vi = h[1][15:70][np.argmin(h[0][15:70])]
        pv = np.max(h[0][70:150])
        vv = np.min(h[0][10:80])
        plt.scatter([pi,vi],[pv,vv])
        selectinfo = info[j]['charge'][(info[j]['minPeak']>3)&(info[j]['charge']<800)]
        results[j] = (pi,vi, pv/vv,np.mean(selectinfo), np.std(selectinfo))
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
    h = ax.hist(info[j]['minPeak'],histtype='step', bins=1000, range=[0,1000], label='baseline - peak')
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
    ax.set_title('$T_R$,$T_d$,FWHM ($V_p>3$mV) distribution')
    ax.hist(info[j]['riseTime'][(info[j]['minPeak']>3)],histtype='step',bins=300, range=[0,30], label='risingtime:{:.2f}ns'.format(np.mean(info[j]['riseTime'][(info[j]['minPeak']>5)])))
    ax.hist(info[j]['downTime'][(info[j]['minPeak']>3)],histtype='step',bins=300, range=[0,30], label='downtime:{:.2f}ns'.format(np.mean(info[j]['downTime'][(info[j]['minPeak']>5)])))
    ax.hist(info[j]['FWHM'][(info[j]['minPeak']>3)],histtype='step',bins=300, range=[0,30], label='FWHM:{:.2f}ns'.format(np.mean(info[j]['FWHM'][(info[j]['minPeak']>5)])))
    ax.set_xlabel('Time/ns')
    ax.set_ylabel('entries')
    ax.legend()
    #ax.set_xlim([1,40])
    pdf.savefig(fig)
    ax.set_yscale('log')
    # pdf.savefig(fig)
    plt.close()

    fig,ax = plt.subplots()
    limits_mu, limits_sigma = np.mean(info[j]['up10'][(info[j]['minPeak']>3)]),np.std(info[j]['up10'][(info[j]['minPeak']>3)])
    limits = [limits_mu-limits_sigma, limits_mu+limits_sigma]
    result, N = fitGaus(info[j]['up10'][(info[j]['minPeak']>3)], limits)
    print(result)
    ax.hist(info[j]['up10'][(info[j]['minPeak']>3)],bins=int(100*limits_sigma),range=[limits_mu-5*limits_sigma, limits_mu+5*limits_sigma], histtype='step', label='$t_{0.1}-t_{trigger}$')
    ax.plot(np.arange(limits_mu-5*limits_sigma, limits_mu+5*limits_sigma, 0.1),result.x[0]*N*0.1*np.exp(-(np.arange(limits_mu-5*limits_sigma, limits_mu+5*limits_sigma,0.1)-result.x[1])**2/2/result.x[2]**2)/np.sqrt(2*np.pi)/result.x[2],'--')
    ax.plot(np.arange(limits[0],limits[1],0.1), result.x[0]*N*0.1*np.exp(-(np.arange(limits[0],limits[1],0.1)-result.x[1])**2/2/result.x[2]**2)/np.sqrt(2*np.pi)/result.x[2],label='fit')
    ax.set_xlabel('TT/ns')
    ax.set_ylabel('Entries')
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color='none', label='$\sigma$={:.3f}'.format(result.x[2])))
    ax.legend(handles=handles)
    print('tts:{:.3f}'.format(result.x[2]*2.355))
    pdf.savefig(fig)
    plt.close()
pdf.close()
with h5py.File(args.opt, 'w') as opt:
    opt.create_dataset('res',data=results, compression='gzip')
