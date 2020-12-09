import matplotlib.pyplot as plt
import h5py, argparse
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output png file')
psr.add_argument('-c', dest='channel', nargs='+', default=[0,1],help='channel used in DAQ')
args = psr.parse_args()
info = []
with h5py.File(args.ipt, 'r') as ipt:
    for j in range(len(args.channel)):
        info.append(ipt['ch{}'.format(args.channel[j])][:])
rangemin =-100
rangemax = 500
bins = rangemax-rangemin
for j in range(len(args.channel)):
    fig, ax = plt.subplots()
    ax.set_title('charge distribution')
    #ax.hist(info[j]['allCharge'],histtype='step', bins=bins, range=[rangemin, rangemax], label='all charge integrate')
    ax.hist(info[j]['minPeakCharge'],histtype='step', bins=bins, range=[rangemin, rangemax], label='[peak-50, peak+50] charge integrate')
    ax.set_xlabel('charge/mVns')
    ax.set_ylabel('entries')
    ax.legend()
    #ax.set_yscale('log')
    plt.savefig('{}/{}chargeLinear.png'.format(args.opt,args.channel[j]))
    plt.close()
    fig, ax = plt.subplots()
    ax.set_title('peak height distribution')
    ax.hist(info[j]['minPeak'],histtype='step', bins=100, range=[0,50], label='baseline - peak')
    ax.set_xlabel('peak height/mV')
    ax.set_ylabel('entries')
    ax.legend()
    #ax.set_yscale('log')
    plt.savefig('{}/{}minpeakLinear.png'.format(args.opt,args.channel[j]))
    plt.close()
