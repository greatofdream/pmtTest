import numpy as np, h5py
import argparse
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
plt.style.use('presentation.mplstyle')

if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input h5 file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    args = psr.parse_args()

    with h5py.File(args.ipt, 'r') as ipt:
        pul_ch2 = ipt['ch2'][:]
        pul_ch3 = ipt['ch3'][:]
    prompt_ch2 = pul_ch2['promptPulse']
    delaypul1_ch2 = pul_ch2['DelayPulse1']
    delaypul10_ch2 = pul_ch2['DelayPulse10']
    prompt_ch3 = pul_ch3['promptPulse']
    delaypul1_ch3 = pul_ch3['DelayPulse1']
    delaypul10_ch3 = pul_ch3['DelayPulse10']
    trig_ch2 = pul_ch2['isTrigger']
    trig_ch3 = pul_ch3['isTrigger']

    prompt_ratio_ch2 = len(np.where((np.logical_and(prompt_ch2>-1,trig_ch2)))[0])/np.sum(trig_ch2) 
    delay1_ratio_ch2 = len(np.where((delaypul1_ch2>-1))[0])/np.sum(trig_ch2)
    delay10_ratio_ch2 = len(np.where((delaypul10_ch2>-1))[0])/np.sum(trig_ch2)

    prompt_ratio_ch3 = len(np.where((np.logical_and(prompt_ch3>-1,trig_ch3)))[0])/np.sum(trig_ch3) 
    delay1_ratio_ch3 = len(np.where((delaypul1_ch3>-1))[0])/np.sum(trig_ch3)
    delay10_ratio_ch3 = len(np.where((delaypul10_ch3>-1))[0])/np.sum(trig_ch3)
    with PdfPages(args.opt) as pdf:
        fig, ax = plt.subplots(dpi=150)
        ax.hist(prompt_ch2[prompt_ch2>-1], range=[100, 300], bins=30, histtype='step', label='ch2')
        ax.hist(prompt_ch3[prompt_ch3>-1], range=[100, 300], bins=30, histtype='step', label='ch3')
        # ax.set_yscale('log')
        ax.set_xlabel('pulsePosition/ns')
        ax.set_ylabel('entries')
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)

        fig, ax = plt.subplots(dpi=150)
        ax.hist(delaypul1_ch2[delaypul1_ch2>-1], range=[600, 1300], bins=70, histtype='step', label='ch2')
        ax.hist(delaypul1_ch3[delaypul1_ch3>-1], range=[600, 1300], bins=70, histtype='step', label='ch3')
        # ax.set_yscale('log')
        ax.set_xlabel('pulsePosition/ns')
        ax.set_ylabel('entries')
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)

        fig, ax = plt.subplots(dpi=150)
        ax.hist(delaypul10_ch2[delaypul10_ch2>-1], range=[1300, 10300], bins=500, histtype='step', label='ch2')
        ax.hist(delaypul10_ch3[delaypul10_ch3>-1], range=[1300, 10300], bins=500, histtype='step', label='ch3')
        # ax.set_yscale('log')
        ax.set_xlabel('pulsePosition/ns')
        ax.set_ylabel('entries')
        ax.legend()
        fig.tight_layout()
        pdf.savefig(fig)

    print(prompt_ratio_ch2, delay1_ratio_ch2, delay10_ratio_ch2, np.sum(trig_ch2))
    print(prompt_ratio_ch3, delay1_ratio_ch3, delay10_ratio_ch3, np.sum(trig_ch3))
