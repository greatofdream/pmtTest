'''
Compare the result of dark noise and laser
'''
import pandas as pd, numpy as np
import h5py
import matplotlib.pyplot as plt
plt.style.use('./journal.mplstyle')
from matplotlib.ticker import MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import cm
import argparse
import config
import os
ADC2mV = config.ADC2mV
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', default='ExPMT/PMTSummary.csv', help='input csv')
psr.add_argument('-o', dest='opt', default='ExPMT/PMTSummary.pdf', help='output csv')
psr.add_argument('--run', type=int, default=-1, help='run number; default using merge result')
psr.add_argument('--dir', default='ExPMT', help='directory of PMT results')
args = psr.parse_args()
if args.run == -1:
    # all pmt
    pmtcsv = pd.read_csv(args.ipt)
    ismcp = np.array([pmt.startswith('PM') for pmt in pmtcsv['PMT'].values])
    with PdfPages(args.opt) as pdf:
        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['Gain_DR'], pmtcsv[~ismcp]['Gain'], xerr=np.sqrt(pmtcsv[~ismcp]['Gain_DRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['GainVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[ismcp]['Gain_DR'], pmtcsv[ismcp]['Gain'], xerr=np.sqrt(pmtcsv[ismcp]['Gain_DRVar']), yerr=np.sqrt(pmtcsv[ismcp]['GainVar']), fmt='o', color='r', label='MCP PMT')
        ax.axline((np.mean(pmtcsv[ismcp]['Gain']), np.mean(pmtcsv[ismcp]['Gain'])), linestyle='--', slope=1)
        ax.set_xlabel('$G_1$ of dark noise stage')
        ax.set_ylabel('$G_1$ of trigger stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        # fig, ax = plt.subplots()
        # ax.scatter(pmtcsv[~ismcp]['GainSigma_DR'], pmtcsv[~ismcp]['GainSigma'], color='g', label='Reference PMT')
        # ax.scatter(pmtcsv[ismcp]['GainSigma_DR'], pmtcsv[ismcp]['GainSigma'], color='r', label='MCP PMT')
        # ax.axline((0.3, 0.3), slope=1)
        # ax.set_xlabel('Gain Sigma of dark noise stage')
        # ax.set_ylabel('Gain Sigma of trigger stage')
        # ax.legend()
        # pdf.savefig(fig)
        # plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['Res_DR'], pmtcsv[~ismcp]['Res'], xerr=np.sqrt(pmtcsv[~ismcp]['Res_DRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['ResVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[ismcp]['Res_DR'], pmtcsv[ismcp]['Res'], xerr=np.sqrt(pmtcsv[ismcp]['Res_DRVar']), yerr=np.sqrt(pmtcsv[ismcp]['ResVar']), fmt='o', color='r', label='MCP PMT')
        ax.axline((0.27, 0.27), linestyle='--', slope=1)
        ax.set_xlabel('$\mathrm{Res}_1$ of dark noise stage')
        ax.set_ylabel('$\mathrm{Res}_1$ of trigger stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['chargeMu_DR']/50/1.6*ADC2mV, pmtcsv[~ismcp]['chargeMu']/50/1.6*ADC2mV, xerr=np.sqrt(pmtcsv[~ismcp]['chargeMu_DRVar'])/50/1.6*ADC2mV, yerr=np.sqrt(pmtcsv[~ismcp]['chargeMuVar'])/50/1.6*ADC2mV, fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[ismcp]['chargeMu_DR']/50/1.6*ADC2mV, pmtcsv[ismcp]['chargeMu']/50/1.6*ADC2mV, xerr=np.sqrt(pmtcsv[ismcp]['chargeMu_DRVar'])/50/1.6*ADC2mV, yerr=np.sqrt(pmtcsv[ismcp]['chargeMuVar'])/50/1.6*ADC2mV, fmt='o', color='r', label='MCP PMT')
        ax.axline((np.mean(pmtcsv[ismcp]['chargeMu']/50/1.6*ADC2mV), np.mean(pmtcsv[ismcp]['chargeMu']/50/1.6*ADC2mV)), linestyle='--', slope=1)
        ax.set_xlabel('$G$ of dark noise stage')
        ax.set_ylabel('$G$ of trigger stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['chargeRes_DR'], pmtcsv[~ismcp]['chargeRes'], xerr=np.sqrt(pmtcsv[~ismcp]['chargeRes_DRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['chargeResVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[ismcp]['chargeRes_DR'], pmtcsv[ismcp]['chargeRes'], xerr=np.sqrt(pmtcsv[ismcp]['chargeRes_DRVar']), yerr=np.sqrt(pmtcsv[ismcp]['chargeResVar']), fmt='o', color='r', label='MCP PMT')
        ax.axline((0.27, 0.27), linestyle='--', slope=1)
        ax.set_xlabel('$\mathrm{Res}$ of dark noise stage')
        ax.set_ylabel('$\mathrm{Res}$ of trigger stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['PV_DR'], pmtcsv[~ismcp]['PV'], xerr=np.sqrt(pmtcsv[~ismcp]['PV_DRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['PVVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[ismcp]['PV_DR'], pmtcsv[ismcp]['PV'], xerr=np.sqrt(pmtcsv[ismcp]['PV_DRVar']), yerr=np.sqrt(pmtcsv[ismcp]['PVVar']), fmt='o', color='r', label='MCP PMT')
        ax.axline((np.mean(pmtcsv[ismcp]['PV']), np.mean(pmtcsv[ismcp]['PV'])), linestyle='--', slope=1)
        ax.set_xlabel('P/V of dark noise stage')
        ax.set_ylabel('P/V of trigger stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['Rise_DR'], pmtcsv[~ismcp]['Rise'], xerr=np.sqrt(pmtcsv[~ismcp]['Rise_DRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['RiseVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[ismcp]['Rise_DR'], pmtcsv[ismcp]['Rise'], xerr=np.sqrt(pmtcsv[ismcp]['Rise_DRVar']), yerr=np.sqrt(pmtcsv[ismcp]['RiseVar']), fmt='o', color='r', label='MCP PMT')
        ax.axline((np.mean(pmtcsv[ismcp]['Rise']), np.mean(pmtcsv[ismcp]['Rise'])), linestyle='--', slope=1)
        ax.set_xlabel('Rise time of dark noise stage')
        ax.set_ylabel('Rise time of trigger stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['Fall_DR'], pmtcsv[~ismcp]['Fall'], xerr=np.sqrt(pmtcsv[~ismcp]['Fall_DRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['FallVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[ismcp]['Fall_DR'], pmtcsv[ismcp]['Fall'], xerr=np.sqrt(pmtcsv[ismcp]['Fall_DRVar']), yerr=np.sqrt(pmtcsv[ismcp]['FallVar']), fmt='o', color='r', label='MCP PMT')
        ax.axline((np.mean(pmtcsv[ismcp]['Fall']), np.mean(pmtcsv[ismcp]['Fall'])), linestyle='--', slope=1)
        ax.set_xlabel('Fall of dark noise stage')
        ax.set_ylabel('Fall of trigger stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['FWHM_DR'], pmtcsv[~ismcp]['FWHM'], xerr=np.sqrt(pmtcsv[~ismcp]['FWHM_DRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['FWHMVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[ismcp]['FWHM_DR'], pmtcsv[ismcp]['FWHM'], xerr=np.sqrt(pmtcsv[ismcp]['FWHM_DRVar']), yerr=np.sqrt(pmtcsv[ismcp]['FWHMVar']), fmt='o', color='r', label='MCP PMT')
        ax.axline((np.mean(pmtcsv[ismcp]['FWHM']), np.mean(pmtcsv[ismcp]['FWHM'])), slope=1)
        ax.set_xlabel('FWHM of dark noise stage')
        ax.set_ylabel('FWHM of trigger stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[ismcp]['PDE'], pmtcsv[ismcp]['DCR'], xerr=np.sqrt(pmtcsv[ismcp]['PDEVar']), yerr=np.sqrt(pmtcsv[ismcp]['DCRVar']), fmt='o', color='r', label='MCP PMT')
        ax.axhline(np.average(pmtcsv[ismcp]['DCR']), linestyle='--', label='Average DCR of MCP PMT')
        ax.set_xlabel('relative PDE')
        ax.set_ylabel('DCR')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['chargeRes'], pmtcsv[~ismcp]['TTS_bin'], xerr=np.sqrt(pmtcsv[~ismcp]['Res_DRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['ResVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[ismcp]['chargeRes'], pmtcsv[ismcp]['TTS_bin'], xerr=np.sqrt(pmtcsv[ismcp]['Res_DRVar']), yerr=np.sqrt(pmtcsv[ismcp]['ResVar']), fmt='o', color='r', label='MCP PMT')
        ax.axhline(np.average(pmtcsv[ismcp]['TTS_bin']), linestyle='--', label='Average TTS of MCP PMT')
        ax.set_xlabel('Single PE resolution of trigger stage')
        ax.set_ylabel('TTS')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['DCR'], pmtcsv[~ismcp]['Pre'], xerr=np.sqrt(pmtcsv[~ismcp]['DCRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['PreVar']), fmt='o', color='g', label='After pulse 300-1000ns')
        ax.errorbar(pmtcsv[ismcp]['DCR'], pmtcsv[ismcp]['Pre'], xerr=np.sqrt(pmtcsv[ismcp]['DCRVar']), yerr=np.sqrt(pmtcsv[ismcp]['PreVar']), fmt='o', color='r', label='After pulse 1000-10000ns')
        ax.axhline(np.average(pmtcsv[ismcp]['Pre']), linestyle='--', label='Average pre-pulse ratio of MCP PMT')
        ax.set_xlabel('DCR in dark noise stage')
        ax.set_ylabel('Pre pulse ratio in [-250,-50] ns')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[~ismcp]['DCR'], pmtcsv[~ismcp]['After1'] + pmtcsv[~ismcp]['After2'], xerr=np.sqrt(pmtcsv[~ismcp]['DCRVar']), yerr=np.sqrt(pmtcsv[~ismcp]['After1Var']+pmtcsv[~ismcp]['After2Var']), fmt='o', color='g', label='After pulse 300-1000ns')
        ax.errorbar(pmtcsv[ismcp]['DCR'], pmtcsv[ismcp]['After1'] + pmtcsv[ismcp]['After2'], xerr=np.sqrt(pmtcsv[ismcp]['DCRVar']), yerr=np.sqrt(pmtcsv[ismcp]['After1Var']+pmtcsv[ismcp]['After2Var']), fmt='o', color='r', label='After pulse 1000-10000ns')
        ax.axhline(np.average(pmtcsv[ismcp]['After1'] + pmtcsv[ismcp]['After2']), label='Average After pulse ratio of MCP PMT')
        ax.set_xlabel('DCR in dark noise stage')
        ax.set_ylabel('After pulse ratio in [300,10000] ns')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[ismcp]['ser_tau'], pmtcsv[ismcp]['ser_sigma'], xerr=np.sqrt(pmtcsv[ismcp]['ser_tauVar']), yerr=np.sqrt(pmtcsv[ismcp]['ser_sigmaVar']), fmt='o', color='r', label='MCP PMT')
        ax.set_xlabel('$\sigma_{\mathrm{ser}}$ of trigger stage')
        ax.set_ylabel(r'$\tau_{\mathrm{ser}}$ of trigger stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        # 绘制后脉冲峰比例
        fig, ax = plt.subplots()
        for pmt in pmtcsv['PMT'].values:
            if pmt.startswith('PM'):
                with h5py.File(args.dir+'/'+pmt+'/laser.h5', 'r') as ipt:
                    afterpulse = ipt['AfterPulse'][:]
                ax.plot(afterpulse['t'], afterpulse['pv']/afterpulse['pv'][0], marker='o')
                print(pmt, afterpulse['t'], afterpulse['pv']/afterpulse['pv'][0])
        ax.set_xlabel('time of peaks of after pulse')
        ax.set_ylabel(r'$\frac{A_i}{A_1}$')
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        pdegrid = np.arange(0.9, 2, 0.01)
        resgrid = np.arange(0.2, 1, 0.01)
        X, Y = np.meshgrid(pdegrid, resgrid)
        resolutions = np.sqrt((1+Y**2)/X)
        im = ax.pcolormesh(X, Y, resolutions)
        CS = ax.contour(X,Y, resolutions, origin='lower', cmap='flag')
        ax.scatter(pmtcsv[ismcp]['PDE'], pmtcsv[ismcp]['chargeRes'], color='g', label='MCP')
        ax.scatter([1], [pmtcsv.set_index('PMT').loc['CR365']['chargeRes']], color='r', label='Reference PMT')
        print(pmtcsv[~ismcp]['PDE'],  pmtcsv[~ismcp]['chargeRes'])
        ax.clabel(CS, CS.levels,  # label every second level
              inline=True, fmt='%.1f', fontsize=14)
        ax.set_xlabel('relative PDE')
        ax.set_ylabel('single PE resolution')
        ax.legend()
        pdf.savefig(fig)
        plt.close()
        print('DCR {:.2f}'.format(np.mean(pmtcsv[ismcp]['DCR'])))
        print('TTS {:.2f}'.format(np.mean(pmtcsv[ismcp]['TTS'])))
        print('PDE {:.2f}'.format(np.mean(pmtcsv[ismcp]['PDE'])))
        print('Res1 {:.2f}'.format(np.mean(pmtcsv[ismcp]['Res'])))
        print('Res {:.2f}'.format(np.mean(pmtcsv[ismcp]['chargeRes'])))
        print('Gain {:.2f}'.format(np.mean(pmtcsv[ismcp]['chargeMu']/50/1.6*ADC2mV)))
        print('tau {:.2f}'.format(np.mean(pmtcsv[ismcp]['ser_tau'])))
        print('sigma {:.2f}'.format(np.mean(pmtcsv[ismcp]['ser_sigma'])))
