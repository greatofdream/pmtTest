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
excludedPMT = np.loadtxt('ExPMT/ExcludePMT.csv', dtype=str)
if args.run == -1:
    # all pmt
    pmtcsv = pd.read_csv(args.ipt)
    isexclude = np.array([(pmt not in excludedPMT) for pmt in pmtcsv['PMT'].values])
    ismcp = np.array([pmt.startswith('PM') for pmt in pmtcsv['PMT'].values])
    selectHama = (~ismcp) & isexclude
    selectMCP = ismcp & isexclude
    with PdfPages(args.opt) as pdf:
        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[selectHama]['Gain_DR'], pmtcsv[selectHama]['Gain'], xerr=np.sqrt(pmtcsv[selectHama]['Gain_DRVar']), yerr=np.sqrt(pmtcsv[selectHama]['GainVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[selectMCP]['Gain_DR'], pmtcsv[selectMCP]['Gain'], xerr=np.sqrt(pmtcsv[selectMCP]['Gain_DRVar']), yerr=np.sqrt(pmtcsv[selectMCP]['GainVar']), fmt='o', color='r', label='MCP-PMT')
        ax.axline((np.mean(pmtcsv[selectMCP]['Gain']), np.mean(pmtcsv[selectMCP]['Gain'])), linestyle='--', slope=1)
        ax.set_xlabel('$G_1$ in dark noise stage')
        ax.set_ylabel('$G_1$ in laser stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        # fig, ax = plt.subplots()
        # ax.scatter(pmtcsv[selectHama]['GainSigma_DR'], pmtcsv[selectHama]['GainSigma'], color='g', label='Reference PMT')
        # ax.scatter(pmtcsv[selectMCP]['GainSigma_DR'], pmtcsv[selectMCP]['GainSigma'], color='r', label='MCP-PMT')
        # ax.axline((0.3, 0.3), slope=1)
        # ax.set_xlabel('Gain Sigma in dark noise stage')
        # ax.set_ylabel('Gain Sigma in laser stage')
        # ax.legend()
        # pdf.savefig(fig)
        # plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[selectHama]['Res_DR'], pmtcsv[selectHama]['Res'], xerr=np.sqrt(pmtcsv[selectHama]['Res_DRVar']), yerr=np.sqrt(pmtcsv[selectHama]['ResVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[selectMCP]['Res_DR'], pmtcsv[selectMCP]['Res'], xerr=np.sqrt(pmtcsv[selectMCP]['Res_DRVar']), yerr=np.sqrt(pmtcsv[selectMCP]['ResVar']), fmt='o', color='r', label='MCP-PMT')
        ax.axline((0.27, 0.27), linestyle='--', slope=1)
        ax.set_xlabel('$\mathrm{Res}_1$ in dark noise stage')
        ax.set_ylabel('$\mathrm{Res}_1$ in laser stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[selectHama]['chargeMu_DR']/50/1.6*ADC2mV, pmtcsv[selectHama]['chargeMu']/50/1.6*ADC2mV, xerr=np.sqrt(pmtcsv[selectHama]['chargeMu_DRVar'])/50/1.6*ADC2mV, yerr=np.sqrt(pmtcsv[selectHama]['chargeMuVar'])/50/1.6*ADC2mV, fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[selectMCP]['chargeMu_DR']/50/1.6*ADC2mV, pmtcsv[selectMCP]['chargeMu']/50/1.6*ADC2mV, xerr=np.sqrt(pmtcsv[selectMCP]['chargeMu_DRVar'])/50/1.6*ADC2mV, yerr=np.sqrt(pmtcsv[selectMCP]['chargeMuVar'])/50/1.6*ADC2mV, fmt='o', color='r', label='MCP-PMT')
        ax.axline((np.mean(pmtcsv[selectMCP]['chargeMu']/50/1.6*ADC2mV), np.mean(pmtcsv[selectMCP]['chargeMu']/50/1.6*ADC2mV)), linestyle='--', slope=1)
        ax.set_xlabel('$G$/1E7 in dark noise stage')
        ax.set_ylabel('$G$/1E7 in laser stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        # ax.errorbar(pmtcsv[selectHama]['Gain'], pmtcsv[selectHama]['Res'], xerr=np.sqrt(pmtcsv[selectHama]['GainVar']), yerr=np.sqrt(pmtcsv[selectHama]['ResVar']), fmt='x', color='g', label='$G_1$,$Res_1$ of Reference PMT')
        # ax.errorbar(pmtcsv[selectHama]['chargeMu']/50/1.6*ADC2mV, pmtcsv[selectHama]['chargeRes'], xerr=np.sqrt(pmtcsv[selectHama]['chargeMuVar'])/50/1.6*ADC2mV, yerr=np.sqrt(pmtcsv[selectHama]['chargeResVar']), fmt='.', color='g', label='$G$,$Res$ of Reference PMT')
        # ax.errorbar(pmtcsv[selectMCP]['Gain'], pmtcsv[selectMCP]['Res'], xerr=np.sqrt(pmtcsv[selectMCP]['GainVar']), yerr=np.sqrt(pmtcsv[selectMCP]['ResVar']), fmt='x', color='r', label='$G_1$,$Res_1$ of MCP PMT')
        # ax.errorbar(pmtcsv[selectMCP]['chargeMu']/50/1.6*ADC2mV, pmtcsv[selectMCP]['chargeRes'], xerr=np.sqrt(pmtcsv[selectMCP]['chargeMuVar'])/50/1.6*ADC2mV, yerr=np.sqrt(pmtcsv[selectMCP]['chargeResVar']), fmt='.', color='r', label='$G$,$Res$ of MCP PMT')
        ax.scatter(pmtcsv[selectHama]['Gain'], pmtcsv[selectHama]['Res'], marker='x', color='g', label='$G_1$,$Res_1$ of Reference PMT')
        ax.scatter(pmtcsv[selectHama]['chargeMu']/50/1.6*ADC2mV, pmtcsv[selectHama]['chargeRes'], marker='.', color='g', label='$G$,$Res$ of Reference PMT')
        ax.scatter(pmtcsv[selectMCP]['Gain'], pmtcsv[selectMCP]['Res'], marker='x', color='r', label='$G_1$,$Res_1$ of MCP PMT')
        ax.scatter(pmtcsv[selectMCP]['chargeMu']/50/1.6*ADC2mV, pmtcsv[selectMCP]['chargeRes'], marker='.', color='r', label='$G$,$Res$ of MCP PMT')
        ax.set_xlabel('Gain/1E7')
        ax.set_ylabel('Res')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[selectHama]['chargeRes_DR'], pmtcsv[selectHama]['chargeRes'], xerr=np.sqrt(pmtcsv[selectHama]['chargeRes_DRVar']), yerr=np.sqrt(pmtcsv[selectHama]['chargeResVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[selectMCP]['chargeRes_DR'], pmtcsv[selectMCP]['chargeRes'], xerr=np.sqrt(pmtcsv[selectMCP]['chargeRes_DRVar']), yerr=np.sqrt(pmtcsv[selectMCP]['chargeResVar']), fmt='o', color='r', label='MCP-PMT')
        ax.axline((0.7, 0.7), linestyle='--', slope=1)
        ax.set_xlabel('$\mathrm{Res}$ in dark noise stage')
        ax.set_ylabel('$\mathrm{Res}$ in laser stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[selectHama]['PV_DR'], pmtcsv[selectHama]['PV'], xerr=np.sqrt(pmtcsv[selectHama]['PV_DRVar']), yerr=np.sqrt(pmtcsv[selectHama]['PVVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[selectMCP]['PV_DR'], pmtcsv[selectMCP]['PV'], xerr=np.sqrt(pmtcsv[selectMCP]['PV_DRVar']), yerr=np.sqrt(pmtcsv[selectMCP]['PVVar']), fmt='o', color='r', label='MCP-PMT')
        ax.axline((np.mean(pmtcsv[selectMCP]['PV']), np.mean(pmtcsv[selectMCP]['PV'])), linestyle='--', slope=1)
        ax.set_xlabel('P/V in dark noise stage')
        ax.set_ylabel('P/V in laser stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        # ax.errorbar(pmtcsv[selectHama]['Rise_DR'], pmtcsv[selectHama]['Rise'], xerr=np.sqrt(pmtcsv[selectHama]['Rise_DRVar']), yerr=np.sqrt(pmtcsv[selectHama]['RiseVar']), fmt='x', color='g', label='$t_r$ Reference PMT')
        # ax.errorbar(pmtcsv[selectHama]['Fall_DR'], pmtcsv[selectHama]['Fall'], xerr=np.sqrt(pmtcsv[selectHama]['Fall_DRVar']), yerr=np.sqrt(pmtcsv[selectHama]['FallVar']), fmt='x', color='r', label='$t_f$ Reference PMT')
        # ax.errorbar(pmtcsv[selectHama]['FWHM_DR'], pmtcsv[selectHama]['FWHM'], xerr=np.sqrt(pmtcsv[selectHama]['FWHM_DRVar']), yerr=np.sqrt(pmtcsv[selectHama]['FWHMVar']), fmt='x', color='k', label='FWHM Reference PMT')
        ax.scatter(pmtcsv[selectHama]['Rise_DR'], pmtcsv[selectHama]['Rise'], marker='x', color='g', label='$t_r$ Reference PMT')
        ax.scatter(pmtcsv[selectHama]['Fall_DR'], pmtcsv[selectHama]['Fall'], marker='x', color='r', label='$t_f$ Reference PMT')
        ax.scatter(pmtcsv[selectHama]['FWHM_DR'], pmtcsv[selectHama]['FWHM'], marker='x', color='k', label='FWHM Reference PMT')
        ax.axline((np.mean(pmtcsv[selectMCP]['Rise']), np.mean(pmtcsv[selectMCP]['Rise'])), linestyle='--', slope=1)
        # ax.errorbar(pmtcsv[selectMCP]['Rise_DR'], pmtcsv[selectMCP]['Rise'], xerr=np.sqrt(pmtcsv[selectMCP]['Rise_DRVar']), yerr=np.sqrt(pmtcsv[selectMCP]['RiseVar']), fmt='.', color='g', label='$t_r$ MCP PMT')
        # ax.errorbar(pmtcsv[selectMCP]['Fall_DR'], pmtcsv[selectMCP]['Fall'], xerr=np.sqrt(pmtcsv[selectMCP]['Fall_DRVar']), yerr=np.sqrt(pmtcsv[selectMCP]['FallVar']), fmt='.', color='r', label='$t_f$ MCP PMT')
        # ax.errorbar(pmtcsv[selectMCP]['FWHM_DR'], pmtcsv[selectMCP]['FWHM'], xerr=np.sqrt(pmtcsv[selectMCP]['FWHM_DRVar']), yerr=np.sqrt(pmtcsv[selectMCP]['FWHMVar']), fmt='.', color='k', label='FWHM MCP PMT')
        ax.scatter(pmtcsv[selectMCP]['Rise_DR'], pmtcsv[selectMCP]['Rise'], marker='.', color='g', label='$t_r$ MCP PMT')
        ax.scatter(pmtcsv[selectMCP]['Fall_DR'], pmtcsv[selectMCP]['Fall'], marker='.', color='r', label='$t_f$ MCP PMT')
        ax.scatter(pmtcsv[selectMCP]['FWHM_DR'], pmtcsv[selectMCP]['FWHM'], marker='.', color='k', label='FWHM MCP PMT')
        ax.set_xlabel('$t$/ns in dark noise stage')
        ax.set_ylabel('$t$/ns in laser stage')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[selectMCP]['PDE'], pmtcsv[selectMCP]['DCR'], xerr=np.sqrt(pmtcsv[selectMCP]['PDEVar']), yerr=np.sqrt(pmtcsv[selectMCP]['DCRVar']), fmt='o', color='r')
        ax.errorbar(pmtcsv[selectMCP]['PDE'], pmtcsv[selectMCP]['DCR_laser'], xerr=np.sqrt(pmtcsv[selectMCP]['PDEVar']), yerr=np.sqrt(pmtcsv[selectMCP]['DCR_laserVar']), fmt='o', color='g')
        # ax.errorbar(
        #     [np.sum(pmtcsv[selectMCP]['PDE']/pmtcsv[selectMCP]['PDEVar'])/np.sum(1/pmtcsv[selectMCP]['PDEVar'])],
        #     [np.sum(pmtcsv[selectMCP]['DCR']/pmtcsv[selectMCP]['DCRVar'])/np.sum(1/pmtcsv[selectMCP]['DCRVar'])],
        #     xerr = [np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['PDEVar']))],
        #     yerr = [np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['DCRVar']))],
        #     marker='*')
        ax.set_xlabel('relative PDE')
        ax.set_ylabel('DCR/kHz')
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[selectHama]['chargeRes'], pmtcsv[selectHama]['TTS_bin'], xerr=np.sqrt(pmtcsv[selectHama]['Res_DRVar']), yerr=np.sqrt(pmtcsv[selectHama]['TTS_binVar']), fmt='o', color='g', label='Reference PMT')
        ax.errorbar(pmtcsv[selectMCP]['chargeRes'], pmtcsv[selectMCP]['TTS_bin'], xerr=np.sqrt(pmtcsv[selectMCP]['Res_DRVar']), yerr=np.sqrt(pmtcsv[selectMCP]['TTS_binVar']), fmt='o', color='r', label='MCP-PMT')
        ax.axhline(np.average(pmtcsv[selectMCP]['TTS_bin']), linestyle='--', label='Average TTS of MCP PMT')
        ax.set_xlabel('Single PE resolution in laser stage')
        ax.set_ylabel('TTS/ns')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        # ax.errorbar(pmtcsv[selectHama]['DCR'], pmtcsv[selectHama]['Pre'], xerr=np.sqrt(pmtcsv[selectHama]['DCRVar']), yerr=np.sqrt(pmtcsv[selectHama]['PreVar']), fmt='o', color='g', label='After pulse 300-1000ns')
        ax.errorbar(pmtcsv[selectMCP]['DCR'], pmtcsv[selectMCP]['Pre'], xerr=np.sqrt(pmtcsv[selectMCP]['DCRVar']), yerr=np.sqrt(pmtcsv[selectMCP]['PreVar']), fmt='o', color='r')
        ax.axhline(np.average(pmtcsv[selectMCP]['Pre']), linestyle='--', label='Average pre-pulse ratio')
        ax.set_xlabel('DCR/kHz in dark noise stage')
        ax.set_ylabel('Pre pulse ratio in [-250,-50] ns')
        ax.legend()
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        # ax.errorbar(pmtcsv[selectHama]['DCR'], pmtcsv[selectHama]['After1'] + pmtcsv[selectHama]['After2'], xerr=np.sqrt(pmtcsv[selectHama]['DCRVar']), yerr=np.sqrt(pmtcsv[selectHama]['After1Var']+pmtcsv[selectHama]['After2Var']), fmt='o', color='g', label='After pulse 300-1000ns')
        ax.errorbar(pmtcsv[selectMCP]['Pre'], pmtcsv[selectMCP]['After1'] + pmtcsv[selectMCP]['After2'], xerr=np.sqrt(pmtcsv[selectMCP]['PreVar']), yerr=np.sqrt(pmtcsv[selectMCP]['After1Var']+pmtcsv[selectMCP]['After2Var']), fmt='o', color='r')
        # ax.axhline(np.average(pmtcsv[selectMCP]['After1'] + pmtcsv[selectMCP]['After2']), linestyle='--', label='Average After pulse ratio')
        ax.set_xlabel('$R_{\mathrm{pre}}$')
        ax.set_ylabel('$R_{\mathrm{after}}$')
        ax.ticklabel_format(style='sci',scilimits=(0,0))
        pdf.savefig(fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.errorbar(pmtcsv[selectMCP]['ser_tau'], pmtcsv[selectMCP]['ser_sigma'], xerr=np.sqrt(pmtcsv[selectMCP]['ser_tauVar']), yerr=np.sqrt(pmtcsv[selectMCP]['ser_sigmaVar']), fmt='o', color='r')
        ax.set_xlabel(r'$\tau_{\mathrm{ser}}$/ns')
        ax.set_ylabel(r'$\sigma_{\mathrm{ser}}$/ns')
        pdf.savefig(fig)
        plt.close()

        # 绘制后脉冲峰比例
        afterratios = []
        afterPeaks = []
        afterSigmas = []
        fig, ax = plt.subplots()
        for pmt in pmtcsv['PMT'].values:
            if pmt.startswith('PM'):
                with h5py.File(args.dir+'/'+pmt+'/laser.h5', 'r') as ipt:
                    afterpulse = ipt['AfterPulse'][:]
                ax.plot(afterpulse['t'], afterpulse['pv']/afterpulse['pv'][0], marker='o', label=pmt)
                afterratios.append(afterpulse['pv']/afterpulse['pv'][0])
                afterPeaks.append(afterpulse['t'])
                afterSigmas.append(afterpulse['sigma'])
                print(pmt, afterpulse['t'], afterpulse['pv']/afterpulse['pv'][0])
        ax.set_xlabel('Relative t/ns of peaks')
        ax.set_ylabel(r'$\frac{A_i}{A_1}$')
        ax.xaxis.set_minor_locator(MultipleLocator(100))
        ax.legend()
        pdf.savefig(fig)
        plt.close()
        afterratios = np.array(afterratios)
        afterPeaks = np.array(afterPeaks)
        afterSigmas = np.array(afterSigmas)

        fig, ax = plt.subplots()
        pdegrid = np.arange(0.9, 2, 0.01)
        resgrid = np.arange(0.2, 1, 0.01)
        X, Y = np.meshgrid(pdegrid, resgrid)
        resolutions = np.sqrt((1+Y**2)/X)
        # im = ax.pcolormesh(X, Y, resolutions, cmap=cm.jet)
        CS = ax.contour(X,Y, resolutions, origin='lower', cmap='flag')
        ax.scatter(pmtcsv[selectMCP]['PDE'], pmtcsv[selectMCP]['chargeRes'], color='g', marker='+', label='MCP-PMT')
        ax.scatter([1], [pmtcsv.set_index('PMT').loc['CR365']['chargeRes']], color='r', label='Reference PMT')
        print(pmtcsv[selectHama]['PDE'],  pmtcsv[selectHama]['chargeRes'])
        ax.clabel(CS, CS.levels,  # label every second level
              inline=True, fmt='%.1f', fontsize=14)
        ax.set_xlabel('relative PDE')
        ax.set_ylabel('Res')
        ax.legend()
        pdf.savefig(fig)
        plt.close()
        print('DCR {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['DCR']/pmtcsv[selectMCP]['DCRVar'])/np.sum(1/pmtcsv[selectMCP]['DCRVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['DCRVar'])),
            np.mean(pmtcsv[selectMCP]['DCR']), np.std(pmtcsv[selectMCP]['DCR'])
            ))
        print('DCR laser {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['DCR_laser']/pmtcsv[selectMCP]['DCR_laserVar'])/np.sum(1/pmtcsv[selectMCP]['DCR_laserVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['DCR_laserVar'])),
            np.mean(pmtcsv[selectMCP]['DCR_laser']), np.std(pmtcsv[selectMCP]['DCR_laser'])
            ))
        print('TTS {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['TTS_bin']/pmtcsv[selectMCP]['TTS_binVar'])/np.sum(1/pmtcsv[selectMCP]['TTS_binVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['TTS_binVar'])),
            np.mean(pmtcsv[selectMCP]['TTS_bin']), np.std(pmtcsv[selectMCP]['TTS_bin'])
            ))
        print('TTS fit {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['TTS']/pmtcsv[selectMCP]['TTSVar'])/np.sum(1/pmtcsv[selectMCP]['TTSVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['TTSVar'])),
            np.mean(pmtcsv[selectMCP]['TTS']), np.std(pmtcsv[selectMCP]['TTS'])
            ))
        print('PDE {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['PDE']/pmtcsv[selectMCP]['PDEVar'])/np.sum(1/pmtcsv[selectMCP]['PDEVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['PDEVar'])),
            np.mean(pmtcsv[selectMCP]['PDE']), np.std(pmtcsv[selectMCP]['PDE'])
            ))
        print('Res1 {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['Res']/pmtcsv[selectMCP]['ResVar'])/np.sum(1/pmtcsv[selectMCP]['ResVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['ResVar'])),
            np.mean(pmtcsv[selectMCP]['Res']), np.std(pmtcsv[selectMCP]['Res'])
            ))
        print('Res {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['chargeRes']/pmtcsv[selectMCP]['chargeResVar'])/np.sum(1/pmtcsv[selectMCP]['chargeResVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['chargeResVar'])),
            np.mean(pmtcsv[selectMCP]['chargeRes']), np.std(pmtcsv[selectMCP]['chargeRes'])
            ))
        print('Gain1 {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['Gain']/pmtcsv[selectMCP]['GainVar'])/np.sum(1//pmtcsv[selectMCP]['GainVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['GainVar'])),
            np.mean(pmtcsv[selectMCP]['Gain']), np.std(pmtcsv[selectMCP]['Gain'])
            ))
        print('Gain {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['chargeMu']/pmtcsv[selectMCP]['chargeMuVar'])/np.sum(1//pmtcsv[selectMCP]['chargeMuVar'])/50/1.6*ADC2mV,
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['chargeMuVar']))/50/1.6*ADC2mV,
            np.mean(pmtcsv[selectMCP]['chargeMu'])/50/1.6*ADC2mV, np.std(pmtcsv[selectMCP]['chargeMu'])/50/1.6*ADC2mV
            ))
        print('tau {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['ser_tau']/pmtcsv[selectMCP]['ser_tauVar'])/np.sum(1/pmtcsv[selectMCP]['ser_tauVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['chargeMuVar'])),
            np.mean(pmtcsv[selectMCP]['ser_tau']), np.std(pmtcsv[selectMCP]['ser_tau'])
            ))
        print('sigma {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['ser_sigma']/pmtcsv[selectMCP]['ser_sigmaVar'])/np.sum(1/pmtcsv[selectMCP]['ser_sigmaVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['ser_sigmaVar'])),
            np.mean(pmtcsv[selectMCP]['ser_sigma']), np.std(pmtcsv[selectMCP]['ser_sigma'])
            ))
        print('P/V {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['PV']/pmtcsv[selectMCP]['PVVar'])/np.sum(1/pmtcsv[selectMCP]['PVVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['PVVar'])),
            np.mean(pmtcsv[selectMCP]['PV']), np.std(pmtcsv[selectMCP]['PV'])
            ))
        print('Pre {:.5f} {:.5f} {:.5f} {:.5f}'.format(
            np.sum(pmtcsv[selectMCP]['Pre']/pmtcsv[selectMCP]['PreVar'])/np.sum(1/pmtcsv[selectMCP]['PreVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['PreVar'])),
            np.mean(pmtcsv[selectMCP]['Pre']), np.std(pmtcsv[selectMCP]['Pre'])
            ))
        print('After {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum((pmtcsv[selectMCP]['After1']+pmtcsv[selectMCP]['After2'])/(pmtcsv[selectMCP]['After1Var']+pmtcsv[selectMCP]['After2Var']))/np.sum(1/(pmtcsv[selectMCP]['After1Var']+pmtcsv[selectMCP]['After2Var'])),
            np.sqrt(1/np.sum(1/(pmtcsv[selectMCP]['After1Var']+pmtcsv[selectMCP]['After2Var']))),
            np.mean(pmtcsv[selectMCP]['After1']+pmtcsv[selectMCP]['After2']), np.std(pmtcsv[selectMCP]['After1']+pmtcsv[selectMCP]['After2'])
            ))
        print('After ratio {} {}'.format(
            np.mean(afterratios, axis=0),
            np.std(afterratios, axis=0)
            ))
        print('After peak time {} {}'.format(
            np.mean(afterPeaks, axis=0),
            np.std(afterPeaks, axis=0)
            ))
        print('After peak sigma {} {}'.format(
            np.mean(afterSigmas, axis=0),
            np.std(afterSigmas, axis=0)
            ))
        print('Rise {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['Rise']/pmtcsv[selectMCP]['RiseVar'])/np.sum(1/pmtcsv[selectMCP]['RiseVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['RiseVar'])),
            np.mean(pmtcsv[selectMCP]['Rise']), np.std(pmtcsv[selectMCP]['Rise'])
            ))
        print('Fall {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['Fall']/pmtcsv[selectMCP]['FallVar'])/np.sum(1/pmtcsv[selectMCP]['FallVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['FallVar'])),
            np.mean(pmtcsv[selectMCP]['Fall']), np.std(pmtcsv[selectMCP]['Fall'])
            ))
        print('FWHM {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            np.sum(pmtcsv[selectMCP]['FWHM']/pmtcsv[selectMCP]['FWHMVar'])/np.sum(1/pmtcsv[selectMCP]['FWHMVar']),
            np.sqrt(1/np.sum(1/pmtcsv[selectMCP]['FWHMVar'])),
            np.mean(pmtcsv[selectMCP]['FWHM']), np.std(pmtcsv[selectMCP]['FWHM'])
            ))
