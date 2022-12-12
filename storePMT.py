'''
store the result of analysis into Database (Currently use csv).
'''
import argparse
import pandas as pd
import h5py
import config
from csvDatabase import OriginINFO
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input PMT summary csv')
psr.add_argument('-o', dest='opt', help='output csv')
psr.add_argument('--pmt', help='run number')
psr.add_argument('--dark', help='dark result')
psr.add_argument('--laser', help='laser result')
args = psr.parse_args()
with h5py.File(args.laser, 'r') as ipt:
    pulse = ipt['mergepulse'][:]
    laserres = ipt['merge'][:]
    serres = ipt['mergeSER'][:]
try:
    with h5py.File(args.dark, 'r') as ipt:
        darkres = ipt['merge'][:]
    darkExpect = True
except:
    darkExpect = False
storecsv = pd.read_csv(args.opt)
tmpcsv = storecsv.set_index('PMT')
## mean
tmpcsv.loc[args.pmt, ['Gain', 'PV', 'Res', 'TTS', 'TTS2', 'TTA', 'TTA2','TTS_bin', 'Rise', 'Fall', 'TH', 'FWHM', 'chargeMu', 'chargeRes', 'PDE']] = (
    laserres[0]['Gain'], laserres[0]['PV'], laserres[0]['Res'], laserres[0]['TTS'], laserres[0]['TTS2'], laserres[0]['TTA'], laserres[0]['TTA2'], laserres[0]['TTS_bin'], laserres[0]['Rise'], laserres[0]['Fall'], laserres[0]['TH'], laserres[0]['FWHM'], laserres[0]['chargeMu'], laserres[0]['chargeRes'], laserres[0]['PDE']
)
tmpcsv.loc[args.pmt, ['Pre', 'After1', 'After2', 'DCR_laser']] = (pulse[0]['promptWODCR'], pulse[0]['delay1WODCR'], pulse[0]['delay10WODCR'], pulse[0]['DCR'])
tmpcsv.loc[args.pmt, ['ser_tau', 'ser_sigma']] = (serres[0]['tau'], serres[0]['sigma'])
## variance
tmpcsv.loc[args.pmt, ['GainVar', 'PVVar', 'ResVar', 'TTSVar', 'TTS_binVar', 'RiseVar', 'FallVar', 'THVar', 'FWHMVar', 'chargeMuVar', 'chargeResVar', 'PDEVar']] = (
    laserres[1]['Gain'], laserres[1]['PV'], laserres[1]['Res'], laserres[1]['TTS'], laserres[1]['TTS_bin'], laserres[1]['Rise'], laserres[1]['Fall'], laserres[1]['TH'], laserres[1]['FWHM'], laserres[1]['chargeMu'], laserres[1]['chargeRes'], laserres[1]['PDE']
)
tmpcsv.loc[args.pmt, ['PreVar', 'After1Var', 'After2Var', 'DCR_laserVar']] = (pulse[1]['promptWODCR'], pulse[1]['delay1WODCR'], pulse[1]['delay10WODCR'], pulse[1]['DCR'])
tmpcsv.loc[args.pmt, ['ser_tauVar', 'ser_sigmaVar']] = (serres[1]['tau'], serres[1]['sigma'])

if darkExpect:
    tmpcsv.loc[args.pmt, ['Gain_DR', 'PV_DR', 'Res_DR', 'DCR', 'Rise_DR', 'Fall_DR', 'TH_DR', 'FWHM_DR', 'chargeMu_DR', 'chargeRes_DR']] = (
        darkres[0]['Gain'], darkres[0]['PV'], darkres[0]['Res'], darkres[0]['DCR'], darkres[0]['Rise'], darkres[0]['Fall'], darkres[0]['TH'], darkres[0]['FWHM'], darkres[0]['chargeMu'], darkres[0]['chargeRes']
    )
    tmpcsv.loc[args.pmt, ['Gain_DRVar', 'PV_DRVar', 'Res_DRVar', 'DCRVar', 'Rise_DRVar', 'Fall_DRVar', 'TH_DRVar', 'FWHM_DRVar', 'chargeMu_DRVar', 'chargeRes_DRVar']] = (
        darkres[1]['Gain'], darkres[1]['PV'], darkres[1]['Res'], darkres[1]['DCR'], darkres[1]['Rise'], darkres[1]['Fall'], darkres[1]['TH'], darkres[1]['FWHM'], darkres[1]['chargeMu'], darkres[1]['chargeRes']
    )
tmpcsv.sort_index().reset_index().to_csv(args.opt, index=False)