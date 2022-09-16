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
try:
    with h5py.File(args.dark, 'r') as ipt:
        darkres = ipt['merge'][:]
    darkExpect = True
except:
    darkExpect = False
storecsv = pd.read_csv(args.opt)
tmpcsv = storecsv.set_index('PMT')
tmpcsv.loc[args.pmt, ['Gain', 'PV', 'Res', 'TTS', 'Rise', 'Fall', 'TH', 'FWHM']] = (
    laserres[0]['Gain'], laserres[0]['PV'], laserres[0]['Res'], laserres[0]['TTS'], laserres[0]['Rise'], laserres[0]['Fall'], laserres[0]['TH'], laserres[0]['FWHM']
)
tmpcsv.loc[args.pmt, ['Pre', 'After1', 'After2']] = (pulse[0]['promptWODCR'], pulse[0]['delay1WODCR'], pulse[0]['delay10WODCR'])
if darkExpect:
    tmpcsv.loc[args.pmt, ['Gain_DR', 'PV_DR', 'Res_DR', 'DCR', 'Rise_DR', 'Fall_DR', 'TH_DR', 'FWHM_DR']] = (
        darkres[0]['Gain'], darkres[0]['PV'], darkres[0]['Res'], darkres[0]['DCR'], darkres[0]['Rise'], darkres[0]['Fall'], darkres[0]['TH'], darkres[0]['FWHM']
    )
tmpcsv.sort_index().reset_index().to_csv(args.opt, index=False)