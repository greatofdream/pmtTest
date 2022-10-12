'''
store the result of analysis into Database (Currently use csv).
'''
import argparse
import pandas as pd
import h5py
import config
import subprocess
from csvDatabase import OriginINFO
# RUNNO,BOXID,CHANNEL,PMT,HV,QE,Gain,PV,Res,PDE,DCR,TTS,Pre,After1,After2,Linear,Rise,Fall,TH,Overshoot,EventNum,TriggerNum,DCR_trigger
def Logger(process):
    while process.poll() is None:
        line = process.stdout.readline()
        line = line.strip()
        if line:
            print(line.decode("utf8", 'ignore'))
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output csv')
psr.add_argument('--run', type=int, help='run number')
psr.add_argument('--pulse', default=False, action='store_true', help='pulse result')
args = psr.parse_args()
metainfo = OriginINFO(config.databaseDir + '/{}.csv'.format(args.run))
# metainfo = OriginINFO('./tmp' + '/{}.csv'.format(args.run))

pmts, splitters, channels, triggerch = metainfo.csv['PMT'].values, metainfo.csv['BOXID'].values, metainfo.csv['CHANNEL'].values, metainfo.csv['TRIGGER'].values[0]
MODE = metainfo.csv['MODE'].values[0]
with h5py.File(args.ipt, 'r') as ipt:
    if not args.pulse:
        res = ipt['res'][:]
    else:
        res = ipt['ratio'][:]
storecsv = pd.read_csv(args.opt)
storecsv['RUNCH'] = storecsv['RUNNO'] * 10 + storecsv['CHANNEL']
tmpcsv = storecsv.set_index('RUNCH')
if MODE:
    # TRIGGER MODE
    if not args.pulse:
        for r, pmt, splitter in zip(res, pmts, splitters):
            tmpcsv.loc[args.run *10 + r['Channel'],['RUNNO', 'BOXID', 'CHANNEL', 'PMT', 'HV', 'QE', 'Gain', 'PV', 'Res',
        'DCR', 'TTS', 'Linear', 'Rise',
       'Fall', 'TH', 'Overshoot', 'EventNum', 'DCR_trigger']] = (
                args.run, splitter, r['Channel'], pmt, 0, 0, r['Gain'], r['PV'], r['GainSigma']/r['Gain'],
                0, r['TTS'],
                0,
                r['Rise'], r['Fall'], r['TH'],
                0, r['TotalNum'], r['DCR']
                )
    else:
        # The trigger store need before pulse store
        for r, pmt in zip(res, pmts):
            tmpcsv.loc[args.run *10 + r['Channel'], 'Pre'] = float(r['prompt'])
            tmpcsv.loc[args.run *10 + r['Channel'],'After1']= float(r['delay1'])
            tmpcsv.loc[args.run *10 + r['Channel'], 'After2'] = float(r['delay10'])
            tmpcsv.loc[args.run *10 + r['Channel'], 'TriggerNum'] = r['TriggerNum']
else:
    # NOISE MODE
    for r, pmt, splitter in zip(res, pmts, splitters):
        tmpcsv.loc[args.run *10 + r['Channel'], ['RUNNO', 'BOXID', 'CHANNEL', 'PMT', 'HV', 'QE', 'Gain', 'PV', 'Res',
        'DCR', 'TTS', 'Pre', 'After1', 'After2', 'Linear', 'Rise',
       'Fall', 'TH', 'Overshoot', 'EventNum', 'TriggerNum', 'DCR_trigger']] = (
            args.run, splitter, r['Channel'], pmt, 0, 0, r['Gain'], r['PV'], r['GainSigma']/r['Gain'],
            r['DCR'], 0,
            0, 0, 0,
            0,
            r['Rise'], r['Fall'], r['TH'],
            0, 0, 0, 0
            )
tmpcsv['RUNNO'] = tmpcsv['RUNNO'].astype('int64')
tmpcsv['CHANNEL'] = tmpcsv['CHANNEL'].astype('int64')
tmpcsv['BOXID'] = tmpcsv['BOXID'].astype('int64')
# RUNNO,CHANNEL,PMT,QE,Gain,PV,Res,PDE,DCR,TTS,Pre,After1,After2,Linear,Rise,Fall,TH,Overshoot
tmpcsv.sort_index().to_csv(args.opt, index=False)
# update config for each PMT
if not args.pulse:
    for pmt in pmts:
        process = subprocess.Popen('make ExPMT/{}/config.csv -f PMT.mk -B'.format(pmt), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        Logger(process)