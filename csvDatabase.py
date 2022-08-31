'''
处理csv格式的数据
+ 获取run信息
+ 获取pmt对应通道信息
python3 CSVDatabase.py --origincsv runinfo/697.csv --runcsv runinfo/RUNINFO.csv --testcsv runinfo/TESTINFO.csv
'''
from multidict import istr
import pandas as pd, numpy as np
import argparse
import datetime
class CSVReader(object):
    def __init__(self, filename):
        self.csv = pd.read_csv(filename)
        self.filename = filename

class PMTINFO(CSVReader):
    # pmt信息获取: PMT,HV_r
    def __init__(self, filename):
        super(PMTINFO, self).__init__(filename)
        self.pmtinfo = self.csv.set_index('PMT')
    def getPMTInfo(self, pmt):
        return self.pmtinfo.loc[pmt]
class OriginINFO(CSVReader):
    # 输入的run信息获取:CHANNEL,BOXID,PMT,TRIGGER,MODE
    def __init__(self, filename):
        super(OriginINFO, self).__init__(filename)
    def getPMT(self):
        return self.csv['PMT']
    def getMode(self):
        return self.csv.iloc[0]['MODE']
class RUNINFO(CSVReader):
    # RUN信息设置: RUNNO,DATE,ISTRIGGER
    def __init__(self, filename):
        super(RUNINFO, self).__init__(filename)
        self.runinfo = self.csv.set_index('RUNNO')
    def updateAppend(self, runno, date, mode):
        # 更新或新增某一个run
        self.runinfo.loc[runno] = (date, mode)
    def getMode(self, runno):
        return self.runinfo.loc[runno]['MODE']
    def save(self):
        self.runinfo.reset_index().to_csv(self.filename, index=False)
class TESTINFO(CSVReader):
    # test信息设置: RUNNO,CHANNEL,BOXID,HV,PMT
    def __init__(self, filename):
        super(TESTINFO, self).__init__(filename)
    def appendRun(self, runno, origininfo, HV):
        origininfo['RUNNO'] = runno
        origininfo['HV'] = HV.astype('float64').values
        self.csv = pd.concat([self.csv, origininfo], join="inner")
    def getChannel(self, runno, istrigger=True):
        testcsv = self.csv.groupby('RUNNO').get_group(runno)
        if istrigger:
            return testcsv.iloc[0]['TRIGGER']
        else:
            return testcsv['CHANNEL'].values
    def save(self):
        self.csv.to_csv(self.filename, index=False)
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('--origincsv', help='origin csv file')
    psr.add_argument('--runcsv', help='run csv file')
    psr.add_argument('--testcsv', help='test csv file')
    psr.add_argument('--run', type=int, default=-1, help='run no')
    psr.add_argument('--para', default='istrigger')
    psr.add_argument('-i', dest='ipt', help='ID of PMT')
    psr.add_argument('-o', dest='opt', help='name of output csv')
    args = psr.parse_args()
    if (not (args.para=='pmts' or args.para=='pmtruns')) and args.run==-1:
        print('run parameter is not set and para is {}'.format(args.para))
        exit(0)
    # origininfo = OriginINFO(args.origincsv)
    runinfo = RUNINFO(args.runcsv)
    testinfo = TESTINFO(args.testcsv)
    if args.para=='istrigger':
        print(runinfo.getMode(args.run))
    elif args.para=='triggerch':
        print(testinfo.getChannel(args.run))
    elif args.para=='ch':
        print(' '.join(map(str,testinfo.getChannel(args.run, istrigger=False))))
    elif args.para=='pmts':
        print(' '.join(np.unique(testinfo.csv['PMT'].values)))
    elif args.para=='pmtruns':
        testinfo.csv[testinfo.csv['PMT']==args.ipt].to_csv(args.opt, index=False)
    else:
        print('error')