import numpy as np
from .waveana import Waveana, interpolate
from .waveana import Qb, Qe
'''
use trigger channel to extract information from other numpy waveforms
'''
class Triggerana(Waveana):
    def __init__(self, wave=[], eid=0) -> None:
        super().__init__(wave, eid)
    def setTriggerWave(self, triggerWave, uprising=True):
        self.triggerWave = triggerWave
        self.triggerTime = self.getTriggerTime(50, 0.1, uprising)
        # print(self.triggerTime)
    def getTriggerTime(self, begin=50, threshold=0.1, uprising=True):
        # TODO: 初始的baseline长度不应该硬编码
        baseline = np.average(self.triggerWave[0:50])
        baseline2 = np.average(self.triggerWave[-150:-50])
        thresholdLevel = baseline2*threshold+baseline*(1-threshold)
        if uprising:
            for i in range(begin, self.triggerWave.shape[0]):
                #print(self.triggerWave[i])
                if self.triggerWave[i]>(thresholdLevel):
                    return i-1+interpolate(thresholdLevel,self.triggerWave[i-1],self.triggerWave[i])
            print('Warning:{} cannot find trigger'.format(self.eid))
            return 175
        else:
             for i in range(begin, self.triggerWave.shape[0]):
                if self.triggerWave[i]<(baseline-threshold):
                    return i
    def integrateWave(self):
        baseline = self.minPeakBaseline
        self.allCharge = np.sum(baseline-self.wave[int(self.triggerTime):])
        return self.allCharge
    def integrateMinPeakWave(self, minIndex, baselength=15, afterlength=40):
        self.minIndex = minIndex
        self.minPeak = self.minPeakBaseline-self.wave[self.minIndex]
        if self.minPeak<0:
            return 0, 0, 0
        self.begin10, self.begin50, self.begin90 = Qb(self.wave, self.minIndex, self.minPeakBaseline)
        self.smooth(self.minIndex+1, min(self.minIndex+30,self.wave.shape[0]-1))
        self.end10, self.end50, self.end90 = Qe(self.s_wave, self.minIndex, self.minPeakBaseline)
        begin = int(self.minIndex) - baselength
        end = int(self.minIndex) + afterlength
        if begin<0:
            begin = 0
        if end>self.wave.shape[0]:
            end = self.wave.shape[0]
        self.minPeakCharge = np.sum(self.minPeakBaseline - self.wave[begin:end])
        return self.minPeakCharge, self.begin90-self.begin10, self.end10-self.end90