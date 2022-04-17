import numpy as np
from .waveana import Waveana, interpolate
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
