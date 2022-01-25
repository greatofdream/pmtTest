import numpy as np
from scipy.optimize import minimize
from numba import jit
'''
use trigger channel to extract information from other numpy waveforms
'''
class waveAna(object):
    def __init__(self, wave=[], eid=0):
        self.eid = eid
        self.wave = wave
        self.meanBaseline = 0
        self.std = 0
        self.allCharge = 0
        self.minPeakCharge = 0
        self.minPeakBaseline = 0
        self.minPeakStd = 0
        self.end5mV = 0
        self.begin5mV = 0
        self.triggerTime = 0
        self.tpl = []
    def setTriggerWave(self, triggerWave, uprising=True):
        self.triggerWave = triggerWave
        self.triggerTime = self.getTriggerTime(100, 0.1, uprising)
        # print(self.triggerTime)
    def getTriggerTime(self, begin=100, threshold=0.1, uprising=True):
        baseline = np.average(self.triggerWave[0:100])
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
    def setWave(self, wave):
        self.wave = wave
        self.meanBaseline = 0
        self.std = 0
        self.allCharge = 0
        self.minPeakCharge = 0
        self.minPeakBaseline = np.average(wave)
        self.minPeakStd = np.std(wave)
        self.end5mV = 0
        self.begin5mV = 0
    def getBaseline(self, nsigma=5):
        std = np.std(self.wave)
        meanBaseline = np.average(self.wave)
        signalPos = np.where(self.wave<(meanBaseline-nsigma*std))[0]
        if signalPos.shape[0]>0:
            maxsignalPos = signalPos[-1]+10
            minsignalPos = signalPos[0]-10
            if maxsignalPos>self.wave.shape[0]:
                maxsignalPos=self.wave.shape[0]
            if minsignalPos<0:
                minsignalPos=0
        else:
            minsignalPos = 0
            maxsignalPos = 0
        if minsignalPos==0 and maxsignalPos==self.wave.shape[0]:
            cutWave = self.wave[100:500]
        else:
            cutWave = np.append(self.wave[0:minsignalPos],self.wave[maxsignalPos:self.wave.shape[0]])
        self.meanBaseline = np.average(cutWave)
        self.std = np.std(cutWave)
        return self.meanBaseline, self.std
    def getBaselineFine(self, minIndex, nsigma=5, expandWidth=500, desiredWidth=100):
        # extract the wave
        if minIndex< desiredWidth:
            begin = minIndex+100
            end = begin+expandWidth
        else:
            begin = minIndex - expandWidth
            end = minIndex - 10
        if begin<0:
            begin = 0
        if end>self.wave.shape[0]:
            #begin = self.wave.shape[0]-expandWidth# np wave in tail
            end = self.wave.shape[0]
        extractWave = self.wave[begin:end]
        #print(begin,end)
        self.hist = np.histogram(extractWave, bins=1000, range=[1,1001])[0]
        baselineEstimate = np.argmax(self.hist)+1
        stdl = np.max([0,baselineEstimate-6])
        stdr = np.min([self.hist.shape[0],baselineEstimate+5])
        for i in range(baselineEstimate-1, np.max([0,baselineEstimate-6]),-1):
            if self.hist[i]<(self.hist[baselineEstimate-1]/2):
                stdl=i
                break
        for i in range(baselineEstimate, np.min([self.hist.shape[0],baselineEstimate+6])):
            if self.hist[i]<(self.hist[baselineEstimate-1]/2):
                stdr=i
                break
        stdEstimate = (stdr-stdl)/2.355/2
        #print('b:{}; hist:{}'.format(baselineEstimate, self.hist[baselineEstimate-5:baselineEstimate+5]))
        # stdEstimate = np.std(self.hist)
        #print('first estimate std {}, {},{}'.format(stdEstimate, np.std(extractWave),np.std(self.hist)))
        signalPos = extractWave<(baselineEstimate-nsigma*stdEstimate)
        for i in np.where(signalPos)[0]:
            if signalPos[i]:
                if i<10:
                    signalPos[0:i] = True
                else:
                    signalPos[(i-10):i] = True
            elif signalPos[i-1]:
                if i>(extractWave.shape[0]-10):
                    signalPos[i:extractWave.shape[0]] = True
                else:
                    signalPos[i:(i+10)] = True
        cutWave = extractWave[~signalPos]
        # if cutWave.shape[0]<=10:
        cutWave2 = self.getSmallStdWave()
        if np.std(cutWave2)<np.std(cutWave) or cutWave.shape[0]<=10:
            cutWave = cutWave2
        baselineEstimate = np.average(cutWave)
        stdEstimate = np.std(cutWave)
        # [begin,end] is the interval of baseline
        self.begin = begin
        self.end = end
        if np.isnan(self.minPeakStd):
            print(cutWave)
            print(self.minPeakStd,self.minPeakBaseline)
            exit(1)
        cutWave0 = cutWave[(cutWave>=(baselineEstimate-nsigma*stdEstimate))&(cutWave<=(baselineEstimate+nsigma*stdEstimate))]
        self.minPeakBaseline = np.average(cutWave0)
        self.minPeakStd = np.std(cutWave0)
        if np.isnan(self.minPeakStd):
            print(cutWave)
            print(cutWave0)
            print(self.minPeakStd,self.minPeakBaseline)
            exit(1)
        return self.minPeakBaseline
    def getSmallStdWave(self, number=10):
        step = int(self.wave.shape[0]/10)
        stdArray = np.zeros(number)
        for i in range(number):
            stdArray[i] = np.std(self.wave[i*step:(i+1)*step])
        smallIndex = np.argmin(stdArray)
        return self.wave[smallIndex*step:(smallIndex+1)*step]
    def findPeakStd(self, npeak=2, nsigma=3, chargeThreshold=15, baselength=50):
        '''
        use baseline noise std* nsigma as threshold for whether signal. nthreshold for the second signal
        '''
        peakflag = 0
        peakList = []
        threshold = self.meanBaseline-self.std*nsigma
        self.modifyWave = self.wave+0
        # if self.minPeak<self.std*nsigma:
        #     return peakList
        # else:
        peakList.append(self.resolve(self.minIndex,nsigma))
        for n in range(npeak-1):
            # negative wave; discrimina by charge
            temp = self.resolve(np.argmin(self.modifyWave),nsigma)
            if temp[2]>chargeThreshold:
                peakList.append(temp)
        return peakList
    def resolve(self, peakPos, nsigma):
        peak = self.modifyWave[peakPos]
        # find the forehead baseline, otherwise back to find; start and end
        cursor = peakPos-1
        threshold = self.meanBaseline-self.std*nsigma
        while cursor>=100 and self.modifyWave[cursor]<=threshold:
            cursor = cursor - 1
        if cursor>= 100:
            baseline = np.average(self.modifyWave[(cursor-100):(cursor-10)])
            std = np.std(self.modifyWave[(cursor-100):(cursor-10)])
        else:
            cursor = peakPos+1
            while cursor<=(self.modifyWave.shape[0]-100) and self.modifyWave[cursor]<=threshold:
                cursor = cursor + 1
            baseline = np.average(self.modifyWave[(cursor+10):(cursor+100)])
            std = np.std(self.modifyWave[(cursor+10):(cursor+100)])
        # update threshold
        threshold = baseline - std*nsigma
        cursor = peakPos - 1
        while cursor>=0 and self.modifyWave[cursor]<=threshold:
            cursor = cursor - 1
        begin = cursor
        if begin>10:
            cBegin = begin-10
        else:
            cBegin = 0
        cursor = peakPos + 1
        while cursor<(self.modifyWave.shape[0]) and self.modifyWave[cursor]<=threshold:
                cursor = cursor + 1
        end = cursor
        if end<(self.modifyWave.shape[0]-10):
            cEnd = end + 10
        else:
            cEnd = self.modifyWave.shape[0]
        peakRiseTime = peakPos-begin
        peakDownTime = end - peakPos
        peakCharge = np.sum(baseline-self.modifyWave[cBegin:cEnd])
        self.modifyWave[begin:end] = baseline
        return (peakPos, peakCharge, peak, peakRiseTime, peakDownTime, baseline, std)
    def integrateWave(self):
        baseline = self.minPeakBaseline
        self.allCharge = np.sum(baseline-self.wave[int(self.triggerTime):])
        return self.allCharge
    def findQb(self,threshold=5):
        threshold10 = self.minPeakBaseline - (self.minPeakBaseline - self.wave[self.minIndex])/10
        threshold90 = self.minPeakBaseline - (self.minPeakBaseline - self.wave[self.minIndex])*0.9
        threshold50 = self.minPeakBaseline - (self.minPeakBaseline - self.wave[self.minIndex])*0.5
        begin90=self.minIndex
        begin10=0
        begin50 = int((begin90+begin10)/2)
        i=0
        for i in range(self.minIndex, 0,-1):
            if self.wave[i]>threshold90:
                begin90 = i
                break
        begin90=i
        for i in range(begin90, 0,-1):
            if self.wave[i]>threshold50:
                begin50 = i
                break
        begin50=i
        for i in range(begin50, 0,-1):
            if self.wave[i]>threshold10:
                begin10 = i
                break
        begin10=i
        offsetThreshold = self.minPeakBaseline - threshold
        if threshold<self.minPeak:
            for i in range(self.minIndex, 0,-1):
                if self.wave[i]>offsetThreshold:
                    self.begin5mV = i
                    break
            self.begin5mV = i
        else:
            self.begin5mV = self.minIndex
        return begin10, begin90, begin50
    def findQe(self, threshold=5):
        threshold10 = self.minPeakBaseline - (self.minPeakBaseline - self.wave[self.minIndex])/10
        threshold90 = self.minPeakBaseline - (self.minPeakBaseline - self.wave[self.minIndex])*0.9
        threshold50 = self.minPeakBaseline - (self.minPeakBaseline - self.wave[self.minIndex])*0.5
        end90=self.minIndex
        end10=self.wave.shape[0]
        end50 = int((end90+end10)/2)
        i=0
        for i in range(self.minIndex,self.wave.shape[0]):
            if self.wave[i]>threshold90:
                end90 = i
                break
        end90=i
        for i in range(end90,self.wave.shape[0]):
            if self.wave[i]>threshold50:
                end50 = i
                break
        end50 = i
        for i in range(end50, self.wave.shape[0]):
            if self.wave[i]>threshold10:
                end10 = i
                break
        end10=i
        offsetThreshold = self.minPeakBaseline - threshold
        if threshold<self.minPeak:
            for i in range(self.minIndex,self.wave.shape[0]):
                if self.wave[i]>offsetThreshold:
                    self.end5mV = i
                    break
            self.end5mV = i
        else:
            self.end5mV = self.minIndex
        return end10, end90, end50

    def integrateMinPeakWave(self, baselength=0):
        self.minIndex, self.nearMax, self.nearPositiveMean, self.nearPositiveStd = findNearMax(self.wave,self.triggerTime, self.minPeakBaseline)
        self.minPeak = self.minPeakBaseline-self.wave[self.minIndex]
        if self.minPeak<0:
            return 0, 0, 0
        self.begin10, self.begin50, self.begin90 = Qb(self.wave, self.minIndex, self.minPeakBaseline)
        self.end10, self.end50, self.end90 = Qe(self.wave, self.minIndex, self.minPeakBaseline)
        self.begin5mV = bxmV(self.wave, self.minIndex, self.minPeakBaseline)
        self.end5mV = exmV(self.wave, self.minIndex, self.minPeakBaseline)
        begin = int(self.begin10)-baselength
        end = int(self.end10)+baselength
        if begin<0:
            begin = 0
        if end>self.wave.shape[0]:
            end = self.wave.shape[0]
        self.minPeakCharge = np.sum(self.minPeakBaseline - self.wave[begin:end])
        return self.minPeakCharge, self.begin90-self.begin10, self.end10-self.end90
    def fit(self):
        begin = int(self.begin10-50)
        end = int(self.end10+20)
        cutWave = self.wave[begin:end]-self.minPeakBaseline
        parList = [[i,-self.minPeak/(np.min(self.tpl)),-0.1] for i in range(self.minIndex-begin-10,self.minIndex-begin,2)]
        fitResult = []
        failResult = []
        for par in parList:
            tempResult = minimize(self.likelihood, par, method='SLSQP', bounds=((0,end-begin),(0.1,100),(-10,-0.01)),args=cutWave,options={'eps':0.1, 'maxiter':500})
            if tempResult.success:
                fitResult.append(tempResult)
            else:
                failResult.append(tempResult)
        if len(fitResult)>0:
            return min(fitResult,key=lambda x: x.fun)
        else:
            print('failed fit eid: {}'.format(self.eid))
            return min(failResult,key=lambda x: x.fun)
    def likelihood(self, paras, *args):
        cutwave = args[0]
        expectHit = np.zeros((cutwave.shape[0],))
        expectHit += addTpl(expectHit, self.tpl, paras[0], paras[1],100,cutwave.shape[0])
        expectHit += paras[2]
        L = np.sum((-cutwave+expectHit)**2)
        return L
    def getwave(self,paras,tpl):
        begin = int(self.begin10-50)
        end = int(self.end10+10)
        expectHit = np.zeros((end-begin,))
        expectHit += addTpl(expectHit, tpl, paras[0], paras[1],100,end-begin)
        print(expectHit)
        expectHit += paras[2]+self.minPeakBaseline
        return expectHit
@jit(nopython=True)
def findNearMax(wave, triggerTime, baseline, length=10):
    minIndex = np.argmin(wave[int(triggerTime):])+int(triggerTime)
    end = minIndex+length
    if end>wave.shape[0]:
        end = wave.shape[0]
    extractWave = wave[(minIndex-length):end]-baseline
    extractWave2 = extractWave[extractWave>0]
    if extractWave2.shape[0]>0:
        nearPositiveMean = np.mean(extractWave2)
        nearPositiveStd = np.std(extractWave2)
        return minIndex, np.max(extractWave), nearPositiveMean, nearPositiveStd
    else:
        return minIndex, 0, 0 , 0
# @jit(nopython=True)
# def findMinIndex(wave):
#     return np.argmin(wave)
# @jit(nopython=True)
# def intMinPeak(wave, minIndex, baseline, begin, end, length=10):
#     minPeak = baseline - wave[minIndex]
#     return minPeak
@jit(nopython=True)
def Qb(wave, minIndex, minPeakBaseline):
    threshold10 = minPeakBaseline - (minPeakBaseline - wave[minIndex])/10
    threshold90 = minPeakBaseline - (minPeakBaseline - wave[minIndex])*0.9
    threshold50 = minPeakBaseline - (minPeakBaseline - wave[minIndex])*0.5
    begin = np.array([0.0,minIndex/2,minIndex])
    threshold = np.array([threshold10,threshold50,threshold90])
    cursor = minIndex
    for j in range(threshold.shape[0],0,-1):
        for i in range(cursor, 0, -1):
            if wave[i]>threshold[j-1]:
                begin[j-1] = i+1-interpolate(threshold[j-1],wave[i+1],wave[i])
                cursor = i
                break
    return begin[0], begin[1], begin[2]
@jit(nopython=True)
def bxmV(wave, minIndex, minPeakBaseline, threshold=5):
    offsetThreshold = minPeakBaseline - threshold
    minPeak = minPeakBaseline - wave[minIndex]
    if threshold<minPeak:
        cursor = minIndex
        for i in range(minIndex, 0,-1):
            if wave[i]>offsetThreshold:
                cursor = i
                break
        return cursor
    else:
        return minIndex
@jit(nopython=True)
def exmV(wave, minIndex, minPeakBaseline, threshold=5):
    offsetThreshold = minPeakBaseline - threshold
    minPeak = minPeakBaseline - wave[minIndex]
    if threshold<minPeak:
        cursor = minIndex
        for i in range(minIndex, wave.shape[0],1):
            if wave[i]>offsetThreshold:
                cursor = i
                break
        return cursor
    else:
        return minIndex
@jit(nopython=True)
def Qe(wave, minIndex, minPeakBaseline):
    threshold10 = minPeakBaseline - (minPeakBaseline - wave[minIndex])/10
    threshold90 = minPeakBaseline - (minPeakBaseline - wave[minIndex])*0.9
    threshold50 = minPeakBaseline - (minPeakBaseline - wave[minIndex])*0.5
    end = np.array([wave.shape[0]+0.0,(wave.shape[0]+minIndex)/2,minIndex])
    threshold = np.array([threshold10,threshold50,threshold90])
    cursor = minIndex
    for j in range(threshold.shape[0],0,-1):
        for i in range(cursor, wave.shape[0], 1):
            if wave[i]>threshold[j-1]:
                end[j-1] = i-1+interpolate(threshold[j-1],wave[i-1],wave[i])
                cursor = i
                break
    return end[0], end[1], end[2]
@jit(nopython=True)
def interpolate(v,l,r):
    # interpolate v between [l,r]
    return (v-l)/(r-l)

@jit(nopython=True)
def addTpl(wave, tpl, t0, A, tplLength, fitlength=500, zoom=1):
    end = np.int(np.ceil(t0)+tplLength*zoom)-1
    if end >fitlength:
        end = fitlength-1
    if zoom==1:
        percent = t0 - np.floor(t0)
        for ti in range(np.int(np.ceil(t0)),end):
            interpB = ti - np.int(np.ceil(t0))
            # print('interB:{};percent:{}'.format(interpB, percent))
            wave[ti] += (tpl[interpB+1]*percent +tpl[interpB]*(1-percent)) * A
    else:
        for ti in range(np.int(np.ceil(t0)),end):
            interpB = np.int(np.floor((ti - t0)/zoom))
            percent = (ti - t0)/zoom-interpB
            # print('interB:{};percent:{}'.format(interpB, percent))
            wave[ti] += (tpl[interpB+1]*percent +tpl[interpB]*(1-percent)) * A/zoom
    return wave