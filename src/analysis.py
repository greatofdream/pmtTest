import numpy as np
'''
extract information from a numpy waveform
'''
def findPeakStd(waveform, nsimga=5, baselength=50):
    '''
    use baseline noise std* nsigma as threshold
    '''
    waveStd = np.std(waveform)
    peakflag = 0
    peakList = []
    baseline = np.average(waveform[0:baselength])
    threshold = baseline-waveStd*nsimga
    for i in range(waveform.shape[0]):
        # negative wave
        if waveform[i]<threshold:
            peakflag = 1
def integrateWave(wave):
    baseline = np.average(wave)
    return np.sum(baseline-wave)
def integrateMinPeakWave(wave, baselength=50):
    baseline = np.average(wave)
    minIndex = np.argmin(wave)
    begin = minIndex-baselength
    end = minIndex+baselength
    if minIndex-baselength<0:
        begin = 0
    if minIndex+baselength>wave.shape[0]:
        end = wave.shape[0]
    return np.sum(baseline-wave[begin:end]), baseline-wave[minIndex]
