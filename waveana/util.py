'''
分析程序杂项
'''
import numpy as np
from scipy.optimize import minimize
def findFirstRisingEdge(edge, counts, window=(2, 10)):
    # 寻找peak分布第一个上升沿
    i = 0
    for i in range(window[0], window[1]):
        if counts[i+1]<counts[i]:
            continue
        else:
            break
    return int(edge[i]), counts[i]
def peakResidual(x, *args):
    # modified least=squares method: gaussian function
    A, mu, sigma = x
    counts, bins = args
    return np.sum((A*np.exp(-(bins-mu)**2/2/sigma**2)-counts)**2/counts)
def vallyResidual(x, *args):
    # modified least=squares method: parabolic
    a, b, c = x
    counts, bins = args
    return np.sum((a/100*(bins - b)**2 + c - counts)**2/counts)

def smooth(x, n=7):
    pad = (n-1)//2
    x0 = np.zeros(x.shape[0] - pad*2)
    for i in range(pad, x.shape[0]-pad):
        x0[i-pad] = np.average(x[(i-pad):(i+pad+1)])
    return x0
def likelihood(x,*args):
    A,mu,sigma = x
    tts,N = args
    return A*N-tts.shape[0]*np.log(A)+np.sum((tts-mu)**2)/2/sigma**2+tts.shape[0]*np.log(sigma)

def fitGaus(tts,limits):
    tts_select = tts[(tts<limits[1])&(tts>limits[0])]
    result = minimize(likelihood,[1, np.mean(tts_select),np.std(tts_select)],args=(tts_select, tts_select.shape[0]), bounds=[(0,None),limits,(0,(limits[1]-limits[0])/2)])
    return result, tts_select.shape[0]