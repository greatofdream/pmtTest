'''
分析程序杂项
'''
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
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
# SER function fit
def SER(x, xs):
    mu, sigma, tau, A= x
    xs = xs - mu
    co = (sigma/tau) **2/2-np.log(2*tau)
    x_erf = (sigma **2 /tau - xs) / (np.sqrt(2.) * sigma)
    eys = A * np.exp(co - xs/tau) * (1.0 - erf(x_erf))
    return eys
def likelihoodSER(x, *args):
    # binned likelihood of SER for least square
    # args = (xs, ys)
    xs, ys = args
    eys = SER(x, xs)
    return np.sum((ys-eys)**2)

def fitSER(xs, ys):
    mu0 = xs[np.argmax(ys)]
    result = minimize(likelihoodSER, [mu0, 5, 10, 8], args=(xs, ys), bounds=[
        (xs[0], mu0 + 20), (0.5, 10), (1, 20), (1, 1000)
    ], constraints=({
        'type': 'ineq', 'fun': lambda x: x[2] - x[1]}
    ),
    method='SLSQP')
    return result

def peakNum(ys, std, padding=2):
    # 局域极值，信号判定
    ys = smooth(ys, 3)
    sig = ys > 5*std
    localmax = np.zeros(ys.shape, dtype=bool)
    localmax[padding:-padding] = (ys[padding:-padding] >= ys[(padding-1):(-padding-1)]) & (ys[padding:-padding] > ys[(padding-2):(-padding-2)]) & (ys[padding:-padding] >= ys[(padding+1):(-padding+1)]) & (ys[padding:-padding] > ys[(padding+2):])
    candidate = (sig&localmax).astype(int)
    return np.sum((candidate[1:] - candidate[:-1])==1)