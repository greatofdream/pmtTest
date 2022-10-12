'''
分析程序杂项
'''
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
import numdifftools as nd
import ROOT
class RootFit():
    def setFunc(self, func, x0):
        self.func = func
        self.func.SetParameters(x0)
    def setHist(self, bins, counts):
        self.hists = ROOT.TH1D("", "", len(bins)-1, bins)
        for i in range(len(bins) - 1):
            self.hists.SetBinContent(i, counts[i])
    def Fit( self ):
        self.hists.Fit(self.func, 'R')
        return self.func.GetParameters(), self.func.GetParErrors()

def centralMoment(xs, xmean, k):
    return np.sum((xs - np.mean(xs))**k) / (len(xs) - 1)
def Hessian(f, x, step):
    df = nd.Hessian(f, step=step)
    return df(x)
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
    return np.concatenate([x[:pad], x0, x[-pad:]])
def likelihood(x, *args):
    mu, sigma = x
    tts, N = args
    return np.sum((tts-mu)**2)/2/sigma**2 + N * np.log(sigma)
def likelihoodB(x, *args):
    # pdf: A * Gaussian + (1-A) * Background
    A, mu, sigma = x
    tts, T = args
    return -np.sum(np.log(np.exp(-(tts - mu)**2 / 2 / sigma**2) * A / np.sqrt(2*np.pi) / sigma + (1 - A) / T))

def fitGaus(tts, limits):
    tts_select = tts[(tts<limits[1])&(tts>limits[0])]
    results = [minimize(likelihood, [np.mean(tts_select), tts_sigma], args=(tts_select, tts_select.shape[0]),
        bounds=[limits, (0,(limits[1]-limits[0])/2)])
        for tts_sigma in np.arange(0.2, 3, 0.1)
        ]
    result = min(results, key=lambda x:x.fun)
    return result, tts_select.shape[0]
def fitGausB(tts, limits, timeLength, b_u):
    results = [minimize(
        likelihoodB,
        [0.999, np.mean(tts), tts_sigma],
        args=(tts, timeLength),
        bounds=[
            (1-b_u, 1), limits, (0.1, (limits[1]-limits[0])/2)
            ],
        method='SLSQP',
        options={'eps':0.0001}
            ) for tts_sigma in np.arange(0.2, 3, 0.05)]
    result = min(results, key=lambda x:x.fun)
    return result
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
    result = minimize(likelihoodSER, [mu0, 2, 5, 8], args=(xs, ys), bounds=[
        (xs[0], mu0 + 20), (0.5, 10), (1, 20), (1, 2000)
    ], constraints=({
        'type': 'ineq', 'fun': lambda x: 2 + x[2] - x[1]}
    ),
    options={'eps':0.0001},
    method='SLSQP')
    return result

def peakfind(ys, thresholds, padding=2):
    ys = smooth(ys, 3)
    sig = ys > thresholds
    localmax = np.zeros(ys.shape, dtype=bool)
    # 寻找不会小于其它部分的长度为5的平台
    localmax[padding:-padding] = (ys[padding:-padding] >= ys[(padding-1):(-padding-1)]) & (ys[padding:-padding] >= ys[(padding-2):(-padding-2)]) & (ys[padding:-padding] >= ys[(padding+1):(-padding+1)]) & (ys[padding:-padding] >= ys[(padding+2):])
    candidate = sig & localmax
    return candidate
def vallyfind(ys, height=0.5):
    # 谷检测,当两个峰较高，且谷的位置小于峰高的x倍
    peak = np.max(ys)
    threshold = peak * height
    peakArea = (ys>threshold).astype(int)
    vallycandidate = np.where(np.abs(peakArea[1:] - peakArea[:-1])==1)[0]
    vallyn = vallycandidate.shape[0]
    if peakArea[vallycandidate[0]] == -1:
        vallyn += 1
    if peakArea[vallycandidate[-1]] == 1:
        vallyn += 1
    return vallyn//2 - 1
def peakNum(ys, std, padding=2):
    # 局域极值，信号判定
    candidate = peakfind(ys, 5*std, padding).astype(int)
    peakn = np.sum((candidate[1:] - candidate[:-1])==1)
    if peakn == 1:
        peakn += vallyfind(ys)
    return peakn
def getIntervals(xs, ys, thresholds, pre_ser_length, after_ser_length, padding=2):
    # 局域极值，信号判定
    candidate = peakfind(ys, thresholds, padding)
    # 按照ser的长度影响进行扩展
    i = 0
    end = len(ys)
    while i < end:
        if candidate[i]:
            candidate[max(i-pre_ser_length, 0):min(i+after_ser_length, end)] = True
            i = i+after_ser_length
        i += 1
    # 第一位和最后一位强制为False，避免区间不闭合
    candidate[-1] = False
    candidate[0] = False
    candidate = candidate.astype(int)
    # 分割波形
    indexs = np.where(np.abs(candidate[1:] - candidate[:-1])==1)[0].reshape((-1,2))
    # 处理接近3mV的未识别的波形
    if len(indexs) == 0:
        indexs = np.array([max(np.argmax(ys) - pre_ser_length, 0), min(np.argmax(ys) + after_ser_length, end-1)]).reshape((-1,2))
    return xs[indexs]
def getTQ(xs, ys, ser):
    ts = xs[np.argmax(ys)]
    qs = np.sum(ys)
    return ts, qs
def likelihoodAt(para, *args):
    n, sumlogn, mu = args
    pelist = np.zeros(n, dtype=[('HitPosInWindow', np.float64), ('Charge', np.float64)])
    # for i in range(n):
    #     pelist[i] = (para[i*2+1], para[i*2]*jppara.peakpara[1])
    # expecty = jppara.genwave(pelist,500,False)
    # # -log likelihood
    # logL = np.sum((self.y-expecty)**2)/2/jppara.baselinerms**2
    # dlogL = 0
    # for i in range(n):
    #     dlogL += jppara.charge50Pdflog(para[i*2]*jppara.peakpara[1])
    # logL -= dlogL
    # # print(logL-np.sum(self.y-expecty)**2)
    # # logL += sumlogn-n*np.log(mu)
    # logL += sumlogn
    # print(sumlogn-n*np.log(mu))
    return 0#logL
# def wavefit():
