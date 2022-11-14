# 积分charge时peak前后区间延展范围
baselength = 15
afterlength = 75
# 拟合时的区间范围，按照PE记录
peakspanl, peakspanr = 0.35, 0.35
vallyspanl, vallyspanr = 0.15, 0.25
# 绘图时zoom in, zoom_out 范围
peak_in_l, peak_in_r = 0, 50
# ADC转换mV因子
ADC2mV = 1000/1023
# SER长度
spelength = 150
spestart = 20
speend = spelength - spestart
# 前后脉冲时间区间
anapromptE = spestart
anadelay1B = 35
promptB = 250
promptE = 50
delay1B = 200
delay1E = 1000
delay10B = 1000
delay10E = 9800
# database
databaseDir = '/tarski/JNE/JinpingData/Jinping_1ton_Data/pmtTest'
PMTResultDir = 'ExPMT'
TestResultDir = 'ExResult'
TestSummaryPath = TestResultDir + '/TestSummary.csv'
# 后脉冲
searchwindowsMCP = [[200, 400], [500, 650], [1100,1300], [1700,2000]]

searchwindowsHama = [[200, 400], [1600,1900], [6000, 8000]]