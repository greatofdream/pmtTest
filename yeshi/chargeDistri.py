import h5py, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

h2104 = np.loadtxt('2104-9107.txt')
heights = pd.read_excel('MCPHeight-SPE.xlsx')

xminorLocator = MultipleLocator(10)
yminorLocator = MultipleLocator(10)
    
fig, ax = plt.subplots(dpi=150)
ax.plot(heights.iloc[0:400,[1,3,5]],label=[i.split('-')[-1] for i in heights.keys()[[1,3,5]]])
ax.plot(h2104[0:400],label='PM2104-9107-50nm'.split('-')[-1])
ax.set_ylim([0,400])
ax.set_xlabel('Charge/25fC')
ax.set_ylabel('Entries')
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.legend()
plt.savefig('heights_select.pdf')

fig, ax = plt.subplots(dpi=150)
ax.plot(np.arange(50,450),heights.iloc[50:450,[1,3,5]].values/np.max(heights.iloc[130:200,[1,3,5]].values,axis=0)*300,label=[i.split('-')[-1] for i in heights.keys()[[1,3,5]]])
ax.plot(np.arange(50,450),h2104[50:450]/np.max(h2104[110:200])*300,label='PM2104-9107-50nm'.split('-')[-1])
ax.set_ylim([0,400])
ax.set_xlim([50,450])
ax.set_xlabel('Charge/25fC')
ax.set_ylabel('Normalized Entries')
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.legend()
plt.savefig('heights_select_normalize.pdf')

fig, ax = plt.subplots(dpi=150)
ax.plot((np.arange(50,450)-100)*1.25,heights.iloc[50:450,[1,3,5]].values/np.max(heights.iloc[130:200,[1,3,5]].values,axis=0)*300,label=[i.split('-')[-1] for i in heights.keys()[[1,3,5]]])
ax.plot((np.arange(50,450)-100)*1.25,h2104[50:450]/np.max(h2104[110:200])*300,label='PM2104-9107-50nm'.split('-')[-1])
ax.set_ylim([0,400])
ax.set_xlim([-50,450])
ax.set_xlabel('charge/[ADC$\cdot$ns]')
ax.set_ylabel('normalize entries')
ax.xaxis.set_minor_locator(xminorLocator)
ax.yaxis.set_minor_locator(yminorLocator)
ax.legend()
plt.savefig('heights_select_normalize_adcns.pdf')