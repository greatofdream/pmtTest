import argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import config
def readCompareDat(f):
    data = np.loadtxt(f)
    # data = np.transpose(data.reshape((data.shape[0], 2, data.shape[1]//2)), (1, 0, 2))
    # return np.array(data[:, :, [0,2]].reshape((2,-1)), dtype=[('X', np.float64), ('X_sigma', np.float64), ('Y', np.float64), ('Y_sigma', np.float64), ('Z', np.float64), ('Z_sigma', np.float64), ('E', np.float64), ('E_sigma', np.float64), ('Neff', np.float64),  ('Neff_sigma', np.float64)]
    # data = data[:, [0, 2, 3, 5]]
    return data
psr = argparse.ArgumentParser()
psr.add_argument('--phase', default='sk7', help='phases')
psr.add_argument('-i', dest='ipt', help='summary csv file')
psr.add_argument('-f', dest='file', nargs='+', help='compare dat files')
psr.add_argument('-o', dest='opt', help='output pdf file')
args = psr.parse_args()
runs_df = pd.read_csv(args.ipt, names=config.summary_header)
runs_df['RunNo'] = runs_df['RunNo'].astype(pd.Int64Dtype())
runs_df['X'] = runs_df['X'].astype(pd.Int64Dtype())
selectRuns_df = runs_df[runs_df['Mode']==0]
filenames = args.file
merge = np.zeros((selectRuns_df.shape[0]), dtype=[('RunNo', int), ('X', int), ('Y', int), ('Z', int), ('E', int), ('X_data', np.float64), ('Y_data', np.float64), ('Z_data', np.float64), ('E_data', np.float64), ('Neff_data', np.float64), ('X_mc', np.float64), ('Y_mc', np.float64), ('Z_mc', np.float64), ('E_mc', np.float64), ('Neff_mc', np.float64)])
mergeError = np.zeros((selectRuns_df.shape[0]), dtype=[('RunNo', int), ('X', int), ('Y', int), ('Z', int), ('E', int), ('X_data', np.float64), ('Y_data', np.float64), ('Z_data', np.float64), ('E_data', np.float64), ('Neff_data', np.float64), ('X_mc', np.float64), ('Y_mc', np.float64), ('Z_mc', np.float64), ('E_mc', np.float64), ('Neff_mc', np.float64)])
mergeSigma = np.zeros((selectRuns_df.shape[0]), dtype=[('RunNo', int), ('X', int), ('Z', int), ('E', int), ('X_data', np.float64), ('Y_data', np.float64), ('Z_data', np.float64), ('E_data', np.float64), ('Neff_data', np.float64), ('X_mc', np.float64), ('Y_mc', np.float64), ('Z_mc', np.float64), ('E_mc', np.float64), ('Neff_mc', np.float64)])
for i, ((index, row), f) in enumerate(zip(selectRuns_df.iterrows(), filenames)):
    assert str(row['RunNo']) in f
    tmp = readCompareDat(f)
    merge[i] = (row['RunNo'], row['X']*100, 0, row['Z']*100, row['E'], *tmp[:,0], *tmp[:,3])
    mergeError[i] = (row['RunNo'], row['X']*100, 0, row['Z']*100, row['E'], *tmp[:,1], *tmp[:,4])
    mergeSigma[i] = (row['RunNo'], row['X'], row['Z'], row['E'], *tmp[:,2], *tmp[:,5])
with PdfPages(args.opt) as pdf:
    xs = np.arange(merge.shape[0])
    fig, axs = plt.subplots(5, 1, figsize=(12, 6), sharex=True)
    handles_mc, handles_data = [], []
    [handles_data.append(axs[i].scatter(xs, merge[j], label=j, s=20, color='k')) for i,j in enumerate(['X_data', 'Y_data', 'Z_data', 'E_data', 'Neff_data'])]
    [handles_mc.append(axs[i].scatter(xs, merge[j], label=j, s=20, color='r')) for i,j in enumerate(['X_mc', 'Y_mc', 'Z_mc', 'E_mc', 'Neff_mc'])]
    [axs[i].set_ylabel(j) for i, j in enumerate(['X[cm]', 'Y[cm]', 'Z[cm]', 'E[MeV]', 'Neff'])]
    axs[0].legend(handles=[handles_data[0], handles_mc[0]], labels=['Data', 'MC'])
    axs[4].set_xticks(xs)
    axs[4].set_xticklabels(merge['RunNo'], rotation=90)
    [axs[i].tick_params('x', labelbottom=False) for i in range(4)]
    fig.subplots_adjust(hspace=0)
    pdf.savefig(fig)
    plt.close()
    
    Nsubfig = 4
    fig, axs = plt.subplots(Nsubfig, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [0.2, 2, 2, 2,]})
    axs[0].axis('off')
    Xs= np.unique(selectRuns_df['X'])
    for x in Xs:
        Xs_index = np.where(selectRuns_df['X']==x)[0]
        axs[0].fill_between([Xs_index[0]-0.3, Xs_index[-1]+0.3], [1, 1], [2, 2], alpha=0.5, color='gray') 
        axs[0].text((Xs_index[0]+Xs_index[-1])/2, 1.5, 'X='+str(x)+'m', ha='center', va='center')
        Zs = np.unique(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)])
        for z in Zs:
            Zs_index = np.where(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)]==z)[0]
            axs[0].fill_between([Xs_index[0]+Zs_index[0]-0.3, Xs_index[0]+Zs_index[-1]+0.3], [0, 0], [1, 1], alpha=0.5, color='violet')
            axs[0].text(Xs_index[0]+(Zs_index[0]+Zs_index[-1])/2, 0.5, 'Z='+str(z)+'m', ha='center', va='center')
    handles_mc, handles_data = [], []
    [handles_data.append(axs[i+1].errorbar(xs, merge[j]-merge[k], yerr=mergeError[j], label=j, fmt='.', color='k')) for i,(j,k) in enumerate(zip(['X_data', 'Y_data', 'Z_data'], ['X', 'Y', 'Z']))]
    [handles_mc.append(axs[i+1].errorbar(xs, merge[j]-merge[k], yerr=mergeError[j], label=j, fmt='.', color='r')) for i,(j,k) in enumerate(zip(['X_mc', 'Y_mc', 'Z_mc'], ['X', 'Y', 'Z']))]
    [axs[i+1].set_ylabel(j) for i, j in enumerate(['X-$X_{lin}$[cm]', 'Y-$Y_{lin}$[cm]', 'Z-$Z_{lin}$[cm]'])]
    axs[1].legend(handles=[handles_data[0], handles_mc[0]], labels=['Data', 'MC'])
    axs[Nsubfig-1].set_xticks(xs)
    axs[Nsubfig-1].set_xticklabels(merge['RunNo'], rotation=90)
    axs[Nsubfig-1].set_xlim([xs[0]-0.3, xs[-1]+0.3])
    [axs[i].tick_params('x', labelbottom=False) for i in range(Nsubfig-1)]
    fig.subplots_adjust(hspace=0)
    pdf.savefig(fig)
    plt.close()
    
    Nsubfig = 4
    fig, axs = plt.subplots(Nsubfig, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [0.2, 2, 2, 2,]})
    axs[0].axis('off')
    Xs= np.unique(selectRuns_df['X'])
    for x in Xs:
        Xs_index = np.where(selectRuns_df['X']==x)[0]
        axs[0].fill_between([Xs_index[0]-0.3, Xs_index[-1]+0.3], [1, 1], [2, 2], alpha=0.5, color='gray') 
        axs[0].text((Xs_index[0]+Xs_index[-1])/2, 1.5, 'X='+str(x)+'m', ha='center', va='center')
        Zs = np.unique(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)])
        for z in Zs:
            Zs_index = np.where(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)]==z)[0]
            axs[0].fill_between([Xs_index[0]+Zs_index[0]-0.3, Xs_index[0]+Zs_index[-1]+0.3], [0, 0], [1, 1], alpha=0.5, color='violet')
            axs[0].text(Xs_index[0]+(Zs_index[0]+Zs_index[-1])/2, 0.5, 'Z='+str(z)+'m', ha='center', va='center')
    handles_mc, handles_data = [], []
    [handles_data.append(axs[i+1].errorbar(xs, merge[j]-merge[k], yerr=mergeError[j], label=j, fmt='.', color='k')) for i,(j,k) in enumerate(zip(['X_data', 'Y_data', 'Z_data'], ['X_mc', 'Y_mc', 'Z_mc']))]
    [axs[i+1].set_ylabel(j) for i, j in enumerate(['$X_{\mathrm{Data}}-X_{\mathrm{MC}}$[cm]', '$Y_{\mathrm{Data}}-Y_{\mathrm{MC}}$[cm]', '$Z_{\mathrm{Data}}-Z_{\mathrm{MC}}$[cm]'])]
    axs[Nsubfig-1].set_xticks(xs)
    axs[Nsubfig-1].set_xticklabels(merge['RunNo'], rotation=90)
    axs[Nsubfig-1].set_xlim([xs[0]-0.3, xs[-1]+0.3])
    [axs[i].tick_params('x', labelbottom=False) for i in range(Nsubfig-1)]
    fig.subplots_adjust(hspace=0)
    pdf.savefig(fig)
    plt.close()
    
    Nsubfig = 4
    fig, axs = plt.subplots(Nsubfig, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [0.2, 2, 2, 2,]})
    axs[0].axis('off')
    Xs= np.unique(selectRuns_df['X'])
    for x in Xs:
        Xs_index = np.where(selectRuns_df['X']==x)[0]
        axs[0].fill_between([Xs_index[0]-0.3, Xs_index[-1]+0.3], [1, 1], [2, 2], alpha=0.5, color='gray') 
        axs[0].text((Xs_index[0]+Xs_index[-1])/2, 1.5, 'X='+str(x)+'m', ha='center', va='center')
        Zs = np.unique(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)])
        for z in Zs:
            Zs_index = np.where(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)]==z)[0]
            axs[0].fill_between([Xs_index[0]+Zs_index[0]-0.3, Xs_index[0]+Zs_index[-1]+0.3], [0, 0], [1, 1], alpha=0.5, color='violet')
            axs[0].text(Xs_index[0]+(Zs_index[0]+Zs_index[-1])/2, 0.5, 'Z='+str(z)+'m', ha='center', va='center')
    handles_mc, handles_data = [], []
    [handles_data.append(axs[i+1].scatter(xs, mergeSigma[j], label=j, s=20, color='k')) for i,j in enumerate(['X_data', 'Y_data', 'Z_data'])]
    [handles_mc.append(axs[i+1].scatter(xs, mergeSigma[j], label=j, s=20, color='r')) for i,j in enumerate(['X_mc', 'Y_mc', 'Z_mc'])]
    [axs[i+1].set_ylabel(j) for i, j in enumerate(['$\sigma(X)$[cm]', '$\sigma(Y)$[cm]', '$\sigma(Z)$[cm]',])]
    [axs[i+1].text(0.05, 0.95, 'E[$\sigma({})$]={:.2f}\nE[$\sigma({})$]={:.2f}'.format(j+'_{Data}', np.mean(mergeSigma[j+'_data']), j+'_{MC}', np.mean(mergeSigma[j+'_mc'])), transform=axs[i+1].transAxes, ha='left', va='top') for i, j in enumerate(['X', 'Y', 'Z'])]
    axs[1].legend(handles=[handles_data[0], handles_mc[0]], labels=['Data', 'MC'])
    axs[Nsubfig-1].set_xticks(xs)
    axs[Nsubfig-1].set_xticklabels(merge['RunNo'], rotation=90)
    axs[Nsubfig-1].set_xlim([xs[0]-0.3, xs[-1]+0.3])
    [axs[i].tick_params('x', labelbottom=False) for i in range(Nsubfig-1)]
    fig.subplots_adjust(hspace=0)
    pdf.savefig(fig)
    plt.close()

    Nsubfig = 3
    fig, axs = plt.subplots(Nsubfig, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [0.2, 2, 2]})
    axs[0].axis('off')
    Xs= np.unique(selectRuns_df['X'])
    for x in Xs:
        Xs_index = np.where(selectRuns_df['X']==x)[0]
        axs[0].fill_between([Xs_index[0]-0.3, Xs_index[-1]+0.3], [1, 1], [2, 2], alpha=0.5, color='gray') 
        axs[0].text((Xs_index[0]+Xs_index[-1])/2, 1.5, 'X='+str(x)+'m', ha='center', va='center')
        Zs = np.unique(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)])
        for z in Zs:
            Zs_index = np.where(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)]==z)[0]
            axs[0].fill_between([Xs_index[0]+Zs_index[0]-0.3, Xs_index[0]+Zs_index[-1]+0.3], [0, 0], [1, 1], alpha=0.5, color='violet')
            axs[0].text(Xs_index[0]+(Zs_index[0]+Zs_index[-1])/2, 0.5, 'Z='+str(z)+'m', ha='center', va='center')
    handles_mc, handles_data = [], []
    [handles_data.append(axs[i+1].errorbar(xs, merge[j], yerr=mergeError[j], label=j, fmt='.', color='k')) for i,(j,k) in enumerate(zip(['E_data', 'Neff_data'], ['E', 'E']))]
    [handles_mc.append(axs[i+1].errorbar(xs, merge[j], yerr=mergeError[j], label=j, fmt='.', color='r')) for i,(j,k) in enumerate(zip(['E_mc', 'Neff_mc'], ['E', 'E']))]
    [axs[i+1].set_ylabel(j) for i, j in enumerate(['E[MeV]', '$N_{eff}$'])]
    axs[1].legend(handles=[handles_data[0], handles_mc[0]], labels=['Data', 'MC'])
    axs[Nsubfig-1].set_xticks(xs)
    axs[Nsubfig-1].set_xticklabels(merge['RunNo'], rotation=90)
    axs[Nsubfig-1].set_xlim([xs[0]-0.3, xs[-1]+0.3])
    [axs[i].tick_params('x', labelbottom=False) for i in range(Nsubfig-1)]
    fig.subplots_adjust(hspace=0)
    pdf.savefig(fig)
    plt.close()

    Nsubfig = 3
    fig, axs = plt.subplots(Nsubfig, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [0.2, 2, 2]})
    axs[0].axis('off')
    Xs= np.unique(selectRuns_df['X'])
    for x in Xs:
        Xs_index = np.where(selectRuns_df['X']==x)[0]
        axs[0].fill_between([Xs_index[0]-0.3, Xs_index[-1]+0.3], [1, 1], [2, 2], alpha=0.5, color='gray') 
        axs[0].text((Xs_index[0]+Xs_index[-1])/2, 1.5, 'X='+str(x)+'m', ha='center', va='center')
        Zs = np.unique(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)])
        for z in Zs:
            Zs_index = np.where(selectRuns_df['Z'][Xs_index[0]:(Xs_index[-1]+1)]==z)[0]
            axs[0].fill_between([Xs_index[0]+Zs_index[0]-0.3, Xs_index[0]+Zs_index[-1]+0.3], [0, 0], [1, 1], alpha=0.5, color='violet')
            axs[0].text(Xs_index[0]+(Zs_index[0]+Zs_index[-1])/2, 0.5, 'Z='+str(z)+'m', ha='center', va='center')
    handles_mc, handles_data = [], []
    Es = np.unique(selectRuns_df['E'])
    Es_marker = ['.', '+', 'x', '^']
    for E, E_marker in zip(Es, Es_marker):
        select = selectRuns_df['E']==E
        [handles_data.append(axs[i+1].scatter(xs[select], mergeSigma[j][select]/merge[j][select], label=j+'_{}MeV'.format(E), s=20, marker=E_marker, color='k')) for i,j in enumerate(['E_data', 'Neff_data'])]
        [handles_mc.append(axs[i+1].scatter(xs[select], mergeSigma[j][select]/merge[j][select], label=j+'_{}MeV'.format(E), s=20, marker=E_marker, color='r')) for i,j in enumerate(['E_mc', 'Neff_mc'])]
    [axs[i+1].set_ylabel(j) for i, j in enumerate(['$\sigma(E)/\mu(E)$', '$\sigma(N_{eff})/\mu(N_{eff})$'])]
    axs[1].legend(handles=handles_data[::2], labels=['Data_{}MeV'.format(i) for i in Es])
    axs[2].legend(handles=handles_mc[::2], labels=['MC_{}MeV'.format(i) for i in Es])
    axs[Nsubfig-1].set_xticks(xs)
    axs[Nsubfig-1].set_xticklabels(merge['RunNo'], rotation=90)
    axs[Nsubfig-1].set_xlim([xs[0]-0.3, xs[-1]+0.3])
    [axs[i].tick_params('x', labelbottom=False) for i in range(Nsubfig-1)]
    fig.subplots_adjust(hspace=0)
    pdf.savefig(fig)
    plt.close()

