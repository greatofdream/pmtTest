import numpy as np, h5py
import pandas as pd
import argparse
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', nargs='+', help='input h5 file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    psr.add_argument('-c', dest='channel', default=2, type=int, help='channel for test pmt')
    psr.add_argument('--ref', default=3, type=int, help='channel for refer pmt')
    args = psr.parse_args()
    testR, refR = 1, 1
    for f in args.ipt:
        with h5py.File(f, 'r') as ipt:
            QEinfo = pd.DataFrame(ipt['QEinfo'][:])
            QEinfo = QEinfo.set_index('ch')
            testR *= QEinfo.loc[args.channel]['ratio']
            refR *= QEinfo.loc[args.ref]['ratio']
    print(np.sqrt(testR/refR))