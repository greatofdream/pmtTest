import numpy as np, h5py
import argparse
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input text file')
    psr.add_argument('-o', dest='opt', help='output text file')
    psr.add_argument('--ref', help='the reference of noise')
    args = psr.parse_args()
    noise = np.genfromtxt(args.ref, dtype=[('thre', '<i2'), ('noi', '<f4')])
    pulse = np.genfromtxt(args.ipt, dtype=[('prompt', '<f4'), ('delay1', '<f4'), ('delay10', '<f4'), ('triggerNum', '<i4')])
    timeInterval = np.array([200, 700, 9000])
    noiserate = noise['noi'].reshape((2,1))*timeInterval*1e-6
    pulse['prompt'] -= noiserate[:,0]
    pulse['delay1'] -= noiserate[:,1]
    pulse['delay10'] -= noiserate[:,2]
    np.savetxt(args.opt, pulse)