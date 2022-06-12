#!/usr/bin/python3
import uproot, argparse
import numpy as np
import subprocess
# 储存每个文件中的entries数目
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-f', dest='fname', help='input root file name')
    psr.add_argument('-o', dest='opt', help='output result file')
    args = psr.parse_args()
    process = subprocess.Popen("ls {} -v".format(args.fname), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    path = process.stdout.read().decode('utf-8').split('\n')
    path = [args.fname+'/'+i for i in path[:-1]]
    nums = np.zeros(len(path), dtype=int)
    for i,p in enumerate(path):
        with uproot.open(p) as ipt:
            eventid = ipt["Readout/TriggerNo"].array(library='np')
            nums[i] = eventid.shape[0]
    np.savetxt(args.opt, nums, fmt='%d')