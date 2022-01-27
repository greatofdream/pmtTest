import json, argparse
import subprocess
if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', help='input h5 file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    args = psr.parse_args()
    with open(args.ipt, 'r') as ipt:
        paras = json.load(ipt)
    if paras['task'] == 'QE':
        files = ' '.join(['../trigger/Ex1/{}/400ns/qe/ratio.h5'.format(i) for i in paras['runno']])
        result = subprocess.run('python3 ../QERelative.py -i {} -o test.h5 -c {} --ref {} > {}'.format(files, paras['testch'], paras['refch'], args.opt), shell=True, capture_output=True)
        print(result)