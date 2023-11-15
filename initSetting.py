#!/usr/bin/env python3
'''
按照 (X, Z, E) 三元组对 run 分类，用于生成对比图的 PDF 文件列表等。
'''
import argparse
import pandas as pd

summary_header = ['RunNo', 'X', 'Z', 'E', 'Mode', 'Time', 'Comment', 'NormalRun']

psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', nargs=2, help='input summary files')
psr.add_argument('-o', dest='opt', help='ouput file')
args = psr.parse_args()

summary = pd.concat([ pd.read_csv(fn, names=summary_header).assign(Phase=fn.split("_")[0]) 
                      for fn in args.ipt ], axis=0)

summary.query("Mode==0", inplace=True)

# 因为 SK6 的 LINAC 刻度只有 X=-12m，此处只做 Z 和 E 的区分。
ZE = set()

with open(args.opt, "w") as opt:
    for k, v in summary.groupby(['Z', 'E', 'Phase']):
        print(f"Z{k[0]}_E{k[1]}_{k[2]}:={' '.join(v['RunNo'].values.astype(str))}", file=opt)
        print(f"Z{k[0]}_E{k[1]}_{k[2]}_x:={' '.join(v['X'].values.astype(str))}", file=opt)
        ZE.add(f"Z{k[0]}_E{k[1]}")
    print(f"ZE:={' '.join(ZE)}", file=opt)
