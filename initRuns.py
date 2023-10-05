#!/usr/bin/env python3
import argparse
import os
import re
from datetime import datetime
from functools import cache
import sys

# def getSetting(summary_files):
def getXZE(comment: str):
    x, z, E = 0, 0, 0
    result = re.match(r'.+x=?([-+]?\d+)m?,? *z=?([-+]?\d+)m?,? ?(\d+) ?mev', comment, re.I)
    if result:
        x, z, E = result.group(1), result.group(2), result.group(3)
    else:
        result = re.match(r'.+(\d+) ?mev,? ?x=?([-+]?\d+)m?,? ?z=?([-+]?\d+)m?', comment, re.I)
        if result:
            x, z, E = result.group(2), result.group(3), result.group(1)
        else:
            result = re.match(r'.+\(([-+]?\d+)m?,? ?([-+]?\d+)m?\),? ?(\d+) ?mev', comment, re.I)
            if result:
                x, z, E = result.group(1), result.group(2), result.group(3)
    return int(x), int(z), int(E)

@cache
def getMeta(run: int) -> dict:
    response = os.popen(f'summary {run}')
    summary = {}
    for res in response:
        kv = [i.strip() for i in res.split('=', 1)]
        if len(kv)==2:
            summary[kv[0]] = kv[1]
    return summary

def getNormalRun(run: int) -> int:
    summary = getMeta(run-1)
    while summary['Run Mode']!='1:Normal':
        run -= 1
        summary = getMeta(run-1)
    return run-1

if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('ipt', nargs="+", type=int, help='run numbers')
    args = psr.parse_args()

    # 91375, 91378, 91391, 91477, 91539 的 run summary 没有写全，使用前一个 run 的信息
    formers = (0, 0, 0, 0) # run, X, Z, E
    for run in args.ipt:
        summary = getMeta(run)

        mode = 0
        if summary['Run Mode']=='10:LINAC' and (not re.search('(retune|error)', summary['End Comment'], re.I)):
            deltatime = (datetime.strptime(summary['End time'], "%a %b %d %H:%M:%S %Y") - datetime.strptime(summary['Start time'], "%a %b %d %H:%M:%S %Y")).total_seconds()/60
        # if summary['Run Mode']=='10:LINAC' and not (re.search('MW', summary['Comment'], 0) or re.search('micro', summary['Comment'], 1) or re.search('[retune|error]', summary['End Comment'], 1)):
            if deltatime > 5:
                if re.search('(mw|micro)', summary['Comment'], re.I):
                    mode = 1
                else:
                    if deltatime<30:
                        continue
                X, Z, E = getXZE(summary['Comment'])
                if run == formers[0] + 1 and formers[1:] != (X, Z, E):
                    print(f'{run} use former run {formers[0]} info', file=sys.stderr)
                    X, Z, E = formers[1:]
                end_comment = summary['End Comment'].replace(',', ' ')
                prev_normal_run = getNormalRun(int(summary['Run number']))
                print(f"{summary['Run number']},{X},{Z},{E},{mode},{int(deltatime)},{end_comment},{prev_normal_run}")
                formers = (run, X, Z, E)
                
