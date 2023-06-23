import argparse, os, re, pandas as pd, numpy as np
from datetime import datetime
# def getSetting(summary_files):
def getXZE(comment):
    x, z, E = 0, 0, 0
    result = re.match(r'.+x=?([-|+]?\d+)m?,? ?z=?([-|+]?\d+)m?,? ?(\d+) ?mev', comment, re.I)
    if result:
        x, z, E = result.group(1), result.group(2), result.group(3)
    else:
        result = re.match(r'.+(\d+) ?mev,? ?x=?([-|+]?\d+)m?,? ?z=?([-|+]?\d+)m?', comment, re.I)
        if result:
            x, z, E = result.group(2), result.group(3), result.group(1)
        else:
            result = re.match(r'.+\(([-|+]?\d+)m?,? ?([-|+]?\d+)m?\),? ?(\d+) ?mev', comment, re.I)
            if result:
                x, z, E = result.group(1), result.group(2), result.group(3)
    return x, z, E

if __name__=="__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('ipt', help='run no')
    psr.add_argument('--badrun', dest='badrun', default='BadRun.txt', help='bad run file')
    args = psr.parse_args()
    badruns = np.loadtxt(args.badrun, dtype=int)
    if int(args.ipt) in badruns:
        exit(0)
    response = os.popen('summary {}'.format(args.ipt))
    summary = {}
    for res in response:
        kv = [i.strip() for i in res.split('=', 1)]
        if len(kv)==2:
            summary[kv[0]] = kv[1]
    # print(summary)
    mode = 0
    if summary['Run Mode']=='10:LINAC' and (not re.search('(retune|error)', summary['End Comment'], re.I)):
        deltatime = (datetime.strptime(summary['End time'], "%a %b %d %H:%M:%S %Y") - datetime.strptime(summary['Start time'], "%a %b %d %H:%M:%S %Y")).total_seconds()/60
    # if summary['Run Mode']=='10:LINAC' and not (re.search('MW', summary['Comment'], 0) or re.search('micro', summary['Comment'], 1) or re.search('[retune|error]', summary['End Comment'], 1)):
        if deltatime > 5:
            if re.search('(mw|micro)', summary['Comment'], re.I):
                mode = 1
            else:
                if deltatime<30:
                    exit(0)
            x, z, E = getXZE(summary['Comment'])
            print('{},{},{},{},{},{},{}'.format(summary['Run number'], x, z, E, mode, int(deltatime), summary['End Comment'].replace(',', ' ')))
