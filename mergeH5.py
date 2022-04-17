import h5py, argparse, numpy as np
psr = argparse.ArgumentParser()
psr.add_argument('-i', dest='ipt', nargs='+', help='input h5 file')
psr.add_argument('-o', dest='opt', help='output h5 file')
args = psr.parse_args()
length = 0
dtype = []
length = []
index = []
info = []
with h5py.File(args.ipt[0], 'r') as ipt:
    keys = list(ipt.keys())
    attrs = list(ipt.attrs)
    attrsValues = [ipt.attrs[i] for i in attrs]
    for i in range(len(keys)):
        dtype.append(ipt[keys[i]].dtype)
        length.append(0)
        index.append(0)
for h5f in args.ipt:
    with h5py.File(h5f, 'r') as ipt:
        for i in range(len(keys)):
            length[i] += len(ipt[keys[i]])
for i in range(len(keys)):
    info.append(np.zeros((length[i],),dtype=dtype[i]))

for h5f in args.ipt:
    with h5py.File(h5f, 'r') as ipt:
        for i in range(len(keys)):
            info[i][index[i]:(index[i]+len(ipt[keys[i]]))] = ipt[keys[i]][:]
            index[i] += len(ipt[keys[i]])
print('length: {}'.format(length))
with h5py.File(args.opt, 'w') as opt:
    for i in range(len(attrs)):
        opt.attrs[attrs[i]] = attrsValues[i]
    for i in range(len(keys)):
        opt.create_dataset(keys[i], data=info[i], compression='gzip')
