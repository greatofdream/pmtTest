import h5py, argparse, numpy as np
class h5Merger():
    def __init__(self, files) -> None:
        self.files = files
        dtype = []
        length = []
        with h5py.File(files[0], 'r') as ipt:
            keys = list(ipt.keys())
            attrs = list(ipt.attrs)
            attrsValues = [ipt.attrs[i] for i in attrs]
            for i in range(len(keys)):
                dtype.append(ipt[keys[i]].dtype)
                length.append(0)
        for h5f in files:
            with h5py.File(h5f, 'r') as ipt:
                for i in range(len(keys)):
                    length[i] += len(ipt[keys[i]])
        self.attrsValues = attrsValues
        self.length = length
        self.dtype = dtype
        self.keys = keys
        self.attrs = attrs
    def read(self):
        info = []
        for i in range(len(self.keys)):
            info.append(np.zeros((self.length[i],), dtype=self.dtype[i]))
        index = np.zeros(len(self.length), dtype=int)
        for h5f in self.files:
            with h5py.File(h5f, 'r') as ipt:
                for i in range(len(self.keys)):
                    info[i][index[i]:(index[i] + len(ipt[self.keys[i]]))] = ipt[self.keys[i]][:]
                    index[i] += len(ipt[self.keys[i]])
        print('length: {}'.format(self.length))
        self.info = info
        return info
if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument('-i', dest='ipt', nargs='+', help='input h5 file')
    psr.add_argument('-o', dest='opt', help='output h5 file')
    args = psr.parse_args()
    reader = h5Merger(args.ipt)
    info = reader.read()
    attrs, attrsValues, keys = reader.attrs, reader.attrsValues, reader.keys
    with h5py.File(args.opt, 'w') as opt:
        for i in range(len(attrs)):
            opt.attrs[attrs[i]] = attrsValues[i]
        for i in range(len(keys)):
            opt.create_dataset(keys[i], data=info[i], compression='gzip')
