from datasketch import MinHashLSHEnsemble, MinHash, MinHashLSH

from dataLoader import DataLoader
import numpy as np



class LSH:
    def __init__(self, cols, perm, path):
        self.cols = cols
        self.size = len(cols)
        self.perm = perm
        self.path = path
        self.mh_sigs = None

    def build_lsh_index(self, threshold):
        lsh = MinHashLSH(threshold=threshold, num_perm=self.perm)
        for i, sig in enumerate(self.mh_sigs):
            key = str(i)
            mh_obj = MinHash(num_perm=self.perm)
            mh_array = np.asarray(sig, dtype=int)
            mh_obj.hashvalues = mh_array
            lsh.insert(key, mh_obj)
        return lsh

    def build_lsh_ensemble_index(self, threshold):
        lsh_ensemble = MinHashLSHEnsemble(threshold=threshold, num_perm=self.perm, num_part=32)
        objects = []
        for i, sig in enumerate(self.mh_sigs):
            key = str(i)
            mh_obj = MinHash(num_perm=self.perm)
            mh_array = np.asarray(sig, dtype=int)
            mh_obj.hashvalues = mh_array
            objects.append((key, mh_obj, len(self.cols[i])))
        lsh_ensemble.index(objects)
        return lsh_ensemble

    def load_sigs(self):
        loader = DataLoader(self.path)
        self.mh_sigs = loader.load_data()

    def build_all_mh_sig(self):
        '''
        build minhash signatures and persist to a file
        :return:
        '''
        f = open(self.path, 'w')
        for set in self.cols:
            mh = self.build_mh_sig(set)
            f.write(str(mh.hashvalues.tolist()) + '\n')

    def build_mh_sig(self, set):
        m = MinHash(num_perm=self.perm)
        for x in set:
            m.update(str(x).encode('utf8'))
        return m

    def build_mh_sig_from_hashvalues(self,sig):
        mh_obj = MinHash(num_perm=self.perm)
        mh_array = np.asarray(sig, dtype=int)
        mh_obj.hashvalues = mh_array
        return mh_obj

if __name__ == '__main__':
    loader = DataLoader('columns.txt')
    cols = loader.load_data()
    perm = 128
    lsh = LSH(cols, perm, 'mh_sig.txt')
    lsh.build_all_mh_sig()