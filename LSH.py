from datasketch import MinHash, MinHashLSH

class LSH:
    def __init__(self, cols, threshold, perm):
        self.cols = cols
        self.threshold = threshold
        self.perm = perm

    def build_index(self):
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.perm)
        for i, col in enumerate(self.cols):
            key = str(i)
            mh = self.build_mh_sig(col)
            lsh.insert(key, mh)
        return lsh

    def build_mh_sig(self, set):
        m = MinHash(num_perm=self.perm)
        for x in set:
            m.update(str(x).encode('utf8'))
        return m

