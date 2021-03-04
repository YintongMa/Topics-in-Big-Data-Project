import random
from scipy.special import comb


class DataGenerator:
    def __init__(self, block_cnt, block_len, maxv, minv):
        self.blocks = self.generate_blocks(block_cnt, block_len, maxv, minv)

    def generate_blocks(self, block_cnt, block_len, maxv, minv):
        # todo introduce data distribution
        res = []
        for i in range(block_cnt):
            # sampling without replacement
            res.append(random.sample(range(minv, maxv), block_len))
        return res

    def generate_columns(self, col_cnt, block_cnt):
        """
        :param col_cnt: number of generated columns
        :param block_cnt: number of blocks to use
        :return:
        """
        res = []
        for i in range(col_cnt):
            candidate_blocks = random.sample(self.blocks, block_cnt)
            flat_block = []
            for block in candidate_blocks:
                flat_block.extend(block)
            res.append(flat_block)
        return res


if __name__ == '__main__':
    block_cnt = 20
    block_len = 30
    dgen = DataGenerator(block_cnt, block_len, 9999, 0)
    random.seed(1)
    f = open('columns.txt', 'w')
    for i in range(1, block_cnt + 1):
        block_to_use = i
        column_cnt = int(comb(block_cnt, block_to_use))
        for col in dgen.generate_columns(20, block_to_use):
            f.write(str(col) + '\n')
