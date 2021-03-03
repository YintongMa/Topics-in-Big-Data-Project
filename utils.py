import random
class DataGenerator:

    def generate_blocks(self, block_cnt, max_block_len, min_block_len, maxv, minv):
        #todo introduce data distribution
        res = []
        for i in range(block_cnt):
            block = []
            length = random.randint(min_block_len, max_block_len)
            for j in range(length):
                block.append(random.randint(minv, maxv))
            res.append(block)
        return res

    def generate_columns(self, col_cnt, max_col_len, col_block_types, block_types, maxv = 9999, minv = 0):
        """
        :param col_cnt: number of generated columns
        :param max_col_len: max length of each column
        :param col_block_types: distinct building blocks for each column
        :param block_types: number of distinct candidate building blocks
        :param maxv: max value of column elements
        :param minv: min value of column elements
        :return:
        """
        res = []
        blocks = self.generate_blocks(block_types, max_col_len//col_block_types, 1, maxv, minv)
        for i in range(col_cnt):
            candidate_blocks = random.sample(blocks, col_block_types)
            flat_block = []
            for block in candidate_blocks:
                for item in block:
                    flat_block.append(item)
            res.append(flat_block*(max_col_len//len(flat_block)))
        return res


if __name__ == '__main__':
    dgen = DataGenerator()
    for i in dgen.generate_columns(10,100,3,5):
        print(i)








