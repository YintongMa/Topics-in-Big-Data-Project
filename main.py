from dataLoader import DataLoader
from utils import DataGenerator
from bloom_filter import BloomFilter


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


# Overlap Similarity = |AnB| / min(|A|,|B|)

def brute_force(candidate_index, cols, threshold):
    res = [False for _ in range(len(cols))]
    candidate = cols[candidate_index]
    for i, col in enumerate(cols):
        if i != candidate_index:
            if len(intersection(candidate, col)) / min(len(candidate), len(col)) >= threshold:
                res[i] = True
        else:
            res[i] = True
    return res


def bloom_filter(candidate_index, cols, threshold, n, p):
    res = [False for _ in range(len(cols))]
    candidate = cols[candidate_index]

    candidate_bloom_filter = BloomFilter(n, p)
    for num in candidate:
        candidate_bloom_filter.add(chr(num))
    estimated_candidate_col_size = candidate_bloom_filter.estimate_num_of_elem()

    for i, col in enumerate(cols):
        if i != candidate_index:
            bloom_filter = BloomFilter(n, p)
            for num in col:
                bloom_filter.add(chr(num))
            estimated_col_size = bloom_filter.estimate_num_of_elem()
            estimated_size_of_intersection = candidate_bloom_filter.estimate_size_of_intersection(bloom_filter)

            if estimated_size_of_intersection / min(estimated_candidate_col_size, estimated_col_size) >= threshold:
                res[i] = True
        else:
            res[i] = True
    return res


def lsh(candidate_index, cols, threshold):
    pass


def false_positive(res, baseline):
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test brute_force
    # lst1 = [4, 9, 9,1, 17, 11, 26, 28, 54, 69]
    # lst2 = [9, 9, 74, 21, 45, 11, 63, 28, 26]
    # print(brute_force(0,[lst1, lst2], 0.6))

    loader = DataLoader('columns.txt')
    cols = loader.load_data()

    # show cols
    print('num of columns', len(cols))

    import numpy as np

    # brute_force
    brute_force_result = brute_force(0, cols, 0.6)
    print("brute_force_result:")
    print(brute_force_result)

    # bloom filter
    block_cnt = 20
    block_len = 30
    n = block_cnt * block_len  # code space. set it to the max size of a col for now
    p = 0.01  # false positive probability
    bloom_filter_result = bloom_filter(0, cols, threshold=0.6, n=n, p=p)
    print("bloom_filter_result:")
    print(bloom_filter_result)

    brute_force_np = np.array(brute_force_result, dtype=bool)
    bloom_filter_np = np.array(bloom_filter_result, dtype=bool)

    correct = (brute_force_np == bloom_filter_np)
    accuracy = correct.sum() / correct.size
    print("accuracy of bloom filter = " + str(accuracy))
