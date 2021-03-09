from dataLoader import DataLoader
from utils import DataGenerator
from bloom_filter import BloomFilter
from LSH import LSH
import time

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


def lsh_method(candidate_index, lsh):
    query_mh = lsh.build_mh_sig_from_hashvalues(lsh.mh_sigs[candidate_index])
    index = lsh.build_lsh_index()
    neighbors = index.query(query_mh)
    res = [False for _ in range(lsh.size)]
    for i in neighbors:
        res[int(i)] = True
    return res

def lsh_ensemble(candidate_index, lsh):
    query_mh = lsh.build_mh_sig_from_hashvalues(lsh.mh_sigs[candidate_index])
    res = [False for _ in range(lsh.size)]
    index = lsh.build_lsh_ensemble_index()
    for key in index.query(query_mh, len(lsh.cols[candidate_index])):
        res[int(key)] = True
    return res

def get_statistics(res, ground_truth):
    TP, TN, FP, FN = 0, 0, 0, 0
    for i, x in enumerate(res):
        if x == True and ground_truth[i] == True:
            TP += 1
        elif x == False and ground_truth[i] == False:
            TN += 1
        elif x == True and ground_truth[i] == False:
            FP += 1
        else:
            FN += 1
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*precision*recall/(precision + recall)
    return round(precision, 3), round(recall, 3), round(f1, 3)


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

    # import numpy as np

    threshold = 0.6
    candidate_index = 166
    # brute_force
    brute_force_result = brute_force(candidate_index, cols, threshold)
    print("brute_force finished")

    # bloom filter
    block_cnt = 20
    block_len = 30
    n = block_cnt * block_len  # code space. set it to the max size of a col for now
    p = 0.01  # false positive probability
    bloom_filter_result = bloom_filter(candidate_index, cols, threshold=threshold, n=n, p=p)
    print("bloom_filter finished")

    precision, recall, f1 = get_statistics(bloom_filter_result, brute_force_result)
    print(precision, recall, f1)

    loader = DataLoader('columns.txt')
    cols = loader.load_data()
    lsh = LSH(cols, 0.6, 128, 'mh_sig.txt')
    lsh.load_sigs()
    print("build lsh signature finished")

    print("lsh")
    res = lsh_method(candidate_index, lsh)
    precision, recall, f1 = get_statistics(res, brute_force_result)
    print(precision, recall, f1)

    print("lsh ensemble")
    res = lsh_ensemble(candidate_index, lsh)
    precision, recall, f1 = get_statistics(res, brute_force_result)
    print(precision, recall, f1)