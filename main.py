from dataLoader import DataLoader
from bloom_filter import BloomFilter
from LSH import LSH
import time
from bitarray import bitarray
import numpy as np
import random
import matplotlib.pyplot as plt


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


def bloom_filter(candidate_index, cols, threshold, bloom_filter_list):
    res = [False for _ in range(len(cols))]

    start = time.time()
    for i in range(len(cols)):
        if i != candidate_index:
            candidate_bloom_filter = bloom_filter_list[candidate_index]
            # estimated_candidate_col_size = candidate_bloom_filter.estimate_num_of_elem()
            estimated_candidate_col_size = len(cols[candidate_index])

            bloom_filter = bloom_filter_list[i]
            # estimated_col_size = bloom_filter.estimate_num_of_elem()
            estimated_col_size = len(cols[i])
            estimated_size_of_intersection = candidate_bloom_filter.estimate_size_of_intersection(bloom_filter)

            if estimated_size_of_intersection / min(estimated_candidate_col_size, estimated_col_size) >= threshold:
                res[i] = True
        else:
            res[i] = True
    processing_time = time.time() - start
    return res, processing_time


def lsh_method(candidate_index, lsh, threshold):
    query_mh = lsh.build_mh_sig_from_hashvalues(lsh.mh_sigs[candidate_index])
    index = lsh.build_lsh_index(threshold)
    res = [False for _ in range(lsh.size)]

    start = time.time()
    neighbors = index.query(query_mh)
    processing_time = time.time() - start

    for i in neighbors:
        res[int(i)] = True
    return res, processing_time


def lsh_ensemble(candidate_index, lsh, threshold):
    query_mh = lsh.build_mh_sig_from_hashvalues(lsh.mh_sigs[candidate_index])
    res = [False for _ in range(lsh.size)]
    index = lsh.build_lsh_ensemble_index(threshold)

    start = time.time()
    for key in index.query(query_mh, len(lsh.cols[candidate_index])):
        res[int(key)] = True
    processing_time = time.time() - start
    return res, processing_time


def lsh_bloom_filter(candidate_index, lsh, lsh_threshold, overlap_threshold, bloom_filter_list):
    query_mh = lsh.build_mh_sig_from_hashvalues(lsh.mh_sigs[candidate_index])
    index = lsh.build_lsh_index(lsh_threshold)
    res = [False for _ in range(lsh.size)]

    start = time.time()
    neighbors = index.query(query_mh)

    for i in neighbors:
        i = int(i)
        if i != candidate_index:
            candidate_bloom_filter = bloom_filter_list[candidate_index]
            estimated_candidate_col_size = len(lsh.cols[candidate_index])

            bloom_filter = bloom_filter_list[i]
            estimated_col_size = len(lsh.cols[i])

            estimated_size_of_intersection = candidate_bloom_filter.estimate_size_of_intersection(bloom_filter)

            if estimated_size_of_intersection / min(estimated_candidate_col_size,
                                                    estimated_col_size) >= overlap_threshold:
                res[i] = True
        else:
            res[i] = True
    processing_time = time.time() - start
    return res, processing_time


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
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    return round(precision, 3), round(recall, 3), round(f1, 3)


def load_bloom_filters(dir_path, count, n, p):
    bloom_filters = []
    for i in range(count):
        file_path = dir_path + "bloom_filter_" + str(i)
        f = open(file_path, 'rb')

        bloom_filter = BloomFilter(n, p)
        bloom_filter.bit_array = bitarray(endian='little')
        bloom_filter.bit_array.frombytes(f.read())
        # print(bloom_filter.bit_array.length())
        bloom_filters.append(bloom_filter)

    # f = open(path, 'rb')
    # # lines = f.readlines()
    # bloom_filters = []
    # output = f.read()
    # bytes_list = output.split(b'\n\n')
    # print("len(bytes_list)", len(bytes_list))
    # for bit_array_in_bytes in bytes_list:
    #     bloom_filter = BloomFilter(n, p)
    #     bloom_filter.bit_array = bitarray(endian='little')
    #     bloom_filter.bit_array.frombytes(bit_array_in_bytes)
    #     print(bloom_filter.bit_array.length())
    #     bloom_filters.append(bloom_filter)
    # for line in lines:
    #     bloom_filter = BloomFilter(n, p)
    #     line_list = line.split(b'\n')
    #     bit_array_in_bytes = line_list[0]
    #     bloom_filter.bitarray = bitarray(endian='little')
    #     bloom_filter.bit_array.frombytes(bit_array_in_bytes)
    #     bloom_filters.append(bloom_filter)
    return bloom_filters


def candidate_generation(cols, similarity):
    # currently we could choose a median column from all columns order by the count of similar neighbours above a
    # similarity
    length = len(cols)
    hmap = {}
    for i in range(length):
        for j in range(i + 1, length):
            print(i, j)
            if len(intersection(cols[i], cols[j])) / min(len(cols[i]), len(cols[j])) >= similarity:
                hmap[i] = hmap.get(i, 0) + 1
                hmap[j] = hmap.get(j, 0) + 1
    res = [(k, v) for k, v in hmap.items()]
    res.sort(key=lambda x: x[1])
    return res[len(res) // 2][0]


def multi_variants_benchmark(col_file_name, similarity_list):
    loader = DataLoader(col_file_name)
    cols = loader.load_data()
    for similarity in similarity_list:
        index = candidate_generation(cols, similarity)
        baseline = brute_force(index, cols, similarity)
        multi_variants_benchmark_bloom_filter(cols, index, similarity, baseline)
        multi_variants_benchmark_lsh()
        multi_variants_benchmark_lsh_ensemble()
        multi_variants_benchmark_lsh_bloom_filter()


def multi_variants_benchmark_bloom_filter(cols, candidate_index, similarity, p_list, block_list, baseline):
    print("benchmarking bloom filter:")
    for p in p_list:
        for block_cnt, block_len in block_list:
            print("p: &f, block_cnt: %d, block_len: %d", p, block_cnt, block_len)
            n = block_cnt * block_len  # code space. set it to the max size of a col for now
            bf_list = []
            for col in cols:
                bf = BloomFilter(n, p)
                for num in col:
                    bf.add(chr(num))
                bf_list.append(bf)
            # bloom filter
            print("bloom filter")
            bf_result, t = bloom_filter(candidate_index, cols, similarity, bf_list)
            precision, recall, f1 = get_statistics(bloom_filter_result, baseline)
            print("bloom_filter finished, used %s s" % str(round(t, 4)))
            print(precision, recall, f1, '\n')


def multi_variants_benchmark_lsh():
    pass


def multi_variants_benchmark_lsh_ensemble():
    pass


def multi_variants_benchmark_lsh_bloom_filter():
    pass


# Press the green button in the gutter to run.sh the script.
if __name__ == '__main__':
    # test brute_force
    # lst1 = [4, 9, 9,1, 17, 11, 26, 28, 54, 69]
    # lst2 = [9, 9, 74, 21, 45, 11, 63, 28, 26]
    # print(brute_force(0,[lst1, lst2], 0.6))

    loader = DataLoader('columns.txt')
    cols = loader.load_data()
    # show cols
    print('num of columns', len(cols))

    threshold = 0.6

    # bloom filter
    block_cnt = 20
    block_len = 30
    n = block_cnt * block_len  # code space. set it to the max size of a col for now
    p = 0.01  # false positive probability
    # load bloom filters from file
    bloom_filter_dir_path = "./bloom_filter/"
    bloom_filter_list = load_bloom_filters(bloom_filter_dir_path, len(cols), n, p)
    print("load bloom filters finished\n")

    lsh = LSH(cols, 128, 'mh_sig.txt')
    lsh.load_sigs()
    print("build lsh signature finished\n")

    num_runs = 20

    labels = ["bloom filter", "lsh", "lsh ensemble", "lsh + bloom filter"]
    precision_array = np.empty((num_runs, len(labels)), dtype=float)
    recall_array = np.empty((num_runs, len(labels)), dtype=float)
    f1_array = np.empty((num_runs, len(labels)), dtype=float)

    for i in range(num_runs):
        print("-------------------------Run " + str(i) + "------------------------------")

        candidate_index = random.randrange(0, len(cols))

        # brute_force
        brute_force_result = brute_force(candidate_index, cols, threshold)
        print("brute_force finished\n")

        # bloom filter
        print("bloom filter")
        bloom_filter_result, t = bloom_filter(candidate_index, cols, threshold, bloom_filter_list)
        precision, recall, f1 = get_statistics(bloom_filter_result, brute_force_result)
        print("bloom_filter finished, used %s s" % str(round(t, 4)))
        print(precision, recall, f1, '\n')
        precision_array[i][0] = precision
        recall_array[i][0] = recall
        f1_array[i][0] = f1

        print("lsh")
        res, t = lsh_method(candidate_index, lsh, threshold)
        precision, recall, f1 = get_statistics(res, brute_force_result)
        print("lsh finished, used %s s" % str(round(t, 4)))
        print(precision, recall, f1, '\n')
        precision_array[i][1] = precision
        recall_array[i][1] = recall
        f1_array[i][1] = f1

        print("lsh ensemble")
        res, t = lsh_ensemble(candidate_index, lsh, threshold)
        print("lsh ensemble finished, used %s s" % str(round(t, 4)))
        precision, recall, f1 = get_statistics(res, brute_force_result)
        print(precision, recall, f1, '\n')
        precision_array[i][2] = precision
        recall_array[i][2] = recall
        f1_array[i][2] = f1

        print("lsh + bloom filter")
        res, t = lsh_bloom_filter(candidate_index, lsh, 0.1, threshold, bloom_filter_list)
        print("lsh + bloom filter finished, used %s s" % str(round(t, 4)))
        precision, recall, f1 = get_statistics(res, brute_force_result)
        print(precision, recall, f1)
        precision_array[i][3] = precision
        recall_array[i][3] = recall
        f1_array[i][3] = f1

    avg_precision = np.mean(precision_array, axis=0)
    avg_recall = np.mean(recall_array, axis=0)
    avg_f1 = np.mean(f1_array, axis=0)

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    precision_bar = ax.bar(x - width, avg_precision, width, label='precision')
    recall_bar = ax.bar(x, avg_recall, width, label='recall')
    f1_bar = ax.bar(x + width, avg_f1, width, label='f1')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title('Average scores over ' + str(num_runs) + ' runs')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()
