import random
import time

import numpy as np
from scipy.special import comb

from LSH import LSH
from bloom_filter import BloomFilter
from dataGenerator import DataGenerator
from dataLoader import DataLoader
from main import lsh_method, get_statistics, lsh_ensemble, lsh_bloom_filter, bloom_filter, load_bloom_filters, \
    brute_force
import matplotlib.pyplot as plt


def bf_benchmark(bloom_filter_list, cols, candidate_index, threshold, brute_force_result):
    # bloom filter
    print("bloom filter")
    bloom_filter_result, t = bloom_filter(candidate_index, cols, threshold, bloom_filter_list)
    precision, recall, f1 = get_statistics(bloom_filter_result, brute_force_result)
    print("bloom_filter finished, used %s s" % str(round(t, 4)))
    print(precision, recall, f1, '\n')
    return precision, recall, f1, t


def lsh_benchmark(lsh, candidate_index, threshold, brute_force_result):
    print("lsh")
    res, t = lsh_method(candidate_index, lsh, threshold)
    precision, recall, f1 = get_statistics(res, brute_force_result)
    print("lsh finished, used %s s" % str(round(t, 4)))
    print(precision, recall, f1, '\n')
    return precision, recall, f1, t


def lsh_ensemble_benchmark(lsh, candidate_index, threshold, brute_force_result):
    print("lsh ensemble")
    res, t = lsh_ensemble(candidate_index, lsh, threshold)
    print("lsh ensemble finished, used %s s" % str(round(t, 4)))
    precision, recall, f1 = get_statistics(res, brute_force_result)
    print(precision, recall, f1, '\n')
    return precision, recall, f1, t


def lsh_bf_benchmark(lsh, bloom_filter_list, candidate_index, threshold, brute_force_result):
    print("lsh + bloom filter")
    res, t = lsh_bloom_filter(candidate_index, lsh, 0.1, threshold, bloom_filter_list)
    print("lsh + bloom filter finished, used %s s" % str(round(t, 4)))
    precision, recall, f1 = get_statistics(res, brute_force_result)
    return precision, recall, f1, t


def generate_bf_list(cols):
    block_cnt = 20
    block_len = 30
    n = block_cnt * block_len  # code space. set it to the max size of a col for now
    p = 0.01  # false positive probability

    # build bloom filter for all cols
    bloom_filter_list = []
    for col in cols:
        bloom_filter = BloomFilter(n, p)
        for num in col:
            bloom_filter.add(chr(num))
        bloom_filter_list.append(bloom_filter)
    return bloom_filter_list


def init(dataset):
    dataset.sort(key=lambda x: len(x))
    bf_lists = []
    lsh_list = []
    for cols in dataset:
        cols.sort(key=lambda x: len(x))
        bf_lists.append(generate_bf_list(cols))
        lsh = LSH(cols, 128, '')
        lsh.build_all_mh_sig()
        lsh_list.append(lsh)
    print("init bloom filters & lsh list finished\n")
    return bf_lists, lsh_list


def benchmark(cols, candidate_index, threshold, bf_list, lsh, brute_force_result, title):
    print("title", title)
    print('num of columns', len(cols))
    print("candidate_index", candidate_index)
    print("threshold", threshold)

    labels = ["bloom filter", "lsh", "lsh ensemble", "lsh + bloom filter"]
    precision = [0 for _ in range(len(labels))]
    recall = [0 for _ in range(len(labels))]
    f1 = [0 for _ in range(len(labels))]
    time = np.empty(len(labels), dtype=float)

    precision[0], recall[0], f1[0], time[0] = bf_benchmark(bf_list, cols, candidate_index, threshold,
                                                           brute_force_result)
    precision[1], recall[1], f1[1], time[1] = lsh_benchmark(lsh, candidate_index, 0.1, brute_force_result)
    precision[2], recall[2], f1[2], time[2] = lsh_ensemble_benchmark(lsh, candidate_index, threshold,
                                                                     brute_force_result)
    precision[3], recall[3], f1[3], time[3] = lsh_bf_benchmark(lsh, bf_list, candidate_index, threshold,
                                                               brute_force_result)

    x = np.arange(len(labels))  # the label locations
    width = 0.25  # the width of the bars
    fig, ax = plt.subplots()
    precision_bar = ax.bar(x - width, precision, 0.5 * width, label='precision')
    recall_bar = ax.bar(x, recall, 0.5 * width, label='recall')
    f1_bar = ax.bar(x + width, f1, 0.5 * width, label='f1')
    # time_bar = ax.bar(x + 2*width, time, 0.5*width, label='time_cost')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    # plt.show()
    fig.savefig("./bench_results/" + title)

    f = open('./bench_results/'+title+".txt", 'w')
    f.write(",".join(labels)+"\n")
    f.write(",".join([str(i) for i in precision])+"\n")
    f.write(",".join([str(i) for i in recall])+"\n")
    f.write(",".join([str(i) for i in f1])+"\n")
    f.write(",".join([str(i) for i in time])+"\n")

    return time


def generate_dataset(dataset_path, cols_size):
    for size in cols_size:
        block_cnt = 20
        block_len = 30
        dgen = DataGenerator(block_cnt, block_len, 9999, 0)
        random.seed(1)
        f = open(dataset_path + "/" + str(size) + ".txt", 'w')
        for i in range(1, block_cnt + 1):
            block_to_use = i
            for col in dgen.generate_columns(size // 20, block_to_use):
                f.write(str(col) + '\n')


def integrated_benchmark(dataset_path):
    """
    Variables:
    Dataset size: number of columns
    Dataset distribution: column length distribution
    threshold
    query column
    """
    loader = DataLoader("")
    dataset = loader.load_dataset(dataset_path)

    bf_lists, lsh_list = init(dataset)
    print("""
Benchmark 1 
Goal: Measure scalability of different methods
Variable: 
    the size of datasets. size: 400, 600, 800, 1000
Fix:
    threshold = 0.6
    query column = median col
Output:
    Runtime
    precision, recall, f1
""")
    labels = ["bloom filter", "lsh", "lsh ensemble", "lsh + bloom filter"]
    time_for_each_size = np.empty((len(dataset), len(labels)), dtype=float)
    x_axis = np.empty(len(dataset), dtype=int)

    for i, cols in enumerate(dataset):
        candidate_index = len(cols) // 2  # median col
        brute_force_result = brute_force(candidate_index, cols, 0.6)
        print("brute_force finished\n")
        time = benchmark(cols, candidate_index, 0.6, bf_lists[i], lsh_list[i], brute_force_result,
                         "Benchmark-1-cols-size-" + str(len(cols)))
        time_for_each_size[i] = time
        x_axis[i] = len(cols)

    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.plot(x_axis, time_for_each_size[:, i], 'o-', label=labels[i])
    ax.legend()
    ax.set_title("Benchmark-1-cols-size")
    ax.set_xticks(x_axis)
    ax.set_xlabel("size")
    ax.set_ylabel("time(s)")
    fig.tight_layout()
    # plt.show()
    fig.savefig("./bench_results/Benchmark-1-cols-size")

    print("""
Benchmark 2
Goal: Measure the effect of threshold
Variable:
   threshold: 0.1 0.3 0.5 0.7 0.9
Fix:
    dataset size = median col
Output
    Runtime
    precision, recall, f1
""")
    threshold_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    time_for_each_threshold = np.empty((len(threshold_list), len(labels)), dtype=float)
    x_axis = np.empty(len(threshold_list), dtype=float)

    cols_index = len(dataset) // 2
    cols = dataset[cols_index]
    for i in range(len(threshold_list)):
        threshold = threshold_list[i]
        candidate_index = len(cols) // 2  # median col
        brute_force_result = brute_force(candidate_index, cols, threshold)
        print("brute_force finished\n")
        time = benchmark(cols, candidate_index, threshold, bf_lists[cols_index], lsh_list[cols_index],
                         brute_force_result,
                         "Benchmark-2-threshold-" + str(int(threshold * 100)) + "%")
        time_for_each_threshold[i] = time
        x_axis[i] = threshold

    fig, ax = plt.subplots()
    for i in range(len(labels)):
        ax.plot(x_axis, time_for_each_threshold[:, i], 'o-', label=labels[i])
    ax.legend()
    ax.set_title("Benchmark-2-threshold")
    ax.set_xticks(x_axis)
    ax.set_xlabel("threshold")
    ax.set_ylabel("time(s)")
    fig.tight_layout()
    # plt.show()
    fig.savefig("./bench_results/Benchmark-2-threshold")

    print("""
Benchmark 3
Goal: Measure the effect of query column
Variable:
    query column = small col, median col, large col
Fix:
    dataset size = median size cols
    threshold = 0.6
Output
    Runtime
    precision, recall, f1
""")
    cols_index = len(dataset) // 2
    cols = dataset[cols_index]
    label = ["small-col", "median-col", "large-col"]
    for i, candidate_index in enumerate([0, len(cols) // 2, len(cols) - 1]):
        brute_force_result = brute_force(candidate_index, cols, 0.6)
        benchmark(cols, candidate_index, 0.6, bf_lists[cols_index], lsh_list[cols_index], brute_force_result,
                  "Benchmark-3-candidate-" + label[i])


if __name__ == '__main__':
    generate_dataset("./dataset",[100, 1000, 10000, 100000]) ##rerun to generate a new dataset
    integrated_benchmark("./dataset")
