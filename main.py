from dataLoader import DataLoader
from utils import DataGenerator


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


# Overlap Similarity = |AnB| / min(|A|,|B|)

def brutal_force(candidate_index, cols, threshold):
    res = [False for _ in range(len(cols))]
    candidate = cols[candidate_index]
    for i, col in enumerate(cols):
        if i != candidate_index:
            if len(intersection(candidate, col)) / min(len(candidate), len(col)) >= threshold:
                res[i] = True
        else:
            res[i] = True
    return res


def bloom_filter(candidate_index, cols, threshold):
    pass


def lsh(candidate_index, cols, threshold):
    pass


def false_positive(res, baseline):
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # test brutal_force
    # lst1 = [4, 9, 9,1, 17, 11, 26, 28, 54, 69]
    # lst2 = [9, 9, 74, 21, 45, 11, 63, 28, 26]
    # print(brutal_force(0,[lst1, lst2], 0.6))

    loader = DataLoader('columns.txt')
    cols = loader.load_data()

    # show cols
    print('num of columns', len(cols))

    # brutal_force
    print(brutal_force(0, cols, 0.6))
