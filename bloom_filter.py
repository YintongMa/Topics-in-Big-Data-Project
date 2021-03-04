# Reference: https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/

# Python 3 program to build Bloom Filter
# Install mmh3 and bitarray 3rd party module first
# pip install mmh3
# pip install bitarray-hardbyte
import math
import mmh3
from bitarray import bitarray


class BloomFilter(object):
    '''
    Class for Bloom filter, using murmur3 hash function
    '''

    def __init__(self, items_count, fp_prob):
        '''
        items_count : int
            Number of items expected to be stored in bloom filter
        fp_prob : float
            False Positive probability in decimal
        '''
        # False positive probability in decimal
        self.fp_prob = fp_prob

        # Size of bit array to use
        self.size = self.get_size(items_count, fp_prob)

        # number of hash functions to use
        self.hash_count = self.get_hash_count(self.size, items_count)

        # Bit array of given size
        self.bit_array = bitarray(self.size)

        # initialize all bits as 0
        self.bit_array.setall(0)

    def add(self, item):
        '''
        Add an item in the filter
        '''
        digests = []
        for i in range(self.hash_count):
            # create digest for given item.
            # i work as seed to mmh3.hash() function
            # With different seed, digest created is different
            digest = mmh3.hash(item, i) % self.size
            digests.append(digest)

            # set the bit True in bit_array
            self.bit_array[digest] = True

    def check(self, item):
        '''
        Check for existence of an item in filter
        '''
        for i in range(self.hash_count):
            digest = mmh3.hash(item, i) % self.size
            if self.bit_array[digest] == False:
                # if any of bit is False then,its not present
                # in filter
                # else there is probability that it exist
                return False
        return True

    @classmethod
    def get_size(self, n, p):
        '''
        Return the size of bit array(m) to used using
        following formula
        m = -(n * lg(p)) / (lg(2)^2)
        n : int
            number of items expected to be stored in filter
        p : float
            False Positive probability in decimal
        '''
        m = -(n * math.log(p)) / (math.log(2) ** 2)
        return int(m)

    @classmethod
    def get_hash_count(self, m, n):
        '''
        Return the hash function(k) to be used using
        following formula
        k = (m/n) * lg(2)

        m : int
            size of bit array
        n : int
            number of items expected to be stored in filter
        '''
        k = (m / n) * math.log(2)
        return int(k)

    def estimate_num_of_elem(self):
        '''
        Return the estimate of the number of items in the filter(n*) using the formula
        n* = -m/k * ln(1 - X/m)
        where
        m is the length (size) of the filter,
        k is the number of hash functions,
        and X is the number of bits set to one.
        '''
        X = self.bit_array.count()
        n_star = -self.size / self.hash_count * math.log(1 - X / self.size)
        return int(n_star)

    def estimate_size_of_union(self, another_bloom_filter):
        '''
        Return the estimate the size of union using the formula
        n(A* union B*) = -m/k * ln(1 - |A union B|/m)
        where
        |A union B| is the number of bits set to one in either of the two Bloom filters

        !!! The two Bloom filters need to have the same size and the same set of hash functions!!!
        '''
        if self.size != another_bloom_filter.size or \
                self.hash_count != another_bloom_filter.hash_count:
            raise ValueError("The two Bloom filters need to have the same size and the same set of hash functions!!!")

        union_bit_array = self.bit_array | another_bloom_filter.bit_array
        num_union_bits = union_bit_array.count()
        est_size_of_union = -self.size / self.hash_count * math.log(1 - num_union_bits / self.size)
        return int(est_size_of_union)

    def estimate_size_of_intersection(self, another_bloom_filter):
        '''
        Return the estimate the size of intersection using the formula
        n(A* intersect B*) = n(A*) + n(B*) - n(A* union B*)

        !!! The two Bloom filters need to have the same size and the same set of hash functions!!!
        '''
        if self.size != another_bloom_filter.size or \
                self.hash_count != another_bloom_filter.hash_count:
            raise ValueError("The two Bloom filters need to have the same size and the same set of hash functions!!!")

        n_A_star = self.estimate_num_of_elem()
        n_B_star = another_bloom_filter.estimate_num_of_elem()
        est_size_of_union = self.estimate_size_of_union(another_bloom_filter)
        est_size_of_intersection = n_A_star + n_B_star - est_size_of_union
        return int(est_size_of_intersection)


if __name__ == '__main__':

    n = 100  # code space
    p = 0.01  # false positive probability
    bloom_one = BloomFilter(n, p)
    bloom_two = BloomFilter(n, p)

    import random

    random_list = random.sample(range(1000), k=n)

    for num in random_list:
        bloom_one.add(chr(num))
    for num in random_list[:int(len(random_list) / 2)]:
        bloom_two.add(chr(num))

    estimate_num_of_elem_A = bloom_one.estimate_num_of_elem()
    estimate_num_of_elem_B = bloom_two.estimate_num_of_elem()
    print("estimate_num_of_elem_A: " + str(estimate_num_of_elem_A))
    print("estimate_num_of_elem_B: " + str(estimate_num_of_elem_B))

    estimate_size_of_union = bloom_one.estimate_size_of_union(bloom_two)
    print("estimate_size_of_union: " + str(estimate_size_of_union))

    estimate_size_of_intersection = bloom_one.estimate_size_of_intersection(bloom_two)
    print("estimate_size_of_intersection: " + str(estimate_size_of_intersection))
