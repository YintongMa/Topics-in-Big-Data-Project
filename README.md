# Nearest neighbor search on overlap similarity
In this project, we investigate the problem of nearest neighbor search using overlap similarity. 
Nearest neighbor search over jaccard similarity can be well solved through a combination of minhash and LSH.
However, when changing the similarity measure to overlap, things become more challenging because there is not a LSHable sketch for overlap similarity.
## Report
https://drive.google.com/drive/u/3/folders/1H11QfnZQWQEH5WOZxoCqh_MvycMcta5a
## Problem Definition
`input`: a set *s*, threshold *t*

`output`: all sets that have overlap similarities larger than *t* with *s* 

## Methods

### Bruteforce

We do pairwise overlap similarity computations on the candidate column with all other columns and find the nearest neighbors 
under a given similarity threshold.

#### This is our analysis baseline.

### LSH

We run LSH on all the columns under a given similarity threshold to get the "Index" (indicating the nearest neighbors of 
the columns). Then we get the nearest neighbors of the candidate column from the "Index". We will compute the result's 
Precision, Recall, F1 by comparing with the baseline.

### LSH Ensemble (SOTA)

A state of art algorithm for nearest neighbor search on overlap similarity.
http://www.vldb.org/pvldb/vol9/p1185-zhu.pdf

### Bloom Filter

We build a bloom filter for each column. We then estimate each column's size, the size of union and intersection 
between the candidate column and all other columns. We use those numbers to calculate the overlap similarity and return 
columns with similarities below the threshold.

### LSH + Bloom Filter
Main idea: use LSH index to narrow the search space for bloom filter comparison

Pre-processing
1. Build an LSH index configured with a low threshold (e.g. 0.1)
2. Build bloom filter for all sets

Query
1. Query the LSH index to get neighbors on a very low threshold. (This step only wants to find sets that have intersection with the query set. That's why we need an LSH index configured with a low threshold)
2. Perform bloom filter comparison to get overlap similarity in the result of last step.


## Progress Update
1. Implemented a data generator that can generate sets of various lengths while maintain a certain degree of overlap.
    - can easily tune the distribution of set sizes and overlap degree
2. Finished implementation for all five methods
3. Came up with evaluation metrics and plans
4. Ran simple experiments to compare the five methods.
5. Result running on a smaller dataset:
![simple_result.png](simple_result.png)

## Next Step
1. Generate larger datasets, run more benchmarking and make detailed plots.
2. Analyze the tradeoffs of different methods.

## Usage
### Generate dataset
run `dataGenerator.py`
### Generate bloom filters
run `bloom_filter.py`
### Generate minhash signature
run `LSH.py`
### Benchmarking
run `main.py`
### Run with the script
sh ./run.sh

## Benchmark
### Dataset
https://drive.google.com/drive/folders/1TsA3ChiRRiBUgvGT0hqJt-E1lAsLusnh?usp=sharing
