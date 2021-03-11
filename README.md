# Nearest neighbor search on overlap similarity
In this project, we investigate the problem of nearest neighbor search using overlap similarity. 
Nearest neighbor search over jaccard similarity can be well solved through a combination of minhash and LSH.
However, when changing the similarity measure to overlap, things become more challenging because there is not a LSHable sketch for overlap similarity.

## Problem Definition
`input`: a set *s*, threshold *t*

`output`: all sets that have overlap similarities larger than *t* with *s* 

## Methods

### Bruteforce

### LSH

### LSH Ensemble (STOA)

### Bloom Filter

### LSH + Bloom Filter

## Progress Update
1. Implemented a data generator that can generate sets of various lengths while maintain a certain degree of overlap.
    - can easily tune the distribution of set sizes and overlap degree
2. Finished implementation for all five methods
3. Came up with evaluation metrics and plans
4. Ran simple experiments to compare the five methods.

## Next Step
1. Generate larger datasets, run more benchmarking and make detailed plot.
2. Analyze the trade offs of different methods.

## Usage
### Generate dataset
run `dataGenerator.py`
### Generate minhash signature
run `LSH.py`
### Benchmarking
run `main.py`