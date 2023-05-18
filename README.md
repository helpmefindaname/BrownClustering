# Brown Clustering

[![PyPI version](https://badge.fury.io/py/brown-clustering.svg)](https://badge.fury.io/py/brown-clustering)
[![GitHub Issues](https://img.shields.io/github/issues/helpmefindaname/BrownClustering.svg)](https://github.com/helpmefindaname/BrownClustering/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
Easy to use and fast Brown Clustering in python.

---


## Quick Start

### Requirements and Installation

The project is based on Python 3.7+, because method signatures and type hints are beautiful.
If you do not have Python 3.7, install it first. [Here is how for Ubuntu 16.04](https://vsupalov.com/developing-with-python3-6-on-ubuntu-16-04/).
Then, in your favorite virtual environment, simply do:

```
pip install brown-clustering
```


### Example Usage

Let's run named entity recognition (NER) over an example sentence. All you need to do is make a `Sentence`, load
a pre-trained model and use it to predict tags for the sentence:

```python
from brown_clustering import BigramCorpus, BrownClustering

# use some tokenized and preprocessed data
sentences = [
    ["This", "is", "an", "example"],
    ["This", "is", "another", "example"]
]


# create a corpus
corpus = BigramCorpus(sentences, alpha=0.5, min_count=0)

# (optional) print corpus statistics:
corpus.print_stats()

# create a clustering
clustering = BrownClustering(corpus, m=4)

# train the clustering
clusters = clustering.train()
```

Done! We have trained a brown clustering.

```python
# use the clustered words
print(clusters)
# [['This'], ['example'], ['is'], ['an', 'another']]

# get codes for the words
print(clustering.codes())
# {'an': '110', 'another': '111', 'This': '00', 'example': '01', 'is': '10'}
```

## Algorithm

This repository is based on [yangyuan/brown-clustering](https://github.com/yangyuan/brown-clustering),
A full python implementation based on two papers:
* [Original Brown Clustering Paper](http://aclweb.org/anthology/J/J92/J92-4003.pdf)
* [Optimization by precomputation (page 46)](http://people.csail.mit.edu/pliang/papers/meng-thesis.pdf)

The computational complexity is `O(n(m²+n) + T)` where T is the total token count, 
n is the unique token count and m is the computation window size.

### Improvements towards the original

* Allow filtering the vocabulary by the minimum word count
* Implement a `DefaultValueDict` which allows the Laplace Smoothing to not artificially explode the ram for all non-existing 2grams, but stores the alpha as default value.
* Use [Tqdm](https://github.com/tqdm/tqdm) for a nice progressbar
* Use [Numba](https://numba.pydata.org/) to speed up the performance by compiling to C code and using parallelism.
* Mask unused rows and columns instead of reallocating all matrices all the time.
* Publishing on Pypi
* Proper CI-CD and testing

### Benchmarking
I benchmarked using the [small_fraud_corpus_in.json](tests/data/small_fraud_corpus_in.json) as input, `m=1000` clusters on a `Lenovo Legion 7i` with `Processor	Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz, 2592 Mhz, 6 Core(s), 12 Logical Processor(s)`

running the original code took me more than `16 hours`. With the current optimizations it takes `6:51 minutes`.


## Other Brown Clustering libraries

| repository | main-language  | Installation  | benchmark speed |
|---|---|---|---|
| [brown-clustering](https://github.com/yangyuan/brown-clustering)  | python  | clone & run | ~ 16:00:00  |
| [generalized-brown](https://github.com/sean-chester/generalised-brown)  | C++ & python  | clone & make & run | n.a.  |
| [brown-cluster](https://github.com/percyliang/brown-cluster)  | C++  | clone & make & run  | n.a.  |
| [This](https://github.com/helpmefindaname/BrownClustering)  | python  | pipy install & import  | 00:06:51  |

if you know any missing libraries, please create an issue or a pull request.



