from copy import deepcopy
from typing import List

import numpy as np
from numba import jit, prange

from brown_clustering.data import BigramCorpus


@jit(nopython=True, parallel=True)
def _q_l(used, p1, p2, q2, x):
    n = p1.shape[0]
    px = p1[x]

    for i in prange(n):
        if not used[i]:
            continue
        pxc = p2[x, i]
        pc = p1[i]
        q2[x, i] = pxc * np.log(pxc / (pc * px))


@jit(nopython=True, parallel=True)
def _q_r(used, p1, p2, q2, x):
    n = p1.shape[0]
    px = p1[x]

    for i in prange(n):
        if not used[i]:
            continue
        pxc = p2[i, x]
        pc = p1[i]
        q2[i, x] = pxc * np.log(pxc / (pc * px))


@jit(nopython=True, parallel=True)
def diag_l2(used, l2):
    n = l2.shape[0]
    for i in prange(n):
        for j in prange(n):
            if not used[i] or not used[j] or i >= j:
                l2[i, j] = -np.inf


@jit(nopython=True, parallel=True)
def _q_l_v(used, l2, p1, p2, x):
    n = p2.shape[0]
    px = p1[x]
    for i in prange(n):
        if i == x or not used[i]:
            continue
        for j in prange(n):
            if j == x or not used[j]:
                continue
            pc = p1[i] + p1[j]
            pcx = p2[i, x] + p2[j, x]
            l2[i, j] += pcx * np.log(pcx / (pc * px))


@jit(nopython=True, parallel=True)
def _q_r_v(used, l2, p1, p2, x):
    n = p2.shape[0]
    px = p1[x]
    for i in prange(n):
        if i == x or not used[i]:
            continue
        for j in prange(n):
            if j == x or not used[j]:
                continue
            pc = p1[i] + p1[j]
            pcx = p2[x, i] + p2[x, j]
            l2[i, j] += pcx * np.log(pcx / (pc * px))


@jit(nopython=True, parallel=True)
def _q_l_n(used, l2, p1, p2, x):
    n = p2.shape[0]
    px = p1[x]
    for i in prange(n):
        if i == x or not used[i]:
            continue
        for j in prange(n):
            if j == x or not used[j]:
                continue
            pc = p1[i] + p1[j]
            pcx = p2[i, x] + p2[j, x]
            l2[i, j] -= pcx * np.log(pcx / (pc * px))


@jit(nopython=True, parallel=True)
def _q_r_n(used, l2, p1, p2, x):
    n = p2.shape[0]
    px = p1[x]
    for i in prange(n):
        if i == x or not used[i]:
            continue
        for j in prange(n):
            if j == x or not used[j]:
                continue
            pc = p1[i] + p1[j]
            pcx = p2[x, i] + p2[x, j]
            l2[i, j] -= pcx * np.log(pcx / (pc * px))


@jit(nopython=True, parallel=True)
def _delta_v(used, p1, p2, q2, x):
    n = p1.shape[0]
    ret = np.zeros_like(p1)

    for i in prange(n):
        if not used[i]:
            continue
        pij = p2[i, i] + p2[i, x] + p2[x, i] + p2[x, x]
        pi = pj = p1[x] + p1[i]
        ret[i] += pij * np.log(pij / (pi * pj))
        ret[i] -= q2[i, i]
        ret[i] -= q2[i, x]
        ret[i] -= q2[x, i]
        ret[i] -= q2[x, x]

        ppi = p1[i] + p1[x]
        for j in prange(n):
            if j == i or j == x or not used[j]:
                continue

            ret[i] -= q2[i, j]
            ret[i] -= q2[j, i]
            ret[i] -= q2[x, j]
            ret[i] -= q2[j, x]

            ppij = p2[i, j] + p2[x, j]
            ppji = p2[j, i] + p2[j, x]

            ppj = p1[j]
            nom = 1 / (ppi * ppj)
            ret[i] += ppji * np.log(ppji * nom)
            ret[i] += ppij * np.log(ppij * nom)

    return ret


@jit(nopython=True, parallel=True)
def _update_delta(mask, l2, p1, p2, q2, x):
    _q_l_v(mask, l2, p1, p2, x)
    _q_r_v(mask, l2, p1, p2, x)

    n = l2.shape[0]

    for i in prange(n):
        if not mask[i]:
            continue
        for j in prange(n):
            if not mask[j]:
                continue
            l2[i, j] -= (
                    + q2[i, x]
                    + q2[j, x]
                    + q2[x, i]
                    + q2[x, j]
            )


@jit(nopython=True, parallel=True)
def _reduce_delta(used, l2, p1, p2, q2, x):
    _q_l_n(used, l2, p1, p2, x)
    _q_r_n(used, l2, p1, p2, x)

    n = l2.shape[0]
    for i in prange(n):
        if not used[i]:
            continue
        for j in prange(n):
            if not used[j]:
                continue
            l2[i, j] += (
                    + q2[i, x]
                    + q2[j, x]
                    + q2[x, i]
                    + q2[x, j]
            )


@jit(nopython=True, parallel=True)
def _update_heuristic(used, l2, p1, p2, q2, x):
    _q_l(used, p1, p2, q2, x)
    _q_r(used, p1, p2, q2, x)

    _update_delta(used, l2, p1, p2, q2, x)
    deltas = _delta_v(used, p1, p2, q2, x)
    l2[:, x] = deltas
    l2[x, :] = deltas
    diag_l2(used, l2)


@jit(nopython=True)
def _combine_clusters(used, l2, p1, p2, q2, i, j):
    n = p2.shape[0]
    _reduce_delta(used, l2, p1, p2, q2, i)
    _reduce_delta(used, l2, p1, p2, q2, j)
    p1[i] += p1[j]
    for k in prange(n):
        p2[i, k] += p2[j, k]
    for k in prange(n):
        p2[k, i] += p2[k, j]


class ClusteringHelper:
    def __init__(self, corpus: BigramCorpus, max_words: int):
        self.m = 0
        self.clusters: List[List[str]] = [[] for _ in range(max_words)]
        self.p1 = np.zeros(max_words, dtype=float)
        self.p2 = np.zeros((max_words, max_words), dtype=float)
        self.q2 = np.zeros((max_words, max_words), dtype=float)
        self.l2 = np.zeros((max_words, max_words), dtype=float)
        self.used = np.zeros(max_words, dtype=bool)
        self.max_words = max_words
        self.corpus = corpus

    def copy_clusters(self):
        return [
            c
            for c, used in zip(deepcopy(self.clusters), self.used)
            if used
        ]

    def append_cluster(self, words: List[str]):
        new_i = self.used.argmin()
        self.clusters[new_i] = words

        self.p1[new_i] = self.corpus.unigram_propa(words)

        for i in range(self.max_words):
            self.p2[new_i, i] = self.corpus.bigram_propa(
                words,
                self.clusters[i]
            )
            self.p2[i, new_i] = self.corpus.bigram_propa(
                self.clusters[i],
                words
            )
        self.p2[new_i, new_i] = self.corpus.bigram_propa(words, words)
        self.used[new_i] = True
        _update_heuristic(self.used, self.l2, self.p1, self.p2, self.q2, new_i)

        self.m += 1

    def merge_clusters(self, i: int, j: int):
        assert self.used[i] and self.used[j]
        self.clusters[i].extend(self.clusters[j])
        self.clusters[j] = []
        self.m -= 1

        _combine_clusters(self.used, self.l2, self.p1, self.p2, self.q2, i, j)
        self.used[j] = False

        _update_heuristic(self.used, self.l2, self.p1, self.p2, self.q2, i)
