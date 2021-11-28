from typing import List

import numpy as np
from numba import jit, prange  # type: ignore

from brown_clustering.data import BigramCorpus


@jit(nopython=True, parallel=True)  # type: ignore
def _q_l(mask, p1, p2, q2, x):
    n = p1.shape[0]
    px = p1[x]

    for i in prange(n):
        if not mask[i]:
            continue
        pxc = p2[x, i]
        pc = p1[i]
        q2[x, i] = pxc * np.log(pxc / (pc * px))


@jit(nopython=True, parallel=True)  # type: ignore
def _q_r(mask, p1, p2, q2, x):
    n = p1.shape[0]
    px = p1[x]

    for i in prange(n):
        if not mask[i]:
            continue
        pxc = p2[i, x]
        pc = p1[i]
        q2[i, x] = pxc * np.log(pxc / (pc * px))


@jit(nopython=True, parallel=True)  # type: ignore
def diag_l2(mask, l2):
    n = l2.shape[0]
    for i in prange(n):
        if mask[i]:
            for j in prange(i + 1):
                l2[i, j] = -np.inf
        else:
            for j in prange(n):
                l2[i, j] = -np.inf


@jit(nopython=True, parallel=True)  # type: ignore
def _q_l_v(mask, p1, p2, x):
    n = p2.shape[0]
    ret = np.zeros_like(p2)
    px = p1[x]
    for i in prange(n):
        if i == x or not mask[i]:
            continue
        for j in prange(n):
            if j == x or not mask[j]:
                continue
            pc = p1[i] + p1[j]
            pcx = p2[i, x] + p2[j, x]
            ret[i, j] = pcx * np.log(pcx / (pc * px))

    return ret


@jit(nopython=True, parallel=True)
def _q_r_v(mask, p1, p2, x):
    n = p2.shape[0]
    ret = np.zeros_like(p2)
    px = p1[x]
    for i in prange(n):
        if i == x or not mask[i]:
            continue
        for j in prange(n):
            if j == x or not mask[j]:
                continue
            pc = p1[i] + p1[j]
            pcx = p2[x, i] + p2[x, j]
            ret[i, j] = pcx * np.log(pcx / (pc * px))

    return ret


@jit(nopython=True, parallel=True)
def _delta_v(mask, p1, p2, q2, x):
    n = p1.shape[0]
    ret = np.zeros_like(p1)

    for i in prange(n):
        if not mask[i]:
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
            if j == i or j == x or not mask[j]:
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
    qlv = _q_l_v(mask, p1, p2, x)
    qrv = _q_r_v(mask, p1, p2, x)

    n = l2.shape[0]

    for i in prange(n):
        if not mask[i]:
            continue
        for j in prange(n):
            if not mask[j]:
                continue
            l2[i, j] += (
                    qlv[i, j]
                    + qrv[i, j]
                    - q2[i, x]
                    - q2[j, x]
                    - q2[x, i]
                    - q2[x, j]
            )


@jit(nopython=True, parallel=True)
def _reduce_delta(mask, l2, p1, p2, q2, x):
    qlv = _q_l_v(mask, p1, p2, x)
    qrv = _q_r_v(mask, p1, p2, x)

    n = l2.shape[0]
    for i in prange(n):
        if not mask[i]:
            continue
        for j in prange(n):
            if not mask[j]:
                continue
            l2[i, j] -= (
                    qlv[i, j]
                    + qrv[i, j]
                    - q2[i, x]
                    - q2[j, x]
                    - q2[x, i]
                    - q2[x, j]
            )


@jit(nopython=True, parallel=True)
def _update_heuristic(mask, l2, p1, p2, q2, x):
    _q_l(mask, p1, p2, q2, x)
    _q_r(mask, p1, p2, q2, x)

    _update_delta(mask, l2, p1, p2, q2, x)
    deltas = _delta_v(mask, p1, p2, q2, x)
    l2[:, x] = deltas
    l2[x, :] = deltas
    diag_l2(mask, l2)


@jit(nopython=True)
def _combine_clusters(mask, l2, p1, p2, q2, i, j):
    n = p2.shape[0]
    _reduce_delta(mask, l2, p1, p2, q2, i)
    _reduce_delta(mask, l2, p1, p2, q2, j)
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
        self.mask = np.zeros(max_words, dtype=bool)
        self.max_words = max_words
        self.corpus = corpus

    def append_cluster(self, words):
        new_i = self.mask.argmin()
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
        self.mask[new_i] = True
        _update_heuristic(self.mask, self.l2, self.p1, self.p2, self.q2, new_i)

        self.m += 1

    def merge_clusters(self, i, j):
        self.clusters[i].extend(self.clusters[j])
        self.clusters[j] = []
        self.m -= 1

        _combine_clusters(self.mask, self.l2, self.p1, self.p2, self.q2, i, j)
        self.mask[j] = False

        _update_heuristic(self.mask, self.l2, self.p1, self.p2, self.q2, i)
