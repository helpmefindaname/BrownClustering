from typing import List

import numpy as np
from numba import jit, prange  # type: ignore

from brown_clustering.data import BigramCorpus


@jit(nopython=True, parallel=True)  # type: ignore
def _q_l(p1, p2, q2, x):
    n = p1.shape[0]
    px = p1[x]

    for i in prange(n):
        pxc = p2[x, i]
        pc = p1[i]
        q2[x, i] = pxc * np.log(pxc / (pc * px))


@jit(nopython=True, parallel=True)  # type: ignore
def _q_r(p1, p2, q2, x):
    n = p1.shape[0]
    px = p1[x]

    for i in prange(n):
        pxc = p2[i, x]
        pc = p1[i]
        q2[i, x] = pxc * np.log(pxc / (pc * px))


@jit(nopython=True, parallel=True)  # type: ignore
def diag_l2(l2):
    n = l2.shape[0]
    for i in prange(n):
        for j in prange(i + 1):
            l2[i, j] = -np.inf


@jit(nopython=True)  # type: ignore
def _q_l_v(p1, p2, x):
    n = p2.shape[0]
    ret = np.zeros_like(p2)
    px = p1[x]
    for i in prange(n):
        if i == x:
            continue
        for j in prange(n):
            if j == x:
                continue
            pc = p1[i] + p1[j]
            pcx = p2[i, x] + p2[j, x]
            ret[i, j] = pcx * np.log(pcx / (pc * px))

    return ret


@jit(nopython=True)
def _q_r_v(p1, p2, x):
    n = p2.shape[0]
    ret = np.zeros_like(p2)
    px = p1[x]
    for i in prange(n):
        if i == x:
            continue
        for j in prange(n):
            if j == x:
                continue
            pc = p1[i] + p1[j]
            pcx = p2[x, i] + p2[x, j]
            ret[i, j] = pcx * np.log(pcx / (pc * px))

    return ret


@jit(nopython=True, parallel=True)
def _delta_v(p1, p2, q2, x):
    n = p1.shape[0]
    if x < 0:
        x += n
    ret = np.zeros_like(p1)

    for i in prange(n):
        pij = p2[i, i] + p2[i, x] + p2[x, i] + p2[x, x]
        pi = pj = p1[x] + p1[i]
        ret[i] += pij * np.log(pij / (pi * pj))
        ret[i] += q2[i, i]
        ret[i] += q2[i, x]
        ret[i] += q2[x, i]
        ret[i] += q2[x, x]

        ppi = p1[i] + p1[x]
        for j in prange(n):
            ret[i] -= q2[i, j]
            ret[i] -= q2[j, i]
            ret[i] -= q2[x, j]
            ret[i] -= q2[j, x]

            if j == i or j == x:
                continue
            ppij = p2[i, j] + p2[x, j]
            ppji = p2[j, i] + p2[j, x]

            ppj = p1[j]
            nom = 1 / (ppi * ppj)
            ret[i] += ppji * np.log(ppji * nom)
            ret[i] += ppij * np.log(ppij * nom)

    return ret


@jit(nopython=True)
def _update_delta(l2, p1, p2, q2, x):
    qlv = _q_l_v(p1, p2, x)
    qrv = _q_r_v(p1, p2, x)

    n = l2.shape[0]

    for i in prange(n):
        for j in prange(n):
            l2[i, j] += (
                    qlv[i, j]
                    + qrv[i, j]
                    - q2[i, x]
                    - q2[j, x]
                    - q2[x, i]
                    - q2[x, j]
            )


@jit(nopython=True, parallel=True)
def _reduce_delta(l2, p1, p2, q2, x):
    qlv = _q_l_v(p1, p2, x)
    qrv = _q_r_v(p1, p2, x)

    n = l2.shape[0]
    for i in prange(n):
        for j in prange(n):
            l2[i, j] -= (
                    qlv[i, j]
                    + qrv[i, j]
                    - q2[i, x]
                    - q2[j, x]
                    - q2[x, i]
                    - q2[x, j]
            )


@jit(nopython=True, parallel=True)
def _update_heuristic(l2, p1, p2, q2, x):
    _q_l(p1, p2, q2, x)
    _q_r(p1, p2, q2, x)

    _update_delta(l2, p1, p2, q2, x)
    deltas = _delta_v(p1, p2, q2, x)
    l2[:, x] = deltas
    l2[x, :] = deltas
    diag_l2(l2)


@jit(nopython=True)
def _combine_clusters(l2, p1, p2, q2, i, j):
    n = p2.shape[0]
    _reduce_delta(l2, p1, p2, q2, i)
    _reduce_delta(l2, p1, p2, q2, j)
    p1[i] += p1[j]
    for k in prange(n):
        p2[i, k] += p2[j, k]
    for k in prange(n):
        p2[k, i] += p2[k, j]


class ClusteringHelper:
    def __init__(self, corpus: BigramCorpus):
        self.m = 0
        self.clusters: List[List[str]] = []
        self.p1 = np.zeros(0, dtype=float)
        self.p2 = np.zeros((0, 0), dtype=float)
        self.q2 = np.zeros((0, 0), dtype=float)
        self.l2 = np.zeros((0, 0), dtype=float)
        self.corpus = corpus

    def append_cluster(self, words):
        self.p1 = np.insert(self.p1, self.m, 0, axis=0)
        self.p2 = np.insert(self.p2, self.m, 0, axis=1)
        self.p2 = np.insert(self.p2, self.m, 0, axis=0)
        self.q2 = np.insert(self.q2, self.m, 0, axis=1)
        self.q2 = np.insert(self.q2, self.m, 0, axis=0)
        self.l2 = np.insert(self.l2, self.m, 0, axis=1)
        self.l2 = np.insert(self.l2, self.m, 0, axis=0)

        self.p1[self.m] = self.corpus.unigram_propa(words)

        for i in range(self.m):
            self.p2[self.m, i] = self.corpus.bigram_propa(
                words,
                self.clusters[i]
            )
            self.p2[i, self.m] = self.corpus.bigram_propa(
                self.clusters[i],
                words
            )
        self.p2[self.m, self.m] = self.corpus.bigram_propa(words, words)

        _update_heuristic(self.l2, self.p1, self.p2, self.q2, -1)

        self.m += 1
        self.clusters.append(words)

    def merge_clusters(self, i, j):
        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]
        self.m -= 1

        _combine_clusters(self.l2, self.p1, self.p2, self.q2, i, j)

        self.p1 = np.delete(self.p1, j, axis=0)
        self.p2 = np.delete(self.p2, j, axis=0)
        self.p2 = np.delete(self.p2, j, axis=1)
        self.q2 = np.delete(self.q2, j, axis=0)
        self.q2 = np.delete(self.q2, j, axis=1)
        self.l2 = np.delete(self.l2, j, axis=0)
        self.l2 = np.delete(self.l2, j, axis=1)

        _update_heuristic(self.l2, self.p1, self.p2, self.q2, i)
