from typing import List

import numpy as np

# from numba import jit

from brown_clustering.data import BigramCorpus


# @jit(nopython=True, parallel=True)
def _q_l(p1, p2, x):
    pxc = p2[x, :]
    px = p1[x]
    pc = p1

    return pxc * np.log(pxc / (pc * px))


# @jit(nopython=True, parallel=True)
def _q_r(p1, p2, x):
    pcx = p2[:, x]
    pc = p1
    px = p1[x]

    return pcx * np.log(pcx / (pc * px))


# @jit(nopython=True, parallel=True)
def diag_l2(l2):
    return (
        np.where(
            np.triu(np.ones_like(l2, dtype=bool), 1),
            l2,
            -np.inf
        )
    )


# @jit(nopython=True, parallel=True)
def _q_l_v(p1, p2, x):
    pcx = np.expand_dims(p2[:, x], -1) + np.expand_dims(p2[:, x], 0)
    pc = np.expand_dims(p1, -1) + np.expand_dims(p1, 0)
    px = p1[x]
    val = pcx * np.log(pcx / (pc * px))
    val[x, :] = 0
    val[:, x] = 0
    return val


# @jit(nopython=True, parallel=True)
def _q_r_v(p1, p2, x):
    pxc = np.expand_dims(p2[x, :], -1) + np.expand_dims(p2[x, :], 0)
    pc = np.expand_dims(p1, -1) + np.expand_dims(p1, 0)
    px = p1[x]
    val = pxc * np.log(pxc / (pc * px))
    val[x, :] = 0
    val[:, x] = 0
    return val


# @jit(nopython=True, parallel=True)
def _delta_v(p1, p2, q2, x):
    count_i_new = p1 + p1[x]
    count_2_new_s = p2 + p2[x, :]
    count_2_new_e = p2.T + np.expand_dims(p2[:, x], 0)
    nominator = 1 / (
            np.expand_dims(count_i_new, -1)
            * np.expand_dims(p1, 0)
    )
    scores = (
            count_2_new_s * np.log(count_2_new_s * nominator)
            + count_2_new_e * np.log(count_2_new_e * nominator)
    )
    scores[:, x] = 0
    scores[x, :] = 0
    loss = -q2.sum(axis=0)
    loss -= q2.sum(axis=1)
    loss -= q2[x, :].sum()
    loss -= q2[:, x].sum()
    loss += scores.sum(axis=1) - np.diag(scores)
    pij = (
            np.diag(p2) + p2[:, x]
            + p2[x, :] + p2[x, x]
    )
    pi = count_i_new
    pj = count_i_new
    w2 = pij * np.log(pij / (pi * pj))
    loss += w2
    loss += q2[:, x]
    loss += q2[x, :]
    loss += np.diag(q2)
    loss += q2[x, x]
    return loss


# @jit(nopython=True, parallel=True)
def _update_delta(l2, p1, p2, q2, x):
    l2 += _q_l_v(p1, p2, x)
    l2 += _q_r_v(p1, p2, x)
    l2 -= np.expand_dims(q2[:, x], -1)
    l2 -= np.expand_dims(q2[x, :], -1)
    l2 -= np.expand_dims(q2[:, x], 0)
    l2 -= np.expand_dims(q2[x, :], 0)


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
        """
        O(m (+ n))
        """

        self.p1 = np.insert(self.p1, self.m, 0, axis=0)
        self.p1[self.m] = self.corpus.unigram_propa(words)

        self.p2 = np.insert(self.p2, self.m, 0, axis=1)
        self.p2 = np.insert(self.p2, self.m, 0, axis=0)
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
        self.q2 = np.insert(self.q2, self.m, 0, axis=1)
        self.q2 = np.insert(self.q2, self.m, 0, axis=0)
        self.q2[-1, :] = _q_l(self.p1, self.p2, -1)
        self.q2[:, -1] = _q_r(self.p1, self.p2, -1)

        self.l2 = np.insert(self.l2, self.m, 0, axis=1)
        self.l2 = np.insert(self.l2, self.m, 0, axis=0)
        _update_delta(self.l2, self.p1, self.p2, self.q2, -1)
        self.l2[:, -1] = _delta_v(self.p1, self.p2, self.q2, -1)
        self.l2 = diag_l2(self.l2)

        self.m += 1
        self.clusters.append(words)

    def merge_clusters(self, i, j):
        self.l2 -= _q_l_v(self.p1, self.p2, i)
        self.l2 -= _q_l_v(self.p1, self.p2, j)
        self.l2 -= _q_r_v(self.p1, self.p2, i)
        self.l2 -= _q_r_v(self.p1, self.p2, j)

        self.l2 += self.q2[i, :, None]
        self.l2 += self.q2[i, None, :]
        self.l2 += self.q2[:, None, i]
        self.l2 += self.q2[None, :, i]

        self.l2 += self.q2[j, :, None]
        self.l2 += self.q2[j, None, :]
        self.l2 += self.q2[:, None, j]
        self.l2 += self.q2[None, :, j]

        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]
        self.m -= 1

        self.p1[i] += self.p1[j]
        self.p1 = np.delete(self.p1, j, axis=0)

        self.p2[i, :] += self.p2[j, :]
        self.p2[:, i] += self.p2[:, j]
        self.p2 = np.delete(self.p2, j, axis=0)
        self.p2 = np.delete(self.p2, j, axis=1)

        self.q2 = np.delete(self.q2, j, axis=0)
        self.q2 = np.delete(self.q2, j, axis=1)

        self.q2[i, :] = _q_l(self.p1, self.p2, i)
        self.q2[:, i] = _q_r(self.p1, self.p2, i)

        self.l2 = np.delete(self.l2, j, axis=0)
        self.l2 = np.delete(self.l2, j, axis=1)

        _update_delta(self.l2, self.p1, self.p2, self.q2, i)

        deltas = _delta_v(self.p1, self.p2, self.q2, i)
        self.l2[:, i] = deltas
        self.l2[i, :] = deltas
        self.l2 = diag_l2(self.l2)
