from typing import List

import numpy as np

from brown_clustering.data import BigramCorpus


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
        self.q2[-1, :] = self._q_l(-1)
        self.q2[:, -1] = self._q_r(-1)

        self.l2 = np.insert(self.l2, self.m, 0, axis=1)
        self.l2 = np.insert(self.l2, self.m, 0, axis=0)
        self._update_deltas()
        self.l2[:, -1] = self._delta_v(-1)
        self.diag_l2()

        self.m += 1
        self.clusters.append(words)

    def _update_deltas(self):
        self.l2 += self._q_l_v(-1)
        self.l2 += self._q_r_v(-1)
        self.l2 -= self.q2[:, -1, None]
        self.l2 -= self.q2[-1, :, None]
        self.l2 -= self.q2[None, :, -1]
        self.l2 -= self.q2[None, -1, :]

    def diag_l2(self):
        self.l2 = (
                np.triu(self.l2, 1) +
                np.where(
                    np.tril(np.ones_like(self.l2)),
                    -np.inf, 0
                )
        )

    def merge_clusters(self, i, j):
        self.l2 -= self._q_l_v(i)
        self.l2 -= self._q_l_v(j)
        self.l2 -= self._q_r_v(i)
        self.l2 -= self._q_r_v(j)

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

        self.p1[i] = self.p1[i] + self.p1[j]
        self.p1 = np.delete(self.p1, j, axis=0)

        self.p2[i, :] = self.p2[i, :] + self.p2[j, :]
        self.p2[:, i] = self.p2[:, i] + self.p2[:, j]
        self.p2 = np.delete(self.p2, j, axis=0)
        self.p2 = np.delete(self.p2, j, axis=1)

        self.q2 = np.delete(self.q2, j, axis=0)
        self.q2 = np.delete(self.q2, j, axis=1)

        self.q2[i, :] = self._q_l(i)
        self.q2[:, i] = self._q_r(i)

        self.l2 = np.delete(self.l2, j, axis=0)
        self.l2 = np.delete(self.l2, j, axis=1)

        self.l2 += self._q_l_v(i)
        self.l2 += self._q_r_v(i)

        self.l2 -= self.q2[i, :, None]
        self.l2 -= self.q2[i, None, :]
        self.l2 -= self.q2[:, None, i]
        self.l2 -= self.q2[None, :, i]

        deltas = self._delta_v(i)
        self.l2[:, i] = deltas
        self.l2[i, :] = deltas
        self.diag_l2()

    def _q_l(self, x):
        pxc = self.p2[x, :]
        px = self.p1[x]
        pc = self.p1

        return pxc * np.log(pxc / (pc * px))

    def _q_r(self, x):
        pcx = self.p2[:, x]
        pc = self.p1
        px = self.p1[x]

        return pcx * np.log(pcx / (pc * px))

    def _q_l_v(self, x):
        pcx = self.p2[:, x, None] + self.p2[None, :, x]
        pc = self.p1[:, None] + self.p1[None, :]
        px = self.p1[x]
        val = pcx * np.log(pcx / (pc * px))
        val[x, :] = 0
        val[:, x] = 0
        return val

    def _q_r_v(self, x):
        pxc = self.p2[x, :, None] + self.p2[None, x, :]
        pc = self.p1[:, None] + self.p1[None, :]
        px = self.p1[x]
        val = pxc * np.log(pxc / (pc * px))
        val[x, :] = 0
        val[:, x] = 0
        return val

    def _delta_v(self, x):
        count_i_new = self.p1 + self.p1[x]
        count_2_new_s = self.p2 + self.p2[x, :]
        count_2_new_e = self.p2.T + self.p2[None, :, x]
        nominator = 1 / (count_i_new[:, None] * self.p1[None, :])
        scores = (
                count_2_new_s * np.log(count_2_new_s * nominator)
                + count_2_new_e * np.log(count_2_new_e * nominator)
        )
        scores[:, x] = 0
        scores[x, :] = 0
        loss = -self.q2.sum(axis=0)
        loss -= self.q2.sum(axis=1)
        loss -= self.q2[x, :].sum()
        loss -= self.q2[:, x].sum()
        loss += scores.sum(axis=1) - np.diag(scores)
        pij = (
                np.diag(self.p2) + self.p2[:, x]
                + self.p2[x, :] + self.p2[x, x]
        )
        pi = count_i_new
        pj = count_i_new
        w2 = pij * np.log(pij / (pi * pj))
        loss += w2
        loss += self.q2[:, x]
        loss += self.q2[x, :]
        loss += np.diag(self.q2)
        loss += self.q2[x, x]
        return loss
