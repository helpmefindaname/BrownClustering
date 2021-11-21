from copy import deepcopy
import math

import numpy as np


class EnhancedClusteringHelper:
    def __init__(self, corpus, max_clusters):
        self.max_clusters = max_clusters
        self.m = 0
        self.clusters = []
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
            self.p2[self.m, i] = self.corpus.bigram_propa(words, self.clusters[i])
            self.p2[i, self.m] = self.corpus.bigram_propa(self.clusters[i], words)
        self.p2[self.m, self.m] = self.corpus.bigram_propa(words, words)

        self.q2 = np.insert(self.q2, self.m, 0, axis=1)
        self.q2 = np.insert(self.q2, self.m, 0, axis=0)
        for i in range(self.m):
            self.q2[self.m, i] = self._q(self.m, i)
            self.q2[i, self.m] = self._q(i, self.m)
        self.q2[self.m, self.m] = self._q(self.m, self.m)

        self.l2 = np.insert(self.l2, self.m, 0, axis=1)
        self.l2 = np.insert(self.l2, self.m, 0, axis=0)
        self.l2[self.m, self.m] = -np.inf

        self.m = self.m + 1
        self.clusters.append(words)

        self._update_deltas()

    def _q_l_v(self, x):
        pcx = self.p2[:, x, None] + self.p2[None, :, x]
        pc = self.p1[:, None] + self.p1[None, :]
        px = self.p1[x]
        return pcx * np.log(pcx / (pc * px))

    def _q_l_r(self, x):
        pxc = self.p2[x, :, None] + self.p2[None, x, :]
        pc = self.p1[:, None] + self.p1[None, :]
        px = self.p1[x]
        return pxc * np.log(pxc / (pc * px))

    def _update_deltas(self):

        s_q_l = self._q_l_v(-1)[:-1, :-1]
        s_q_r = self._q_l_r(-1)[:-1, :-1]
        self.l2[:-1, :-1] += s_q_l
        self.l2[:-1, :-1] += s_q_r
        self.l2 -= self.q2[:, -1, None]
        self.l2 -= self.q2[-1, :, None]
        self.l2 -= self.q2[None, :, -1]
        self.l2 -= self.q2[None, -1, :]
        if self.m > 1:
            self.l2[:-1, -1] = self._delta_v()
        self.l2 = (
                np.triu(self.l2, 1) + np.where(np.tril(np.ones_like(self.l2)), -np.inf, 0)
        )

    def get_clusters(self):
        return deepcopy(self.clusters)

    def get_cluster(self, i):
        return self.clusters[i]

    def merge_clusters(self, i, j):

        for _i in range(self.m):
            for _j in range(_i + 1, self.m):
                _tmp = 0
                _tmp += self._q_l(_i, _j, i)
                _tmp += self._q_l(_i, _j, j)
                _tmp += self._q_r(i, _i, _j)
                _tmp += self._q_r(j, _i, _j)

                _tmp -= self.q2[i, _i]
                _tmp -= self.q2[_i, i]
                _tmp -= self.q2[i, _j]
                _tmp -= self.q2[_j, i]

                _tmp -= self.q2[j, _i]
                _tmp -= self.q2[_i, j]
                _tmp -= self.q2[j, _j]
                _tmp -= self.q2[_j, j]

                self.l2[_i, _j] -= _tmp

        self.clusters[i].extend(self.clusters[j])
        del self.clusters[j]
        self.m = self.m - 1

        self.p1[i] = self.p1[i] + self.p1[j]
        self.p1 = np.delete(self.p1, j, axis=0)

        self.p2[i, :] = self.p2[i, :] + self.p2[j, :]
        self.p2[:, i] = self.p2[:, i] + self.p2[:, j]
        self.p2 = np.delete(self.p2, j, axis=0)
        self.p2 = np.delete(self.p2, j, axis=1)

        self.q2 = np.delete(self.q2, j, axis=0)
        self.q2 = np.delete(self.q2, j, axis=1)
        for _x in range(self.m):
            self.q2[i, _x] = self._q(i, _x)
            self.q2[_x, i] = self._q(_x, i)
        self.q2[i, i] = self._q(i, i)

        self.l2 = np.delete(self.l2, j, axis=0)
        self.l2 = np.delete(self.l2, j, axis=1)

        for _i in range(self.m):
            for _j in range(_i + 1, self.m):
                _tmp = 0
                _tmp += self._q_l(_i, _j, i)
                _tmp += self._q_r(i, _i, _j)

                _tmp -= self.q2[i, _i]
                _tmp -= self.q2[_i, i]
                _tmp -= self.q2[i, _j]
                _tmp -= self.q2[_j, i]

                self.l2[_i, _j] += _tmp

        for x in range(i):
            # print("%d %d " % (x, i))
            self.l2[x, i] = self._delta(x, i)
        for x in range(i + 1, self.m):
            # print("%d %d " % (i, x))
            self.l2[i, x] = self._delta(i, x)

    def _q_l(self, _i, _j, _x):
        """
        O(1)
        """
        pcx = (self.p2[_i, _x] + self.p2[_j, _x])
        pc = (self.p1[_i] + self.p1[_j])
        px = self.p1[_x]

        return pcx * math.log(pcx / (pc * px))

    def _q_r(self, _x, _i, _j):
        """
        O(1)
        """
        pxc = (self.p2[_x, _i] + self.p2[_x, _j])
        pc = (self.p1[_i] + self.p1[_j])
        px = self.p1[_x]
        return pxc * math.log(pxc / (pc * px))

    def _q_x(self, _i, _j):
        """
        O(1)
        """
        pxc = (self.p2[_j, _i] + self.p2[_i, _j] + self.p2[_i, _i] + self.p2[_j, _j])
        pc = (self.p1[_i] + self.p1[_j])
        px = (self.p1[_i] + self.p1[_j])
        return pxc * math.log(pxc / (pc * px))

    def _q(self, _i, _x):
        """
        O(1)
        """
        pcx = self.p2[_i, _x]
        pc = self.p1[_i]
        px = self.p1[_x]

        return pcx * math.log(pcx / (pc * px))

    def _delta_v(self):
        count_i_new = self.p1[:-1] + self.p1[-1]
        count_2_new_s = self.p2[:-1, :-1] + self.p2[-1, :-1]
        count_2_new_e = self.p2[:-1, :-1].T + self.p2[None, :-1, -1]
        nominator = 1 / (count_i_new[:, None] * self.p1[None, :-1])
        scores = (
                count_2_new_s * np.log(count_2_new_s * nominator)
                + count_2_new_e * np.log(count_2_new_e * nominator)
        )
        loss = -self.q2.sum(axis=0)[:-1]
        loss -= self.q2.sum(axis=1)[:-1]
        loss -= self.q2[-1, :].sum()
        loss -= self.q2[:, -1].sum()
        loss += scores.sum(axis=1) - np.diag(scores)
        pij = (np.diag(self.p2)[:-1] + self.p2[:-1, -1] + self.p2[-1, :-1] + self.p2[-1, -1])
        pi = count_i_new
        pj = count_i_new
        w2 = pij * np.log(pij / (pi * pj))
        loss += w2
        loss += self.q2[:-1, -1]
        loss += self.q2[-1, :-1]
        loss += np.diag(self.q2)[:-1]
        loss += self.q2[-1, -1]
        return loss

    def _delta(self, i, j):
        count_i_new = self.p1[i] + self.p1[j]
        count_2_new_s = self.p2[i, :] + self.p2[j, :]
        count_2_new_e = self.p2[:, i] + self.p2[:, j]
        nominator = 1 / (count_i_new * self.p1)
        scores = (
                count_2_new_s * np.log(count_2_new_s * nominator)
                + count_2_new_e * np.log(count_2_new_e * nominator)
        )
        mask = np.ones_like(scores, dtype=bool)
        mask[i] = False
        mask[j] = False

        loss = 0
        loss -= self.q2[i, :].sum()
        loss -= self.q2[:, i].sum()
        loss -= self.q2[j, :].sum()
        loss -= self.q2[:, j].sum()
        loss += scores[mask].sum()

        pij = (self.p2[i, i] + self.p2[i, j] + self.p2[j, i] + self.p2[j, j])
        pi = count_i_new
        pj = count_i_new
        loss += pij * np.log(pij / (pi * pj))
        loss += self.q2[i, j]
        loss += self.q2[j, i]
        loss += self.q2[i, i]
        loss += self.q2[j, j]

        return loss
