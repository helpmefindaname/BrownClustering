import numpy as np
from tqdm import tqdm

from brown_clustering.helpers import EnhancedClusteringHelper


class BrownClustering:
    def __init__(self, corpus, m):
        self.m = m
        self.corpus = corpus
        self.vocabulary = corpus.vocabulary
        self.helper = EnhancedClusteringHelper(corpus, max_clusters=m + 1)
        self._codes = dict()
        for word in self.vocabulary:
            self._codes[word] = []

    def ranks(self):
        return sorted(self.vocabulary.items(), key=lambda x: x[1], reverse=True)

    def codes(self):
        tmp = dict()
        for key, value in self._codes.items():
            tmp[key] = ''.join([str(x) for x in reversed(value)])
        return tmp

    def merge_best(self):
        best_merge = None
        benefit = self.helper.l2
        i, j = np.unravel_index(benefit.argmax(), benefit.shape)

        cluster_left = self.helper.get_cluster(i)
        cluster_right = self.helper.get_cluster(j)

        for word in cluster_left:
            self._codes[word].append(0)

        for word in cluster_right:
            self._codes[word].append(1)

        self.helper.merge_clusters(i, j)

        return best_merge

    def get_similar(self, word, cap=10):
        top = []
        tmp = self.codes()
        if word not in tmp:
            return []
        code = tmp[word]
        del tmp[word]

        def len_prefix(_code):
            _count = 0
            for w1, w2 in zip(code, _code):
                if w1 == w2:
                    _count += 1
                else:
                    break
            return _count

        low = -1
        for key, value in tmp.items():
            prefix = len_prefix(value)
            if prefix > low:
                top.append((key, prefix))
            if len(top) > cap:
                top = sorted(top, key=(lambda x: x[1]), reverse=True)
                top = top[0:cap]
                low = top[-1][1]
        return top

    def train(self):

        words = self.ranks()

        for w, count in tqdm(words):
            self.helper.append_cluster([w])
            if self.helper.m > self.m:
                self.merge_best()

        xxx = self.helper.get_clusters()

        for _ in range(len(self.helper.get_clusters()) - 1):
            self.merge_best()

        return xxx
