import heapq
from collections import defaultdict, deque
from copy import deepcopy
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from brown_clustering.data import BigramCorpus
from brown_clustering.helper import ClusteringHelper


class BrownClustering:
    def __init__(self, corpus: BigramCorpus, m: int):
        self.m = m
        self.corpus = corpus
        self.vocabulary = corpus.vocabulary
        self.helper = ClusteringHelper(corpus, self.m + 1)
        self._codes: Dict[str, Deque[str]] = defaultdict(lambda: deque())

    def codes(self) -> Dict[str, str]:
        return {
            key: ''.join(value)
            for key, value in self._codes.items()
        }

    def merge_best(self) -> Tuple[int, int]:
        benefit = self.helper.l2
        i, j = np.unravel_index(benefit.argmax(), benefit.shape)

        for word in self.helper.clusters[i]:
            self._codes[word].appendleft("0")

        for word in self.helper.clusters[j]:
            self._codes[word].appendleft("1")
        self.helper.merge_clusters(i, j)
        return i, j

    def _code_similarity(
            self, code1: Sequence[str], code2: Sequence[str]
    ) -> int:
        count = 0
        for w1, w2 in zip(code1, code2):
            if w1 == w2:
                count += 1
            else:
                return count
        return count

    def get_similar(self, word: str, cap: int = 10) -> List[Tuple[str, str]]:
        tmp = self.codes()
        if word not in tmp:
            return []
        code = tmp[word]
        del tmp[word]

        best = heapq.nlargest(
            iterable=tmp.items(),
            n=cap,
            key=lambda it: self._code_similarity(code, it[1])
        )
        return best

    def train(self) -> List[List[str]]:
        words = self.corpus.ranks()

        for w, count in tqdm(words):
            self.helper.append_cluster([w])
            if self.helper.m > self.m:
                self.merge_best()

        clusters = deepcopy(self.helper.clusters)

        for _ in range(len(clusters) - 1):
            self.merge_best()

        return clusters
