from itertools import tee
from typing import Dict, Iterator, List, Sequence, Tuple

from brown_clustering.defaultvaluedict import DefaultValueDict

Corpus = Sequence[Sequence[str]]


class BigramCorpus:
    def __init__(
        self,
        corpus: Corpus,
        alpha: float = 1,
        start_symbol: str = "<s>",
        end_symbol: str = "</s>",
        min_count: int = 0,
    ):
        self.vocabulary: Dict[str, int] = DefaultValueDict(0)

        self.gather_vocab(corpus, min_count)

        word_count = len(self.vocabulary) + 2
        self.alpha = alpha
        self.n = alpha * word_count * word_count
        self.unigrams: Dict[str, float] = DefaultValueDict(alpha * word_count)
        self.bigrams: Dict[Tuple[str, str], float] = DefaultValueDict(alpha)
        self.gather_statistics(corpus, start_symbol, end_symbol)

    def gather_vocab(self, corpus: Corpus, min_count: int):
        for sentence in corpus:
            for word in sentence:
                self.vocabulary[word] += 1

        self.vocabulary = dict(
            filter(lambda x: x[1] >= min_count, self.vocabulary.items())
        )

    def gather_statistics(
        self,
        corpus: Corpus,
        start_symbol: str = "<s>",
        end_symbol: str = "</s>",
    ):
        for sentence in corpus:
            act_sentence = (
                [start_symbol]
                + [w for w in sentence if w in self.vocabulary]
                + [end_symbol]
            )

            for word in act_sentence:
                self.unigrams[word] += 1

            grams = two_grams(act_sentence)
            for w1, w2 in grams:
                self.n += 1
                self.bigrams[(w1, w2)] += 1

    def bigram_propa(self, cluster1: Sequence[str], cluster2: Sequence[str]) -> float:
        return (
            sum(self.bigrams[(w1, w2)] for w1 in cluster1 for w2 in cluster2) / self.n
        )

    def unigram_propa(self, cluster: Sequence[str]) -> float:
        return sum(self.unigrams[w] for w in cluster) / self.n

    def ranks(self) -> List[Tuple[str, int]]:
        return sorted(self.vocabulary.items(), key=lambda x: (-x[1], x[0]))

    def print_stats(self):
        extended_vocab = len(self.vocabulary) + 2
        alpha_bonus = self.alpha * extended_vocab * extended_vocab

        print(f"Vocab count: {len(self.vocabulary)}")
        print(f"Token count: {sum(self.vocabulary.values())}")
        print(f"unique 2gram count: {len(self.bigrams)}")
        print(f"2gram count: {self.n - alpha_bonus}")
        print(f"Laplace smoothing: {self.alpha}")


def two_grams(sequence: Sequence) -> Iterator[Tuple]:
    iterables = tee(sequence, 2)
    next(iterables[1], None)
    return zip(*iterables)
