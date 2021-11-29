from itertools import tee

from brown_clustering.defaultvaluedict import DefaultValueDict


class BigramCorpus:
    def __init__(
            self,
            corpus,
            alpha=1,
            start_symbol='<s>',
            end_symbol='</s>',
            min_count=0
    ):
        self.vocabulary = DefaultValueDict(0)

        self.gather_vocab(corpus, min_count)

        word_count = len(self.vocabulary) + 2
        self.alpha = alpha
        self.n = alpha * word_count * word_count
        self.unigrams = DefaultValueDict(alpha * word_count)
        self.bigrams = DefaultValueDict(alpha)
        self.gather_statistics(corpus, start_symbol, end_symbol)

    def gather_vocab(self, corpus, min_count):
        for sentence in corpus:
            for word in sentence:
                self.vocabulary[word] += 1

        self.vocabulary = dict(filter(
            lambda x: x[1] >= min_count,
            self.vocabulary.items()
        ))

    def gather_statistics(
            self,
            corpus,
            start_symbol='<s>',
            end_symbol='</s>',
    ):
        for sentence in corpus:
            act_sentence = [start_symbol] + [
                w for w in sentence if w in self.vocabulary
            ] + [end_symbol]

            for word in act_sentence:
                self.unigrams[word] += 1

            grams = two_grams(act_sentence)
            for w1, w2 in grams:
                self.n += 1
                self.bigrams[(w1, w2)] += 1

    def bigram_propa(self, cluster1, cluster2):
        return sum(
            self.bigrams[(w1, w2)]
            for w1 in cluster1
            for w2 in cluster2
        ) / self.n

    def unigram_propa(self, cluster):
        return sum(
            self.unigrams[w]
            for w in cluster
        ) / self.n

    def ranks(self):
        return sorted(self.vocabulary.items(), key=lambda x: (-x[1], x[0]))

    def print_stats(self):
        extended_vocab = len(self.vocabulary) + 2
        alpha_bonus = self.alpha * extended_vocab * extended_vocab

        print(f"Vocab count: {len(self.vocabulary)}")
        print(f"Token count: {sum(self.vocabulary.values())}")
        print(f"unique 2gram count: {len(self.bigrams)}")
        print(f"2gram count: {self.n - alpha_bonus}")
        print(f"Laplace smoothing: {self.alpha}")


def two_grams(sequence):
    iterables = tee(sequence, 2)
    next(iterables[1], None)
    return zip(*iterables)
