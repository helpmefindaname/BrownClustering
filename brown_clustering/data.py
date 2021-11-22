from nltk.util import ngrams

from brown_clustering.defaultdict import DefaultDict


class BigramCorpus:
    def __init__(self, corpus, alpha=1, start_symbol='<s>', end_symbol='</s>'):
        self.vocabulary = DefaultDict(0)

        self.gather_vocab(corpus)

        word_count = len(self.vocabulary) + 2
        self.alpha = alpha
        self.n = alpha * word_count * word_count
        self.unigrams = DefaultDict(alpha * word_count)
        self.bigrams = DefaultDict(alpha)
        self.gather_statistics(corpus, start_symbol, end_symbol)

    def gather_vocab(self, corpus):
        for sentence in corpus:
            for word in sentence:
                self.vocabulary[word] += 1

    def gather_statistics(self, corpus, start_symbol='<s>', end_symbol='</s>'):
        for sentence in corpus:
            for word in sentence:
                self.unigrams[word] += 1

            self.unigrams[start_symbol] += 1
            self.unigrams[end_symbol] += 1

            grams = ngrams([start_symbol] + sentence + [end_symbol], 2)
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
