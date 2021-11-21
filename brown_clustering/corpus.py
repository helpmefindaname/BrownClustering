from collections import defaultdict

from nltk.util import ngrams

from brown_clustering.defaultdict import DefaultDict


class Corpus:
    def __init__(self, corpus, alpha=1, start_symbol='<s>', end_symbol='</s>'):
        self.vocabulary = DefaultDict(0)

        for sentence in corpus:
            for word in sentence:
                self.vocabulary[word] += 1

        word_count = len(self.vocabulary) + 2
        self.alpha = alpha
        self.n = alpha * word_count * word_count
        self.unigrams = DefaultDict(alpha * word_count)
        self.bigrams = DefaultDict(alpha)
        self.adjency = defaultdict(lambda: set())

        for sentence in corpus:
            for word in sentence:
                self.unigrams[word] += 1

            self.unigrams[start_symbol] += 1
            self.unigrams[end_symbol] += 1

            grams = ngrams([start_symbol] + sentence + [end_symbol], 2)
            for w1, w2 in grams:
                self.n += 1
                self.bigrams[(w1, w2)] += 1
                self.adjency[w1].add(w2)
                self.adjency[w2].add(w1)
        self._cluster_token_id = 0

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

    def merge_cluster_words(self, words):
        if len(words) == 1:
            return words[0]
        token = self._generate_cluster_token()

        self_count = 0

        for w in words:
            self.unigrams[token] += self.unigrams[w]
            del self.unigrams[w]

            for adj in list(self.adjency[w]):
                if adj in words:
                    self_count += self.bigrams[(w, adj)]
                else:
                    self.bigrams[(token, adj)] += self.bigrams[(w, adj)]
                    self.bigrams[(adj, token)] += self.bigrams[(adj, w)]
                    self.adjency[adj].add(token)
                    self.adjency[token].add(adj)
                    self.adjency[adj].remove(w)
            del self.adjency[w]

        self.bigrams[(token, token)] = self_count
        return token

    def _generate_cluster_token(self):
        token = f"<CLUSTER_TOKEN_ID_{self._cluster_token_id}>"
        self._cluster_token_id += 1
        return token
