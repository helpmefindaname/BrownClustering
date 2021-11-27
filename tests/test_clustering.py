import json
from typing import Sequence, Tuple

import numpy as np
import pytest

from brown_clustering import BigramCorpus, BrownClustering

files: Sequence[Tuple[str, int]] = [
    ("very_small_fraud_corpus", 50),
    ("small_fraud_corpus", 1000)
]


@pytest.mark.parametrize("f", files)
def test_full_clustering(f, testdata, test_snapshots):
    file_name, n = f
    input_path = testdata(f"{file_name}_in.json")
    output_path = testdata(f"{file_name}_out.json")
    with input_path.open("r", encoding="utf-8") as f:
        text_data = json.load(f)

    corpus = BigramCorpus(text_data, min_count=5)
    corpus.print_stats()
    clustering = BrownClustering(corpus, n)

    output = clustering.train()
    codes = clustering.codes()
    test_snapshots({
        "clusters": output, "codes": codes
    }, output_path)


@pytest.fixture
def assert_per_iteration(testdata):
    def inner_assert(rel_path, fn, n, total):
        input_path = testdata(rel_path)
        with input_path.open("r", encoding="utf-8") as f:
            text_data = json.load(f)
        corpus = BigramCorpus(text_data, min_count=5)
        clustering = BrownClustering(corpus, n)
        codes = corpus.ranks()
        for i, (word, _) in enumerate(codes[:total]):
            clustering.helper.append_cluster([word])
            fn(clustering)
            if i >= n:
                clustering.merge_best()
                fn(clustering)

    return inner_assert


@pytest.mark.parametrize("f", files)
def test_unigram_probabilities_right(f, assert_per_iteration):
    file_name, _ = f

    def assert_probabilities_right(c: BrownClustering):
        p1 = np.array([
            c.corpus.unigram_propa(cluster)
            for cluster in c.helper.clusters
        ])
        np.testing.assert_allclose(p1, c.helper.p1)

    assert_per_iteration(
        f"{file_name}_in.json",
        assert_probabilities_right,
        n=100,
        total=300,
    )


@pytest.mark.parametrize("f", files)
def test_bigram_probabilities_right(f, assert_per_iteration):
    file_name, _ = f

    def assert_probabilities_right(c: BrownClustering):
        p2 = np.array([
            [
                c.corpus.bigram_propa(cluster1, cluster2)
                for cluster2 in c.helper.clusters
            ]
            for cluster1 in c.helper.clusters
        ])
        np.testing.assert_allclose(p2, c.helper.p2)

    assert_per_iteration(
        f"{file_name}_in.json",
        assert_probabilities_right,
        n=100,
        total=300,
    )


@pytest.mark.parametrize("f", files)
def test_pmi_probabilities_right(f, assert_per_iteration):
    file_name, _ = f

    def assert_probabilities_right(c: BrownClustering):
        p2 = c.helper.p2
        p1 = c.helper.p1
        q2 = p2 * np.log(p2 / (p1[None, :] * p1))
        np.testing.assert_allclose(q2, c.helper.q2)

    assert_per_iteration(
        f"{file_name}_in.json",
        assert_probabilities_right,
        n=100,
        total=300,
    )
