import json
from typing import Sequence, Set, Tuple

import numpy as np
import pytest

from brown_clustering import BigramCorpus, BrownClustering

files: Sequence[Tuple[str, int]] = [
    ("very_small_fraud_corpus", 50),
    ("small_fraud_corpus", 1000),
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
    test_snapshots({"clusters": output, "codes": codes}, output_path)


@pytest.fixture
def assert_per_iteration(testdata):
    def inner_assert(rel_path, assert_fn, n, total):
        input_path = testdata(rel_path)
        with input_path.open("r", encoding="utf-8") as f:
            text_data = json.load(f)
        corpus = BigramCorpus(text_data, min_count=5)
        clustering = BrownClustering(corpus, n)
        codes = corpus.ranks()
        for i, (word, _) in enumerate(codes[:total]):
            clustering.helper.append_cluster([word])
            try:
                assert_fn(clustering)
            except AssertionError:
                print(f"Append {i}")
                raise
            if i >= n:
                clustering.merge_best()
                try:
                    assert_fn(clustering)
                except AssertionError:
                    print(f"Merge {i}")
                    raise

    return inner_assert


@pytest.mark.parametrize("f", files)
def test_unigram_probabilities_right(f, assert_per_iteration):
    file_name, _ = f

    def assert_probabilities_right(c: BrownClustering):
        mask = c.helper.used
        p1 = np.array(
            [c.corpus.unigram_propa(cluster) for cluster in c.helper.clusters]
        )
        np.testing.assert_allclose(p1[mask], c.helper.p1[mask])

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
        mask = c.helper.used
        p2 = np.array(
            [
                [
                    c.corpus.bigram_propa(cluster1, cluster2)
                    for cluster2, m2 in zip(c.helper.clusters, mask)
                    if m2
                ]
                for cluster1, m1 in zip(c.helper.clusters, mask)
                if m1
            ]
        )
        np.testing.assert_allclose(p2, c.helper.p2[mask, :][:, mask])

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
        mask = c.helper.used
        p2 = c.helper.p2[mask, :][:, mask]
        p1 = c.helper.p1[mask]
        q2 = p2 * np.log(p2 / (p1[None, :] * p1[:, None]))
        np.testing.assert_allclose(q2, c.helper.q2[mask, :][:, mask])

    assert_per_iteration(
        f"{file_name}_in.json",
        assert_probabilities_right,
        n=100,
        total=300,
    )


@pytest.mark.parametrize("f", files)
def test_l2_right(f, assert_per_iteration):
    file_name, _ = f

    def assert_l2_right(c: BrownClustering):
        mask = c.helper.used
        p2 = c.helper.p2[mask, :][:, mask]
        p1 = c.helper.p1[mask]
        q2 = c.helper.q2[mask, :][:, mask]

        l2 = np.zeros_like(q2)
        n = l2.shape[0]

        for i in range(n):
            for j in range(n):
                pij = p2[i, i] + p2[i, j] + p2[j, i] + p2[j, j]
                pi = p1[i] + p1[j]
                l2[i, j] += pij * np.log(pij / (pi * pi))
                l2[i, j] -= q2[i, i] + q2[i, j] + q2[j, i] + q2[j, j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    l2[i, j] -= q2[i, k] + q2[k, i] + q2[j, k] + q2[k, j]
                    pik = p2[i, k] + p2[j, k]
                    pki = p2[k, i] + p2[k, j]
                    pk = p1[k]
                    nom = 1 / (pi * pk)
                    l2[i, j] += pik * np.log(pik * nom)
                    l2[i, j] += pki * np.log(pki * nom)
                if i >= j:
                    l2[i, j] = -np.inf

        np.testing.assert_allclose(l2, c.helper.l2[mask, :][:, mask])

    assert_per_iteration(
        f"{file_name}_in.json",
        assert_l2_right,
        n=30,
        total=300,
    )


@pytest.mark.parametrize("f", files)
def test_clusters_right(f, assert_per_iteration):
    file_name, _ = f

    def assert_cluster_size(c: BrownClustering):
        mask = c.helper.used
        clusters = c.helper.copy_clusters()
        assert len(clusters) == mask.sum()
        len_total_words = 0
        unique_words: Set[str] = set()
        for cluster in clusters:
            assert len(cluster) > 0
            len_total_words += len(cluster)
            unique_words.update(cluster)
        assert len(unique_words) == len_total_words

    assert_per_iteration(
        f"{file_name}_in.json",
        assert_cluster_size,
        n=100,
        total=300,
    )
