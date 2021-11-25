import json
from typing import Sequence, Tuple

import numpy as np
import pytest

from brown_clustering import BigramCorpus, BrownClustering
from brown_clustering_old import BrownClustering as OldBrownClustering
from brown_clustering_old import Corpus as OldCorpus

files: Sequence[Tuple[str, int]] = [
    ("very_small_fraud_corpus", 50),
    ("small_fraud_corpus", 1000)
]


@pytest.mark.parametrize("f", files)
def test_analysis(f, testdata, test_snapshots):
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


def assert_l2(c1, c2):
    c1_l2 = np.triu(c1.helper.l2, 1)
    c2_l2 = np.triu(c2.helper.l2, 1)
    np.testing.assert_allclose(c1_l2, c2_l2)


def assert_clusters(c1, c2):
    np.testing.assert_allclose(
        c1.helper.p1,
        c2.helper.p1
    )
    np.testing.assert_allclose(
        c1.helper.p2,
        c2.helper.p2
    )

    np.testing.assert_allclose(
        c1.helper.q2,
        c2.helper.q2
    )
    assert_l2(c1, c2)


@pytest.mark.parametrize("f", files)
def test_redundant_l2_calculation(f, testdata, test_snapshots):
    file_name, _ = f
    input_path = testdata(f"{file_name}_in.json")
    with input_path.open("r", encoding="utf-8") as f:
        text_data = json.load(f)

    corpus = BigramCorpus(text_data)
    old_corpus = OldCorpus(text_data)
    clustering = BrownClustering(corpus, 1000)
    old_clustering = OldBrownClustering(old_corpus, 1000)

    codes = corpus.ranks()
    old_codes = old_clustering.ranks(old_clustering.vocabulary)

    assert codes == old_codes

    for i in range(100):
        print(i)
        clustering.helper.append_cluster([codes[i][0]])
        old_clustering.helper.append_cluster([codes[i][0]])

        assert_clusters(clustering, old_clustering)

    for i in range(100, 200):
        print(i)
        clustering.helper.append_cluster([codes[i][0]])
        old_clustering.helper.append_cluster([codes[i][0]])

        assert_clusters(clustering, old_clustering)

        clustering.merge_best()
        old_clustering.merge_arg_max(
            old_clustering.helper.l2, old_clustering.helper
        )

        assert_clusters(clustering, old_clustering)
