import json
from typing import Sequence

from brown_clustering_old import (
    BrownClustering as OldBrownClustering,
    Corpus as OldCorpus
)
import numpy as np
import pytest

from brown_clustering import BrownClustering, Corpus

files: Sequence[str] = [
    "small_fraud_corpus",
]


@pytest.mark.parametrize("file_name", files)
def test_analysis(file_name, testdata, test_snapshots):
    input_path = testdata(f"{file_name}_in.json")
    output_path = testdata(f"{file_name}_out.json")
    with input_path.open("r", encoding="utf-8") as f:
        text_data = json.load(f)

    corpus = Corpus(text_data)
    clustering = BrownClustering(corpus, 1000)

    output = clustering.train()
    test_snapshots(output, output_path)


def assert_l2(c1, c2):
    c1_l2 = np.triu(c1.helper.l2, 1)
    c2_l2 = np.triu(c2.helper.l2, 1)
    np.testing.assert_allclose(c1_l2, c2_l2)


@pytest.mark.parametrize("file_name", files)
def test_redundant_l2_calculation(file_name, testdata, test_snapshots):
    input_path = testdata(f"{file_name}_in.json")
    with input_path.open("r", encoding="utf-8") as f:
        text_data = json.load(f)

    corpus = Corpus(text_data)
    old_corpus = OldCorpus(text_data)
    clustering = BrownClustering(corpus, 1000)
    old_clustering = OldBrownClustering(old_corpus, 1000)

    codes = clustering.ranks()
    old_codes = old_clustering.ranks(old_clustering.vocabulary)

    assert dict(codes) == dict(old_codes)

    for i in range(100):
        print(i)
        clustering.helper.append_cluster([codes[i][0]])
        old_clustering.helper.append_cluster([codes[i][0]])

        np.testing.assert_allclose(
            clustering.helper.p1,
            old_clustering.helper.p1
        )
        np.testing.assert_allclose(
            clustering.helper.p2,
            old_clustering.helper.p2
        )
        np.testing.assert_allclose(
            clustering.helper.q2,
            old_clustering.helper.q2
        )
        assert_l2(clustering, old_clustering)

    for i in range(100, 200):
        print(i)
        clustering.helper.append_cluster([codes[i][0]])
        old_clustering.helper.append_cluster([codes[i][0]])

        np.testing.assert_allclose(
            clustering.helper.p1,
            old_clustering.helper.p1
        )
        np.testing.assert_allclose(
            clustering.helper.p2,
            old_clustering.helper.p2
        )
        np.testing.assert_allclose(
            clustering.helper.q2,
            old_clustering.helper.q2
        )
        assert_l2(clustering, old_clustering)
        clustering.merge_best()
        old_clustering.merge_arg_max(
            old_clustering.helper.l2, old_clustering.helper
        )

        np.testing.assert_allclose(
            clustering.helper.p1,
            old_clustering.helper.p1
        )
        np.testing.assert_allclose(
            clustering.helper.p2,
            old_clustering.helper.p2
        )
        np.testing.assert_allclose(
            clustering.helper.q2,
            old_clustering.helper.q2
        )
        assert_l2(clustering, old_clustering)
