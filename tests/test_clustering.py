import json
from typing import Sequence

from brown_clustering_old import BrownClustering, Corpus
import pytest


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
