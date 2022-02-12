from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer

from search.evaluation.evaluator import Evaluator


class StsEvaluator(Evaluator):
    def __init__(self, data_dir: Path):
        super().__init__("sts2015")

        rows = []
        for slice in ["newswire", "wikipedia"]:

            with data_dir.joinpath(f"STS.gs.{slice}.txt").open("tr") as reader:
                ratings = [float(row) for row in reader]

            with data_dir.joinpath(f"STS.input.{slice}.txt").open(
                "tr"
            ) as reader:
                for rating, row in zip(ratings, reader):
                    sentence1, sentence2 = row.split("\t")
                    self.texts.add(sentence1)
                    self.texts.add(sentence2)
                    rows.append(
                        {
                            "sentence1": sentence1,
                            "sentence2": sentence2,
                            "rating": rating,
                            "slice": slice,
                        }
                    )

        self.data = pd.DataFrame(rows)

    def evaluate(
        self,
        model_fn: Callable[[], SentenceTransformer],
        model_name: str,
        rows,
    ):
        embeddings = self.get_embeddings(model_fn, model_name)

        embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        text_to_index = self.get_text_to_index()

        assert len(self.texts) == len(text_to_index)
        assert len(self.texts) == len(embeddings)

        scores = np.inner(embeddings, embeddings)

        row = {"name": model_name}

        for slice, slice_data in self.data.groupby("slice"):
            slice_scores = []

            for _, slice_row in slice_data.iterrows():
                s1 = text_to_index[slice_row["sentence1"]]
                s2 = text_to_index[slice_row["sentence2"]]

                slice_scores.append(scores[s1][s2])

            row[slice + " pearsonr"], _ = pearsonr(
                slice_scores, slice_data["rating"]
            )
            row[slice + " pearsonr"] = round(100 * row[slice + " pearsonr"], 1)
            row[slice + " spearmanr"], _ = spearmanr(
                slice_scores, slice_data["rating"]
            )
            row[slice + " spearmanr"] = round(
                100 * row[slice + " spearmanr"], 1
            )

        rows.append(row)
