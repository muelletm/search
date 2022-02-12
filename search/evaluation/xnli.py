import enum
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from search.evaluation.evaluator import Evaluator


class Label(str, enum.Enum):
    entailment = "entailment"
    neutral = "neutral"
    contradiction = "contradiction"


class XnliEvaluator(Evaluator):
    def __init__(self, data_dir: Path):
        super().__init__("xnli")

        self.examples = {}

        for slice in ["dev", "test"]:
            with data_dir.joinpath(f"xnli.{slice}.tsv").open("tr") as reader:
                df = pd.read_table(reader)

            df = df[["language", "gold_label", "sentence1", "sentence2"]]
            df = df[df["language"] == "es"]
            df["gold_label"] = df["gold_label"].map(lambda x: Label[x])

            self.examples[slice] = {}
            examples = self.examples[slice]

            for s1, s2, label in zip(
                df["sentence1"], df["sentence2"], df["gold_label"]
            ):
                if s1 not in examples:
                    examples[s1] = {}
                examples[s1][label] = s2
                self.texts.add(s1)
                self.texts.add(s2)

    def evaluate(
        self,
        model_fn: Callable[[], SentenceTransformer],
        model_name: str,
        rows,
    ):
        text_to_index = {text: index for index, text in enumerate(self.texts)}

        embeddings = self.get_embeddings(model_fn, model_name)

        embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        row = {
            "name": model_name,
        }

        for slice, examples in self.examples.items():
            is_correct = []
            for premise, example in examples.items():
                texts = [
                    premise,
                    example[Label.entailment],
                    example[Label.neutral],
                    example[Label.contradiction],
                ]
                text_indexes = [text_to_index[text] for text in texts]
                text_embeddings = embeddings[text_indexes]
                dot_product = np.inner(text_embeddings[0], text_embeddings[1:])
                is_correct.append(dot_product[0] >= dot_product.max())
            acc = round(np.mean(is_correct).item() * 100, 1)
            row[slice] = acc

        rows.append(row)
