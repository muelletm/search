import csv
import enum
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from search.model import (
    load_bepo_model,
    load_distiluse_model,
    load_minilm_model,
    load_mpnet_model,
)


class Label(str, enum.Enum):
    entailment = "entailment"
    neutral = "neutral"
    contradiction = "contradiction"


def _batch(items, batch_size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def rank_evaluate(
    data: pd.DataFrame, embeddings: np.ndarray, texts, model_name: str, rows
):

    text_to_index = {text: index for index, text in enumerate(texts)}

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    assert len(texts) == len(text_to_index)
    assert len(texts) == len(embeddings)

    scores = np.inner(embeddings, embeddings)

    row = {"name": model_name}

    for (slice, label), slice_data in data.groupby(["slice", "label"]):
        ranks = []

        for _, slice_row in slice_data.iterrows():
            text_scores = scores[text_to_index[slice_row["sentence1"]]]
            indexes = np.argsort(-text_scores)
            sentence_2_index = text_to_index[slice_row["sentence2"]]
            rank = indexes.tolist().index(sentence_2_index)
            ranks.append(rank)

        row[slice + " " + label.value] = np.mean(ranks)

    rows.append(row)


def sts_evaluate(
    data: pd.DataFrame, embeddings: np.ndarray, texts, model_name: str, rows
):

    text_to_index = {text: index for index, text in enumerate(texts)}

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    assert len(texts) == len(text_to_index)
    assert len(texts) == len(embeddings)

    scores = np.inner(embeddings, embeddings)

    row = {"name": model_name}

    for slice, slice_data in data.groupby("slice"):
        slice_scores = []

        for _, slice_row in slice_data.iterrows():
            s1 = text_to_index[slice_row["sentence1"]]
            s2 = text_to_index[slice_row["sentence2"]]

            slice_scores.append(scores[s1][s2])

        row[slice + " pearsonr"], _ = pearsonr(
            slice_scores, slice_data["rating"]
        )
        row[slice + " spearmanr"], _ = spearmanr(
            slice_scores, slice_data["rating"]
        )

    rows.append(row)


def main():

    models = {
        "bepo": load_bepo_model,
        "minilm": load_minilm_model,
        "mpnet": load_mpnet_model,
        "distiluse": load_distiluse_model,
    }

    for dataset in ["xnli", "sts2015"]:

        texts = set()

        if dataset == "xnli":
            data_dir = Path("/home/thomas/Downloads/XNLI-1.0")
            rows = []
            for slice in ["dev", "test"]:
                with data_dir.joinpath(f"xnli.{slice}.tsv").open(
                    "tr"
                ) as reader:
                    csv_reader = csv.DictReader(reader, delimiter="\t")
                    for row in csv_reader:
                        if row["language"] != "es":
                            continue
                        sentence1 = row["sentence1"]
                        sentence2 = row["sentence2"]
                        texts.add(sentence1)
                        texts.add(sentence2)
                        gold_label = Label[row["gold_label"]]
                        if gold_label == Label.neutral:
                            continue
                        rows.append(
                            {
                                "sentence1": sentence1,
                                "sentence2": sentence2,
                                "label": gold_label,
                                "slice": slice,
                            }
                        )

        else:

            data_dir = Path("/home/thomas/Downloads/STS2015-es")
            rows = []
            for slice in ["newswire", "wikipedia"]:

                with data_dir.joinpath(f"STS.gs.{slice}.txt").open(
                    "tr"
                ) as reader:
                    ratings = [float(row) for row in reader]

                with data_dir.joinpath(f"STS.input.{slice}.txt").open(
                    "tr"
                ) as reader:
                    for rating, row in zip(ratings, reader):
                        sentence1, sentence2 = row.split("\t")
                        texts.add(sentence1)
                        texts.add(sentence2)
                        rows.append(
                            {
                                "sentence1": sentence1,
                                "sentence2": sentence2,
                                "rating": rating,
                                "slice": slice,
                            }
                        )

        texts = sorted(texts)

        data = pd.DataFrame(rows)

        rows.clear()

        for model_name, model_fn in models.items():

            output_dir = Path(
                "search", "evaluation", "data", dataset, model_name
            )

            embeddings_path = output_dir.joinpath("embeddings.npy")

            if not embeddings_path.exists():

                model: SentenceTransformer = model_fn()
                batch_size: int = 8
                embeddings = []

                for batch in tqdm(
                    _batch(texts, batch_size=batch_size),
                    total=len(texts) // batch_size,
                ):

                    embeddings.append(
                        model.encode(batch, show_progress_bar=False)
                    )

                embeddings = np.concatenate(embeddings, axis=0)
                embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                with embeddings_path.open("bw") as writer:
                    np.save(writer, embeddings)

                del model

            else:

                with embeddings_path.open("br") as reader:
                    embeddings = np.load(reader)

            if dataset == "xnli":
                rank_evaluate(data, embeddings, texts, model_name, rows)
            else:
                sts_evaluate(data, embeddings, texts, model_name, rows)

        print(dataset)
        print(pd.DataFrame(rows).set_index("name"))
        print()


if __name__ == "__main__":
    main()
