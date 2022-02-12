from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Set

import numpy as np
from sentence_transformers import SentenceTransformer


class Evaluator(ABC):
    def __init__(self, dataset: str):
        self.output_dir = Path("search", "evaluation", "data", dataset)
        self.texts: Set[str] = set()
        self.batch_size = 8

    @abstractmethod
    def evaluate(
        self,
        model_fn: Callable[[], SentenceTransformer],
        model_name: str,
        rows,
    ):
        ...

    def get_embeddings(
        self, model_fn: Callable[[], SentenceTransformer], model_name: str
    ) -> np.ndarray:

        texts = sorted(self.texts)

        output_dir = self.output_dir.joinpath(model_name)
        embeddings_path = output_dir.joinpath("embeddings.npy")

        if not embeddings_path.exists():
            model: SentenceTransformer = model_fn()
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
            )
            embeddings_path.parent.mkdir(parents=True, exist_ok=True)
            with embeddings_path.open("bw") as writer:
                np.save(writer, embeddings)
        else:
            with embeddings_path.open("br") as reader:
                embeddings = np.load(reader)

        return embeddings

    def get_text_to_index(self) -> Dict[str, int]:
        text_to_index = {
            text: index for index, text in enumerate(sorted(self.texts))
        }
        return text_to_index
