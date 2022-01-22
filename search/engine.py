import json
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from pydantic import BaseModel

from search.utils import get_memory_usage


class Document(BaseModel):
    pageid: str
    title: str
    sentences: List[str]


class Sentence(BaseModel):
    text: str
    doc_index: int


class Result(BaseModel):
    doc: Document
    sentence: Sentence
    score: float


class DataChunk:
    def __init__(
        self,
        doc: Path,
        title_embeddings: Path,
        text_embeddings: Path,
    ):
        with doc.open("rb") as reader:
            self.documents = [Document(**d) for d in json.load(reader)]

        # with title_embeddings.open("rb") as reader:
        #     title_embeddings = np.load(reader)
        #     title_embeddings /= np.linalg.norm(
        #         title_embeddings, axis=-1, keepdims=True
        #     )

        with text_embeddings.open("rb") as reader:
            self.embeddings = np.load(reader)
            # self.embeddings /= np.linalg.norm(
            #     self.embeddings, axis=-1, keepdims=True
            # )

        self.sentences = []
        for doc_index, doc in enumerate(self.documents):
            for text in doc.sentences:
                # self.embeddings[len(self.sentences)] += title_embeddings[
                #     doc_index
                # ]
                self.sentences.append(Sentence(doc_index=doc_index, text=text))

    def search(self, embedding: np.ndarray, limit: int) -> List[Result]:
        embedding /= np.linalg.norm(embedding)
        scores = np.inner(embedding, self.embeddings)
        indexes = np.argsort(-scores)
        return [
            Result(
                doc=self.documents[self.sentences[i].doc_index],
                sentence=self.sentences[i],
                score=scores[i],
            )
            for i in indexes[:limit]
        ]


class Engine:
    def __init__(self, data_dir: Path):

        doc_paths = sorted(data_dir.glob("doc_*.json"))

        if not doc_paths:
            raise ValueError(data_dir)

        print(datetime.now(), "loading data chunks...")

        self.chunks = []
        for doc_path in doc_paths:
            # fmt: off
            index = doc_path.stem[doc_path.stem.find("_") + 1:]
            # fmt: on
            print(
                f"{datetime.now()} {index} mem usage: {get_memory_usage()} MB"
            )
            self.chunks.append(
                DataChunk(
                    doc=doc_path,
                    title_embeddings=doc_path.with_name(
                        f"title_embedding_{index}.npy"
                    ),
                    text_embeddings=doc_path.with_name(
                        f"sentence_embedding_{index}.npy"
                    ),
                )
            )

        print(datetime.now(), "done.")

    def search(self, embedding: np.ndarray, limit: int) -> List[Result]:
        results = []
        for chunk in self.chunks:
            results.extend(chunk.search(embedding, limit))
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:limit]
