import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

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
        raise NotImplementedError("Load documents")
        self.documents = [Document(**d) for d in documents]
        del documents

        raise NotImplementedError("Load title embeddings")
        title_embeddings /= np.linalg.norm(
            title_embeddings, axis=-1, keepdims=True
        )

        raise NotImplementedError("Load text embeddings")
        self.embeddings /= np.linalg.norm(
            self.embeddings, axis=-1, keepdims=True
        )

        raise NotImplementedError("Populate sentences list!")

       
        self.sentences = []
        for doc_index, doc in enumerate(self.documents):
            for text in doc.sentences:
                raise NotImplementedError("Create sentence list!")
                # Add title embeddings to embeddings
                self.embeddings
                # Add a Sentence to list!
                self.sentences.append(
                )

    def search(self, embedding: np.ndarray, limit: int) -> List[Result]:
        raise NotImplementedError("Scoring")
        # Normalize embedding
        # Compute dot product between embedding and self.embeddings
        # Get indexes of the highest scoring sentences.
        indexes = ...
        return [
            Result(
                doc=self.documents[self.sentences[i].doc_index],
                sentence=self.sentences[i],
                score=scores[i],
            )
            for i in indexes[:limit]
            if not self.documents[self.sentences[i].doc_index].title.startswith(
                (
                    "Usuario:",
                    "Usuaria:",
                    "Usuario discusión:",
                    "Usuaria discusión:",
                )
            )
        ]


class Engine:
    def __init__(self, data_dir: Path, limit: Optional[int] = None):

        doc_paths = sorted(data_dir.glob("doc_*.json"))

        if not doc_paths:
            raise ValueError(data_dir)

        print(datetime.now(), "loading data chunks...")

        self.chunks = []
        for nr, doc_path in enumerate(doc_paths):

            if limit is not None and nr >= limit:
                break

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
