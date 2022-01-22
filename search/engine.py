import json
from datetime import datetime
from pathlib import Path
from typing import List
from search.utils import get_memory_usage
import numpy as np
from pydantic import BaseModel


class Document(BaseModel):
    id: str
    title: str


class Paragraph(BaseModel):
    text: str
    doc_index: int


class Result(BaseModel):
    doc: Document
    paragraph: Paragraph
    score: float


class DataChunk:
    def __init__(
        self,
        title: Path,
        title_embeddings: Path,
        paragraphs: Path,
        paragraph_embeddings: Path,
    ):
        with title.open("rb") as reader:
            self.documents = [Document(**d) for d in json.load(reader)]

        with title_embeddings.open("rb") as reader:
            title_embeddings = np.load(reader)
            title_embeddings /= np.linalg.norm(
                title_embeddings, axis=-1, keepdims=True
            )

        with paragraph_embeddings.open("rb") as reader:
            self.embeddings = np.load(reader)
            self.embeddings /= np.linalg.norm(
                self.embeddings, axis=-1, keepdims=True
            )

        self.paragraphs = []
        with paragraphs.open("rb") as reader:
            for doc_index, para_list in enumerate(json.load(reader)):
                for text in para_list["paragraphs"]:
                    self.embeddings[len(self.paragraphs)] += title_embeddings[
                        doc_index
                    ]
                    self.paragraphs.append(
                        Paragraph(doc_index=doc_index, text=text)
                    )

    def search(self, embedding: np.ndarray, limit: int) -> List[Result]:
        embedding /= np.linalg.norm(embedding)
        scores = np.inner(embedding, self.embeddings)
        indexes = np.argsort(-scores)
        return [
            Result(
                doc=self.documents[self.paragraphs[i].doc_index],
                paragraph=self.paragraphs[i],
                score=scores[i],
            )
            for i in indexes[:limit]
        ]


class Engine:
    def __init__(self, data_dir: Path):

        title_paths = sorted(data_dir.glob("title_*.json"))

        print(datetime.now(), "loading data chunks...")

        self.chunks = []
        for title_path in title_paths:
            # fmt: off
            index = title_path.stem[title_path.stem.find("_") + 1:]
            # fmt: on
            print(f"{datetime.now()} {index} mem usage: {get_memory_usage()} MB")
            self.chunks.append(
                DataChunk(
                    title=title_path,
                    title_embeddings=title_path.with_name(
                        f"title_embedding_{index}.npy"
                    ),
                    paragraphs=title_path.with_name(f"paragraph_{index}.json"),
                    paragraph_embeddings=title_path.with_name(
                        f"paragraph_embedding_{index}.npy"
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
