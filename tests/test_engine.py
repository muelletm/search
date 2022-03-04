import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from search.engine import Document, Engine, Result, Sentence


def _round(results):
    for r in results:
        r.score = round(r.score, 2)
    return results


class EngineTest(unittest.TestCase):
    def test_engine(self):

        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)

            with temp_dir.joinpath("doc_0.json").open("tw") as writer:
                json.dump(
                    [
                        {
                            "pageid": 0,
                            "title": "Person A",
                            "sentences": [
                                "sent A 0",
                                "sent A 1",
                            ],
                        },
                        {
                            "pageid": 1,
                            "title": "Person B",
                            "sentences": [
                                "sent B 0",
                                "sent B 1",
                                "sent B 2",
                            ],
                        },
                    ],
                    writer,
                )

            with temp_dir.joinpath("title_embedding_0.npy").open(
                "bw"
            ) as writer:
                np.save(writer, np.array([[0, 0, 1], [0, 0, -1]], dtype=float))

            with temp_dir.joinpath("sentence_embedding_0.npy").open(
                "bw"
            ) as writer:
                np.save(
                    writer,
                    np.array(
                        [
                            [1, 0, 0],
                            [0, 1, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, -1, 0],
                        ],
                        dtype=float,
                    ),
                )

            engine = Engine(Path(temp_dir))

            results = _round(
                engine.search(np.array([1, 0, 1], dtype=float), limit=1)
            )

            self.assertEqual(
                results,
                [
                    Result(
                        doc=Document(
                            pageid="0",
                            title="Person A",
                            sentences=["sent A 0", "sent A 1"],
                        ),
                        sentence=Sentence(text="sent A 0", doc_index=0),
                        score=1.41,
                    ),
                ],
            )

            results = _round(
                engine.search(np.array([0, -1, -1], dtype=float), limit=1)
            )

            self.assertEqual(
                results,
                [
                    Result(
                        doc=Document(
                            pageid="1",
                            title="Person B",
                            sentences=["sent B 0", "sent B 1", "sent B 2"],
                        ),
                        sentence=Sentence(text="sent B 2", doc_index=1),
                        score=1.41,
                    ),
                ],
            )


if __name__ == "__main__":

    unittest.main()
