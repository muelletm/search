from pathlib import Path

import pandas as pd

from search.evaluation.sts import StsEvaluator
from search.evaluation.xnli import XnliEvaluator
from search.model import (
    load_bepo_model,
    load_distiluse_model,
    load_minilm_model,
    load_mpnet_model,
)


def main():

    models = {
        "bepo": load_bepo_model,
        "minilm": load_minilm_model,
        "mpnet": load_mpnet_model,
        "distiluse": load_distiluse_model,
    }

    for dataset in ["xnli", "sts2015"]:

        if dataset == "xnli":
            evaluator = XnliEvaluator(Path("/home/thomas/Downloads/XNLI-1.0"))
        elif dataset == "sts2015":
            evaluator = StsEvaluator(Path("/home/thomas/Downloads/STS2015-es"))

        rows = []
        for model_name, model_fn in models.items():
            evaluator.evaluate(model_fn, model_name, rows)

        print(dataset)
        print(pd.DataFrame(rows).set_index("name"))
        print()


if __name__ == "__main__":
    main()
