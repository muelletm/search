from typing import Optional

from sentence_transformers import SentenceTransformer, models


def load_bepo_model(
    max_seq_length: Optional[int] = None,
) -> SentenceTransformer:
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    word_embedding_model = models.Transformer(
        model_name, max_seq_length=max_seq_length
    )
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension()
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


def load_minilm_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def load_mpnet_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")


def load_distiluse_model() -> SentenceTransformer:
    return SentenceTransformer("distiluse-base-multilingual-cased-v2")
