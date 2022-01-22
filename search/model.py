from sentence_transformers import SentenceTransformer, models


def load_bepo_model() -> SentenceTransformer:
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    word_embedding_model = models.Transformer(model_name, max_seq_length=16)
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension()
    )
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


def load_minilm_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
