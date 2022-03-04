from sentence_transformers import SentenceTransformer


def load_minilm_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
