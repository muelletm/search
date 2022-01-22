import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

from search.model import load_bepo_model, load_minilm_model
from search.utils import get_memory_usage


@st.cache(allow_output_mutation=True)
def load_model() -> SentenceTransformer:
    return load_bepo_model()


@st.cache(allow_output_mutation=True)
def load_other_model() -> SentenceTransformer:
    return load_minilm_model()


def run(text, model):

    sentences = text.strip().split("\n")

    embeddings = model.encode(sentences, show_progress_bar=False)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    st.write(embeddings.shape)

    dot_products = np.inner(embeddings, embeddings)

    rows = []

    for index, scores in enumerate(dot_products):

        rows.append(
            {
                "text": sentences[index],
                **{
                    sentences[other_index]: round(score.item(), 2)
                    for other_index, score in enumerate(scores)
                },
            }
        )

    df = pd.DataFrame(rows)
    df.set_index("text", inplace=True)
    st.dataframe(df)


model = load_model()

other_model = load_other_model()

text = st.text_area(
    "input",
    "María tiene un coche rápido.\n"
    "María tiene un coche bonito.\n"
    "María tiene una moto bonita.\n"
    "María tiene una moto rápida.\n"
    "María tiene un vehículo rápido.\n"
    "María tiene un vehículo bonito.\n",
)

run(text, model)

run(text, other_model)


st.markdown(f"**Mem Usage**: {get_memory_usage()}MB")
