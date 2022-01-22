import os
from pathlib import Path
from typing import List

import streamlit as st
from sentence_transformers import SentenceTransformer

from search.engine import Engine, Result
from search.utils import get_memory_usage

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Search Engine", layout="wide")


@st.cache(allow_output_mutation=True)
def load_engine() -> Engine:
    engine = Engine(
        data_dir=Path("/home/thomas/Downloads/wikicorpus_pm_minilm"),
    )
    return engine


@st.cache(allow_output_mutation=True)
def load_model() -> SentenceTransformer:
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


engine = load_engine()

model = load_model()

query = st.text_input("query", value="¿Dónde nació Leonardo Da Vinci?")

limit = st.slider("limit", min_value=1, max_value=50, value=5)


with st.spinner("Querying index ..."):

    embedding = model.encode([query])[0]
    results: List[Result] = engine.search(embedding, limit=limit)


for r in results:

    st.markdown(f"### {r.doc.title}")
    st.markdown(f"**paragraph**: {r.paragraph.text}")
    st.markdown(f"**score**: {r.score}")



st.markdown(f"**Mem Usage**: {get_memory_usage()}MB")
