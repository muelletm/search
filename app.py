import collections
import os
from pathlib import Path
from typing import List

import streamlit as st
from sentence_transformers import SentenceTransformer

from search.engine import Engine, Result
from search.model import load_minilm_model
from search.utils import get_memory_usage

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_DATA_DIR = os.environ.get("DATA_DIR", "data/people_pm_minilm")

st.set_page_config(page_title="Search Engine", layout="wide")

st.markdown(
    """
<style>
.big-font {
    font-size:20px;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache(allow_output_mutation=True)
def load_engine() -> Engine:
    engine = Engine(
        data_dir=Path(_DATA_DIR),
    )
    return engine


@st.cache(allow_output_mutation=True)
def load_model() -> SentenceTransformer:
    return load_minilm_model()


engine = load_engine()

model = load_model()

st.error("Create a text input for the query.")

st.error("Create a slider with the number of results to retrieve.")

with st.spinner("Querying index ..."):
    st.error("Get query embedding.")
    st.error("Search results (engine.search).")

# Show the results.
# You can use st.markdown to render markdown.
# e.g. st.markdown("**text**") will add text in bold font.

st.error("Render results")
st.markdown(f"**Mem Usage**: {get_memory_usage()}MB")
