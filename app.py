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
        data_dir=Path("/home/thomas/Downloads/people_pm_minilm"),
    )
    return engine


@st.cache(allow_output_mutation=True)
def load_model() -> SentenceTransformer:
    return load_minilm_model()


engine = load_engine()

model = load_model()

query = st.text_input("query", value="¿Dónde nació Leonardo Da Vinci?")

limit = st.slider("limit", min_value=1, max_value=50, value=5)


with st.spinner("Querying index ..."):

    embedding = model.encode([query], show_progress_bar=False)[0]
    results: List[Result] = engine.search(embedding, limit=limit)


results_by_doc = collections.defaultdict(list)
for r in results:
    results_by_doc[r.doc.pageid].append(r)

for rs in results_by_doc.values():
    rs.sort(key=lambda r: r.score, reverse=True)

result_groups = sorted(
    results_by_doc.values(), key=lambda rs: rs[0].score, reverse=True
)

for rs in result_groups:

    st.markdown(
        f"#### {rs[0].doc.title} ([link](http://es.wikipedia.org/?curid={rs[0].doc.pageid}))"
    )
    for r in rs:
        st.markdown(
            f'<p class="big-font">{r.sentence.text}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(f"(**score**: {round(100.0*r.score,2)})")


st.markdown(f"**Mem Usage**: {get_memory_usage()}MB")
