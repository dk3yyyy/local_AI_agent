from pathlib import Path

import pandas as pd
import streamlit as st
from httpx import HTTPError
from ollama import ResponseError

from agent import answer_question, create_chat_model
from dashboard_support import DatasetSelection, prepare_uploaded_dataset
from ollama_health import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_HOST,
    check_ollama,
)
from vector import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DATA_PATH,
    DEFAULT_DATABASE_PATH,
    ReviewDataError,
    create_vector_store,
    dataset_summary,
    filter_reviews,
    index_count,
    load_reviews,
)

st.set_page_config(
    page_title="Restaurant Review Intelligence",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #18221d;
        --forest: #174f3a;
        --mint: #dff4e8;
        --coral: #ef6a4b;
        --paper: #fbfaf6;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #f0f5ef; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #4e6258; }
    [data-testid="stChatInput"] textarea:disabled {
        color: #4e6258;
        -webkit-text-fill-color: #4e6258;
        opacity: 1;
    }
    .hero {
        padding: 1.8rem 2rem;
        border-radius: 1.25rem;
        background: linear-gradient(120deg, #153f32 0%, #236a4e 62%, #d96b4e 160%);
        color: white;
        margin-bottom: 1.25rem;
    }
    .hero-kicker {
        text-transform: uppercase;
        letter-spacing: .14em;
        font-size: .72rem;
        font-weight: 700;
        opacity: .78;
    }
    .hero h1 { margin: .3rem 0 .45rem; font-size: clamp(2rem, 5vw, 3.5rem); }
    .hero p { margin: 0; max-width: 48rem; font-size: 1.02rem; opacity: .88; }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce7df;
        padding: .9rem 1rem;
        border-radius: .9rem;
        box-shadow: 0 8px 24px rgba(18, 62, 44, .05);
    }
    div[data-testid="stChatMessage"] {
        border: 1px solid #dce7df;
        border-radius: 1rem;
        background: rgba(255,255,255,.78);
    }
    .source-label { color: var(--forest); font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=10, show_spinner=False)
def cached_health(host: str, chat_model: str, embedding_model: str):
    return check_ollama(
        host=host,
        required_models=(chat_model, embedding_model),
    )


@st.cache_resource(show_spinner=False)
def cached_vector_store(
    csv_path: str,
    database_path: str,
    collection_name: str,
    embedding_model: str,
    ollama_host: str,
):
    return create_vector_store(
        Path(csv_path),
        database_path=Path(database_path),
        collection_name=collection_name,
        embedding_model=embedding_model,
        ollama_host=ollama_host,
    )


def default_selection() -> DatasetSelection:
    dataframe = load_reviews(DEFAULT_DATA_PATH)
    return DatasetSelection(
        csv_path=DEFAULT_DATA_PATH,
        database_path=DEFAULT_DATABASE_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
        digest="bundled-dataset",
        review_count=len(dataframe),
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("#### Evidence")
    for source in sources:
        label = (
            f"[{source['citation_number']}] {source['rating']}/5"
            f" · {source['date']} · relevance distance {source['score']:.3f}"
        )
        with st.expander(label):
            st.write(source["content"])


def serialize_sources(result) -> list[dict]:
    return [
        {
            "citation_number": source.citation_number,
            "rating": source.document.metadata.get("rating", "?"),
            "date": source.document.metadata.get("date", "unknown"),
            "content": source.document.page_content,
            "score": source.score,
        }
        for source in result.sources
    ]


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Local review intelligence</div>
      <h1>Ask the reviews, not a black box.</h1>
      <p>Explore ratings and dates, then ask grounded questions. Every answer links back to the reviews that support it.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Workspace")
    uploaded_file = st.file_uploader(
        "Review dataset",
        type=("csv",),
        help="Required columns: Title, Date, Rating, Review",
    )
    selection = default_selection()
    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        if len(upload_bytes) > 10 * 1024 * 1024:
            st.error("The uploaded CSV must be 10 MB or smaller.")
            st.stop()
        try:
            selection = prepare_uploaded_dataset(upload_bytes)
        except ReviewDataError as error:
            st.error(str(error))
            st.stop()
        st.success(f"Validated {selection.review_count} uploaded reviews")
    else:
        st.caption("Using the bundled 123-review dataset")

    st.divider()
    st.subheader("Local models")
    ollama_host = st.text_input("Ollama host", value=DEFAULT_OLLAMA_HOST)
    chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
    embedding_model = st.text_input("Embedding model", value=DEFAULT_EMBEDDING_MODEL)
    health = cached_health(ollama_host, chat_model, embedding_model)
    if health.ok:
        st.success("Ollama and both models are ready")
    elif not health.service_available:
        st.error("Ollama is not reachable")
        st.code("ollama serve")
    else:
        st.warning("One or more models are missing")
        for model_name in health.missing_models:
            st.code(f"ollama pull {model_name}")
    if health.error:
        st.caption(health.error)

try:
    dataframe = load_reviews(selection.csv_path)
except ReviewDataError as error:
    st.error(str(error))
    st.stop()

summary = dataset_summary(dataframe)

with st.sidebar:
    st.divider()
    st.subheader("Review filters")
    rating_range = st.slider(
        "Rating range",
        min_value=1,
        max_value=5,
        value=(1, 5),
    )
    selected_dates = st.date_input(
        "Date range",
        value=(summary.first_date, summary.last_date),
        min_value=summary.first_date,
        max_value=summary.last_date,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date = end_date = selected_dates  # type: ignore[assignment]
    retrieval_limit = st.slider("Evidence per answer", 1, 10, 5)

filtered = filter_reviews(
    dataframe,
    min_rating=rating_range[0],
    max_rating=rating_range[1],
    start_date=start_date,
    end_date=end_date,
)

metric_columns = st.columns(4)
metric_columns[0].metric("Reviews in view", f"{len(filtered):,}")
metric_columns[1].metric(
    "Average rating",
    f"{filtered['Rating'].mean():.2f}" if len(filtered) else "No data",
)
metric_columns[2].metric("Positive", int((filtered["Rating"] >= 4).sum()))
metric_columns[3].metric("Low-rated", int((filtered["Rating"] <= 2).sum()))

analytics_column, sample_column = st.columns((1, 1.35), gap="large")
with analytics_column:
    st.subheader("Rating distribution")
    rating_counts = (
        filtered["Rating"]
        .value_counts()
        .reindex(range(1, 6), fill_value=0)
        .sort_index()
    )
    chart_data = pd.DataFrame(
        {
            "Rating": [f"{rating} star" for rating in rating_counts.index],
            "Reviews": rating_counts.values,
        }
    ).set_index("Rating")
    st.bar_chart(
        chart_data,
        color="#236a4e",
        horizontal=True,
        height=285,
    )

with sample_column:
    st.subheader("Reviews in the current view")
    if filtered.empty:
        st.info("No reviews match these filters.")
    else:
        preview = filtered[["Date", "Rating", "Title", "Review"]].sort_values(
            "Date", ascending=False
        )
        st.dataframe(
            preview,
            hide_index=True,
            width="stretch",
            height=285,
            column_config={
                "Review": st.column_config.TextColumn(width="large"),
                "Rating": st.column_config.NumberColumn(format="%d ★"),
            },
        )

st.divider()
st.subheader("Ask the review evidence")
st.caption(
    "Answers are generated locally and constrained to the reviews selected by the current rating and date filters."
)
if not health.ok:
    st.warning(
        "Chat is disabled until Ollama is running and both required models are available."
    )

history_key = f"messages_{selection.digest}"
if history_key not in st.session_state:
    st.session_state[history_key] = []

for message in st.session_state[history_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

question = st.chat_input(
    "What do guests say about the crust, service, or delivery?",
    disabled=not health.ok or filtered.empty,
)

if question:
    st.session_state[history_key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Indexing missing reviews and gathering evidence..."):
                vector_store = cached_vector_store(
                    str(selection.csv_path),
                    str(selection.database_path),
                    selection.collection_name,
                    embedding_model,
                    ollama_host,
                )
                model = create_chat_model(model=chat_model, base_url=ollama_host)
                result = answer_question(
                    question,
                    vector_store=vector_store,
                    model=model,
                    limit=retrieval_limit,
                    min_rating=rating_range[0],
                    max_rating=rating_range[1],
                    start_date=start_date,
                    end_date=end_date,
                )
            st.markdown(result.answer)
            serialized_sources = serialize_sources(result)
            render_sources(serialized_sources)
            st.session_state[history_key].append(
                {
                    "role": "assistant",
                    "content": result.answer,
                    "sources": serialized_sources,
                }
            )
            st.session_state[f"index_count_{selection.digest}"] = index_count(
                vector_store
            )
        except (
            ConnectionError,
            HTTPError,
            OSError,
            ResponseError,
            ReviewDataError,
            RuntimeError,
            ValueError,
        ) as error:
            message = f"The local answer could not be generated: {error}"
            st.error(message)
            st.session_state[history_key].append(
                {"role": "assistant", "content": message, "sources": []}
            )

with st.sidebar:
    st.divider()
    st.subheader("Index status")
    current_index_count = st.session_state.get(f"index_count_{selection.digest}")
    if current_index_count is None:
        st.caption("The index initializes when you ask the first question.")
    else:
        st.success(f"{current_index_count} reviews indexed")
    if st.button("Clear conversation", width="stretch"):
        st.session_state[history_key] = []
        st.rerun()
