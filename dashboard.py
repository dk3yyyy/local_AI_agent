from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st
from httpx import HTTPError
from ollama import ResponseError

from agent import answer_question, create_chat_model
from dashboard_support import (
    DatasetSelection,
    prepare_uploaded_dataset,
    read_csv_columns,
)
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
    ColumnMapping,
    ReviewDataError,
    create_vector_store,
    dataset_summary,
    filter_reviews,
    index_count,
    load_reviews,
    suggest_column_mapping,
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
    mapping: ColumnMapping,
    embedding_model: str,
    ollama_host: str,
):
    return create_vector_store(
        Path(csv_path),
        mapping=mapping,
        database_path=Path(database_path),
        collection_name=collection_name,
        embedding_model=embedding_model,
        ollama_host=ollama_host,
    )


def default_selection() -> DatasetSelection:
    mapping = ColumnMapping(
        review="Review",
        title="Title",
        date="Date",
        rating="Rating",
    )
    dataframe = load_reviews(DEFAULT_DATA_PATH, mapping=mapping)
    return DatasetSelection(
        csv_path=DEFAULT_DATA_PATH,
        database_path=DEFAULT_DATABASE_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
        digest="bundled-dataset",
        review_count=len(dataframe),
        mapping=mapping,
    )


def _optional_mapping_select(
    label: str,
    role: str,
    columns: tuple[str, ...],
    suggestions: dict[str, str | None],
    key_prefix: str,
) -> str | None:
    options = ("Not mapped", *columns)
    suggestion = suggestions[role]
    index = options.index(suggestion) if suggestion in options else 0
    selected = st.selectbox(label, options, index=index, key=f"{key_prefix}_{role}")
    return None if selected == "Not mapped" else selected


def mapping_controls(content: bytes) -> ColumnMapping | None:
    columns = read_csv_columns(content)
    suggestions = suggest_column_mapping(columns)
    key_prefix = sha256(content).hexdigest()[:12]
    with st.expander("Column mapping", expanded=True):
        st.caption("Review text is required. Every other role is optional.")
        review_suggestion = suggestions["review"]
        review_index = (
            columns.index(review_suggestion) if review_suggestion in columns else None
        )
        review_column = st.selectbox(
            "Review text",
            columns,
            index=review_index,
            placeholder="Select the column containing review text",
            key=f"{key_prefix}_review",
        )
        title_column = _optional_mapping_select(
            "Review title", "title", columns, suggestions, key_prefix
        )
        date_column = _optional_mapping_select(
            "Review date", "date", columns, suggestions, key_prefix
        )
        rating_column = _optional_mapping_select(
            "Rating or stars", "rating", columns, suggestions, key_prefix
        )
        sentiment_column = _optional_mapping_select(
            "Sentiment", "sentiment", columns, suggestions, key_prefix
        )
        restaurant_column = _optional_mapping_select(
            "Restaurant", "restaurant", columns, suggestions, key_prefix
        )
        country_column = _optional_mapping_select(
            "Country or region", "country", columns, suggestions, key_prefix
        )
    if review_column is None:
        return None
    return ColumnMapping(
        review=review_column,
        title=title_column,
        date=date_column,
        rating=rating_column,
        sentiment=sentiment_column,
        restaurant=restaurant_column,
        country=country_column,
    )


def _source_label(source: dict) -> str:
    details = [f"[{source['citation_number']}]"]
    for key in ("restaurant", "country", "sentiment"):
        value = source.get(key)
        if value is not None:
            details.append(str(value))
    if source.get("rating") is not None:
        details.append(f"{source['rating']}/5")
    if source.get("date") is not None:
        details.append(str(source["date"]))
    details.append(f"distance {source['score']:.3f}")
    return " · ".join(details)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("#### Evidence")
    for source in sources:
        with st.expander(_source_label(source)):
            st.write(source["content"])


def serialize_sources(result) -> list[dict]:
    return [
        {
            "citation_number": source.citation_number,
            "rating": source.document.metadata.get("rating"),
            "date": source.document.metadata.get("date"),
            "sentiment": source.document.metadata.get("sentiment"),
            "restaurant": source.document.metadata.get("restaurant"),
            "country": source.document.metadata.get("country"),
            "content": source.document.page_content,
            "score": source.score,
        }
        for source in result.sources
    ]


def _positive_sentiment_count(dataframe: pd.DataFrame) -> int:
    positive_labels = {"positive", "pos", "favorable", "favourable", "happy"}
    return int(
        dataframe["Sentiment"]
        .dropna()
        .astype(str)
        .str.casefold()
        .isin(positive_labels)
        .sum()
    )


def _negative_sentiment_count(dataframe: pd.DataFrame) -> int:
    negative_labels = {"negative", "neg", "unfavorable", "unfavourable", "unhappy"}
    return int(
        dataframe["Sentiment"]
        .dropna()
        .astype(str)
        .str.casefold()
        .isin(negative_labels)
        .sum()
    )


st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Adaptive review intelligence</div>
      <h1>Bring your columns. Keep the evidence.</h1>
      <p>Map almost any review CSV, explore the fields it actually contains, and ask grounded questions with source records attached.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Workspace")
    uploaded_file = st.file_uploader(
        "Review dataset",
        type=("csv",),
        help="Only a review-text column is required. Map optional fields after upload.",
    )
    selection = default_selection()
    if uploaded_file is not None:
        upload_bytes = uploaded_file.getvalue()
        if len(upload_bytes) > 10 * 1024 * 1024:
            st.error("The uploaded CSV must be 10 MB or smaller.")
            st.stop()
        try:
            uploaded_mapping = mapping_controls(upload_bytes)
            if uploaded_mapping is None:
                st.info("Select the column containing the review text to continue.")
                st.stop()
            selection = prepare_uploaded_dataset(
                upload_bytes,
                mapping=uploaded_mapping,
            )
        except (ReviewDataError, ValueError) as error:
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
    dataframe = load_reviews(selection.csv_path, mapping=selection.mapping)
except ReviewDataError as error:
    st.error(str(error))
    st.stop()

summary = dataset_summary(dataframe)
has_rating = dataframe["Rating"].notna().any()
has_date = dataframe["Date"].notna().any()
has_sentiment = dataframe["Sentiment"].notna().any()
has_restaurant = dataframe["Restaurant"].notna().any()
has_country = dataframe["Country"].notna().any()

with st.sidebar:
    st.divider()
    st.subheader("Review filters")
    min_rating = max_rating = None
    if has_rating:
        rating_range = st.slider("Rating range", 1, 5, (1, 5))
        min_rating, max_rating = rating_range

    start_date = end_date = None
    if has_date and summary.first_date and summary.last_date:
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

    sentiment_filter: list[str] = []
    if has_sentiment:
        sentiment_filter = st.multiselect(
            "Sentiment",
            sorted(dataframe["Sentiment"].dropna().astype(str).unique()),
        )
    restaurant_filter: list[str] = []
    if has_restaurant:
        restaurant_filter = st.multiselect(
            "Restaurant",
            sorted(dataframe["Restaurant"].dropna().astype(str).unique()),
        )
    country_filter: list[str] = []
    if has_country:
        country_filter = st.multiselect(
            "Country or region",
            sorted(dataframe["Country"].dropna().astype(str).unique()),
        )
    retrieval_limit = st.slider("Evidence per answer", 1, 10, 5)

filtered = filter_reviews(
    dataframe,
    min_rating=min_rating,
    max_rating=max_rating,
    start_date=start_date,
    end_date=end_date,
    sentiments=sentiment_filter,
    restaurants=restaurant_filter,
    countries=country_filter,
)

metric_columns = st.columns(4)
metric_columns[0].metric("Reviews in view", f"{len(filtered):,}")
if has_rating:
    average_rating = filtered["Rating"].mean()
    metric_columns[1].metric(
        "Average rating",
        f"{average_rating:.2f}" if pd.notna(average_rating) else "No data",
    )
    metric_columns[2].metric("Positive", int((filtered["Rating"] >= 4).sum()))
    metric_columns[3].metric("Low-rated", int((filtered["Rating"] <= 2).sum()))
elif has_sentiment:
    metric_columns[1].metric("Sentiment labels", filtered["Sentiment"].nunique())
    metric_columns[2].metric("Positive", _positive_sentiment_count(filtered))
    metric_columns[3].metric("Negative", _negative_sentiment_count(filtered))
elif has_restaurant:
    metric_columns[1].metric("Restaurants", filtered["Restaurant"].nunique())
    metric_columns[2].metric("Countries", filtered["Country"].nunique())
    metric_columns[3].metric("Dated records", int(filtered["Date"].notna().sum()))
else:
    metric_columns[1].metric("Dated records", int(filtered["Date"].notna().sum()))
    metric_columns[2].metric("With titles", int(filtered["Title"].notna().sum()))
    metric_columns[3].metric(
        "Extra fields", int(sum(bool(item) for item in filtered["_extra"]))
    )

analytics_column, sample_column = st.columns((1, 1.35), gap="large")
with analytics_column:
    if has_rating:
        chart_title = "Rating distribution"
        counts = (
            filtered["Rating"]
            .value_counts()
            .reindex(range(1, 6), fill_value=0)
            .sort_index()
        )
        labels = [f"{rating} star" for rating in counts.index]
    elif has_sentiment:
        chart_title = "Sentiment distribution"
        counts = filtered["Sentiment"].value_counts().sort_values(ascending=True)
        labels = [str(value) for value in counts.index]
    elif has_restaurant:
        chart_title = "Reviews by restaurant"
        counts = filtered["Restaurant"].value_counts().head(10).sort_values()
        labels = [str(value) for value in counts.index]
    elif has_country:
        chart_title = "Reviews by country"
        counts = filtered["Country"].value_counts().head(10).sort_values()
        labels = [str(value) for value in counts.index]
    else:
        chart_title = "Dataset composition"
        counts = pd.Series([len(filtered)], index=["Reviews"])
        labels = ["Reviews"]
    st.subheader(chart_title)
    chart_data = pd.DataFrame({"Category": labels, "Reviews": counts.values}).set_index(
        "Category"
    )
    st.bar_chart(chart_data, color="#236a4e", horizontal=True, height=285)

with sample_column:
    st.subheader("Reviews in the current view")
    if filtered.empty:
        st.info("No reviews match these filters.")
    else:
        candidate_columns = (
            "Date",
            "Rating",
            "Sentiment",
            "Restaurant",
            "Country",
            "Title",
            "Review",
        )
        preview_columns = [
            column
            for column in candidate_columns
            if column == "Review" or filtered[column].notna().any()
        ]
        preview = filtered[preview_columns]
        if "Date" in preview:
            preview = preview.sort_values("Date", ascending=False)
        column_config = {"Review": st.column_config.TextColumn(width="large")}
        if "Rating" in preview:
            column_config["Rating"] = st.column_config.NumberColumn(format="%d ★")
        st.dataframe(
            preview,
            hide_index=True,
            width="stretch",
            height=285,
            column_config=column_config,
        )

st.divider()
st.subheader("Ask the review evidence")
st.caption(
    "Answers use only retrieved records and respect every active filter supported by this dataset."
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
    "What patterns or complaints appear in these reviews?",
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
                    selection.mapping,
                    embedding_model,
                    ollama_host,
                )
                model = create_chat_model(model=chat_model, base_url=ollama_host)
                result = answer_question(
                    question,
                    vector_store=vector_store,
                    model=model,
                    limit=retrieval_limit,
                    min_rating=min_rating,
                    max_rating=max_rating,
                    start_date=start_date,
                    end_date=end_date,
                    sentiments=sentiment_filter,
                    restaurants=restaurant_filter,
                    countries=country_filter,
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
