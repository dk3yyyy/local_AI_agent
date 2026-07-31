from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

project_root = Path(__file__).resolve().parent
df = pd.read_csv(project_root / "realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = project_root / "chrome_langchain_db"
documents = []
ids = []

for i, row in df.iterrows():
    document = Document(
        page_content=row["Title"] + " " + row["Review"],
        metadata={"rating": row["Rating"], "date": row["Date"]},
        id=str(i),
    )
    ids.append(str(i))
    documents.append(document)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=str(db_location),
    embedding_function=embeddings,
)

existing_ids = set(vector_store.get(ids=ids, include=[])["ids"])
missing_indexes = [
    index for index, item_id in enumerate(ids) if item_id not in existing_ids
]
missing_documents = [documents[index] for index in missing_indexes]
missing_ids = [ids[index] for index in missing_indexes]

if missing_documents:
    vector_store.add_documents(documents=missing_documents, ids=missing_ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
