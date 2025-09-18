from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, Union, List

from src.llm import OllamaLLM
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
import os

# --- модели ---
dense_model = SentenceTransformer("cointegrated/rubert-tiny2")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

# --- Qdrant client ---
qdrant_address = os.getenv("QDRANT_ADDRESS", "http://localhost:6333")
qdrant = QdrantClient(qdrant_address)
collection_name = "items"

# --- LLM ---
llm = OllamaLLM()  # объект Ollama

# --- FastAPI app ---
app = FastAPI(title="Product Search API")


class SearchResponse(BaseModel):
    id: Optional[Union[str, int]] = None
    name: Optional[str] = None
    score: float


class SearchResult(BaseModel):
    results: List[SearchResponse]
    answer: str


def search_items(product_name: str, top_k: int = 10):
    """Поиск по Qdrant с уже выделенным названием товара"""
    dense_vec = dense_model.encode([product_name]).tolist()[0]
    sparse_emb = list(sparse_model.embed([product_name]))[0]
    sparse_vec = models.SparseVector(indices=sparse_emb.indices, values=sparse_emb.values)

    prefetch = [
        models.Prefetch(query=dense_vec, using="dense", limit=1),
        models.Prefetch(query=sparse_vec, using="sparse", limit=10),
    ]

    result = qdrant.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        with_payload=True,
        limit=top_k
    )

    items = [
        {"name": pt.payload.get("name"), "score": pt.score}
        for pt in result.points
    ]

    return items


@app.get("/search", response_model=SearchResult)
def search(query: str = Query(...), top_k: int = 5):
    product_name = llm.extract_product_name(query)
    print("Название товара:", product_name)

    items = search_items(product_name, top_k)
    print("Найденные товары:", items)

    answer = llm.recommend_products(product_name, items)
    print("Рекомендации модели:", answer)

    return {"results": items, "answer": answer}


