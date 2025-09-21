import os
from typing import List, Optional, Union

from fastapi import FastAPI, Query
from fastembed import SparseTextEmbedding
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import NamedSparseVector
from sentence_transformers import SentenceTransformer

from src.llm import OllamaLLM
from src.utils import char_ngrams

from dotenv import load_dotenv
load_dotenv()

dense_model = SentenceTransformer("cointegrated/rubert-tiny2")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

qdrant_address = os.getenv("QDRANT_ADDRESS", "http://localhost:6333")
qdrant = QdrantClient(qdrant_address)
collection_name = os.getenv("COLLECTION")

llm = OllamaLLM(model_name=os.getenv("MODEL_NAME"))
print(llm)
app = FastAPI(title="Product Search API")


class SearchQuery(BaseModel):
    query: str
    top_k: int = 10


class Item(BaseModel):
    id: Optional[Union[str, int]] = None
    name: Optional[str] = None
    description: Optional[str] = None
    score: float


class LLMResponse(BaseModel):
    results: List[Item]
    answer: str


def search_items(product_name: str, top_k: int = 10):
    """Поиск по Qdrant с уже выделенным названием товара"""
    dense_vec = dense_model.encode([product_name]).tolist()[0]

    sparse_text = " ".join(char_ngrams(product_name))

    sparse_emb = list(sparse_model.embed([sparse_text]))[0]
    sparse_vec = models.SparseVector(
        indices=sparse_emb.indices, values=sparse_emb.values
    )

    prefetch = [
        models.Prefetch(query=dense_vec, using="dense", limit=top_k),
        models.Prefetch(query=sparse_vec, using="sparse", limit=top_k * 2),
    ]

    result = qdrant.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.DBSF),
        with_payload=True,
        limit=top_k,
    )

    items = [
        {
            "name": pt.payload.get("name"),
            "id": pt.payload.get("id"),
            "description": pt.payload.get("description", "нет описания"),
            "score": pt.score,
        }
        for pt in result.points
    ]

    return items


def search_sparse(product_name: str, top_k: int = 10):
    sparse_text = " ".join(char_ngrams(product_name.lower()))
    sparse_emb = list(sparse_model.embed([sparse_text]))[0]

    sparse_vec = NamedSparseVector(
        name="sparse",
        vector=models.SparseVector(
            indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist()
        ),
    )

    result = qdrant.search(
        collection_name=collection_name,
        query_vector=sparse_vec,
        limit=top_k,
        with_payload=True,
    )

    return [{"name": pt.payload.get("name"), "score": pt.score} for pt in result]


def search_dense(product_name: str, top_k: int = 10):
    dense_vec = dense_model.encode([product_name.lower()]).tolist()[0]

    result = qdrant.search(
        collection_name=collection_name,
        query_vector=models.NamedVector(name="dense", vector=dense_vec),
        limit=top_k,
        with_payload=True,
    )

    return [{"name": pt.payload.get("name"), "score": pt.score} for pt in result]


@app.post("/search_llm", response_model=LLMResponse)
def search(request: SearchQuery):
    items = [
        Item(
            id=pt.get("id"),
            name=pt.get("name"),
            description=pt.get("description"),
            score=pt.get("score"),
        )
        for pt in search_items(request.query, request.top_k)
    ]
    answer = llm.recommend_products(request.query, [i.dict() for i in items])
    return LLMResponse(results=items, answer=answer)


@app.get("/sparse", response_model=List[Item])
def sparse(query: str = Query(...), top_k: int = 10):
    items = [
        Item(name=pt.get("name"), score=pt.get("score"), description="нет описания")
        for pt in search_sparse(query, top_k)
    ]
    return items


@app.get("/dense", response_model=List[Item])
def dense(query: str = Query(...), top_k: int = 10):
    items = [
        Item(name=pt.get("name"), score=pt.get("score"), description="нет описания")
        for pt in search_dense(query, top_k)
    ]
    return items
