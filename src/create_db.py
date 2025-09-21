import json
import os

from fastembed import SparseTextEmbedding
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.utils import char_ngrams

path = "data/parsed_data.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)


data = [item for item in data if "name" in item and item["name"]]
data = [{k.lower(): str(v).lower() for k, v in item.items()} for item in data]
print(data[0])


DENSE_ATTRS = ["name", "бренд", "коллекция", "цвет", "материал"]


qdrant_address = os.getenv("QDRANT_ADDRESS", "http://localhost:6333")
qdrant = QdrantClient(qdrant_address)

collection_name = "items"

dense_model = SentenceTransformer("cointegrated/rubert-tiny2")
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

print(dense_model)
print(sparse_model)

qdrant.recreate_collection(
    collection_name=collection_name,
    vectors_config={
        "dense": models.VectorParams(
            size=dense_model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        )
    },
    sparse_vectors_config={
        "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
    },
)
batch_size = 1000
total = len(data)

for i in tqdm(
    range(0, total, batch_size),
    total=(total + batch_size - 1) // batch_size,
    desc="Uploading",
):
    batch_payloads = data[i : i + batch_size]

    # sparse текст для текущего батча (каждая строка = один товар)
    sparse_texts = [" ".join(char_ngrams(item["name"])) for item in batch_payloads]

    # dense текст для текущего батча
    batch_dense_texts = [
        " ".join(
            str(v).lower()
            for k, v in item.items()
            if any(attr in k.lower() for attr in DENSE_ATTRS)
        )
        for item in batch_payloads
    ]

    # эмбеддинги dense
    batch_dense = dense_model.encode(
        batch_dense_texts, show_progress_bar=False
    ).tolist()

    # эмбеддинги sparse
    sparse_embeddings = list(sparse_model.embed(sparse_texts))

    batch_sparse = [
        models.SparseVector(indices=emb.indices, values=emb.values)
        for emb in sparse_embeddings
    ]

    points = [
        models.PointStruct(
            id=i + idx,
            vector={"dense": batch_dense[idx], "sparse": batch_sparse[idx]},
            payload=batch_payloads[idx],
        )
        for idx in range(len(batch_payloads))
    ]

    # загружаем в Qdrant
    qdrant.upload_points(
        collection_name=collection_name, points=points, batch_size=batch_size
    )
    for item in batch_payloads:
        logger.info(f"Товар ID {item['id']} загружен ✅")
