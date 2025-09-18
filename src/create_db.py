import json
import os
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

path = "/src/test_with_description.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[:1000]
print(data[0])
sparse_texts = [item["name"] for item in data if "name" in item and item["name"]]

dense_texts = [
    f"{item.get('name', '')}. {item.get('description', '')}"
    for item in data
]


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
        "sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF
        )
    },
)


batch_size = 10
total = len(data)


for i in tqdm(range(0, total, batch_size), total=(total + batch_size - 1) // batch_size, desc="Uploading"):
    batch_payloads = data[i:i + batch_size]

    # dense embeddings по json
    batch_dense = dense_model.encode(
        dense_texts[i:i + batch_size],
        show_progress_bar=False
    ).tolist()

    # sparse embeddings по name
    sparse_embeddings = list(
        sparse_model.embed(sparse_texts[i:i + batch_size])
    )

    batch_sparse = [
        models.SparseVector(indices=emb.indices, values=emb.values)
        for emb in sparse_embeddings
    ]

    points = [
        models.PointStruct(
            id=i + idx,
            vector={
                "dense": batch_dense[idx],
                "sparse": batch_sparse[idx]
            },
            payload=batch_payloads[idx]
        )
        for idx in range(len(batch_payloads))
    ]

    qdrant.upload_points(
        collection_name=collection_name,
        points=points,
        batch_size=batch_size
    )




