from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
import uuid

# In-memory storage for info
info_storage = {}

# Initialize Qdrant client
def initialize_qdrant():
    qdrant = QdrantClient(host="localhost", port=6333)
    collection_name = "jarvis_assistant"

    # Get the list of existing collections
    existing_collections = qdrant.get_collections().collections
    collection_names = [collection.name for collection in existing_collections]

    if collection_name not in collection_names:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    return qdrant

# Store embeddings in Qdrant
def store_embeddings(qdrant, text, embeddings):
    qdrant.upsert(
        collection_name="jarvis_assistant",
        points=[{"id": 1, "vector": embeddings, "payload": {"text": text}}]
    )

# Search for similar embeddings in Qdrant
def search_embeddings(qdrant, query_vector):
    search_result = qdrant.search(
        collection_name="jarvis_assistant",
        query_vector=query_vector,
        limit=1
    )
    return search_result

# Store headers in Qdrant
def store_headers_in_qdrant(qdrant, header, unique_id, embedding, info):
    qdrant.upsert(
        collection_name="jarvis_assistant",
        points=[
            {
                "id": unique_id,
                "vector": embedding,
                "payload": {"header": header, "info": info}
            }
        ]
    )

# Search for similar headers in Qdrant
def search_headers_in_qdrant(qdrant, query_vector):
    search_result = qdrant.search(
        collection_name="jarvis_assistant",
        query_vector=query_vector,
        limit=1
    )
    return search_result

# Function to get info by ID
def get_info_by_id(unique_id):
    return info_storage.get(unique_id, "")

# Function to update info by ID
def update_info_in_qdrant(qdrant, point_id, new_info):
    qdrant.set_payload(
        collection_name="jarvis_assistant",
        payload={"info": new_info},
        points=[point_id]
    )

# Function to update info by ID
def update_info_by_id(unique_id, new_info):
    info_storage[unique_id] = new_info