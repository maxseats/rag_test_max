import json
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Step 1: Load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Step 2: Prepare Chroma collection
def prepare_chroma_collection(
    data, collection_name="contents_collection", openai_model="text-embedding-ada-002"
):

    # Initialize Chroma client
    # client = chromadb.Client(
    #     Settings(persist_directory="./chroma_db", chroma_db_impl="duckdb+parquet")
    # )
    # client = chromadb.Client(
    #     Settings(persist_directory="./chroma_db", chroma_db_impl="duckdb+parquet")
    # )
    client = chromadb.Client()

    # Create or load collection
    collection = client.get_or_create_collection(name=collection_name)

    # Add data to the collection
    for i, item in enumerate(tqdm(data, desc="Processing data")):
        # Use OpenAI API to generate embeddings
        response = openai.Embedding.create(
            input=item["contents"], model=openai_model, api_key=OPENAI_API_KEY
        )
        embedding = response["data"][0]["embedding"]
        collection.add(
            ids=[str(i)],
            documents=[item["contents"]],
            metadatas=[{"id": i, "contents": item["contents"]}],
            embeddings=[embedding],
        )
    return collection


# Step 3: Hybrid search (semantic + keyword)
def hybrid_search(collection, user_input, k=5):

    openai_model = "text-embedding-ada-002"
    embedding = openai.Embedding.create(
        # input=item["contents"], model=openai_model, api_key=OPENAI_API_KEY
        input=user_input,
        model=openai_model,
        api_key=OPENAI_API_KEY,
    )

    # Semantic search
    # semantic_results = collection.query(query_texts=[user_input], n_results=k)
    semantic_results = collection.query(query_embeddings=[embedding], n_results=k)

    # Keyword search (simple filtering based on keyword presence)
    keyword_results = [
        {"content": doc["contents"], "score": 1.0}
        for doc in collection.get()["metadatas"]
        if user_input.lower() in doc["contents"].lower()
    ]

    # Combine results (prioritize semantic results, then add keyword results)
    combined_results = semantic_results["metadatas"]
    for keyword_result in keyword_results:
        if keyword_result not in combined_results:
            combined_results.append(keyword_result)

    return combined_results[:k]


# Step 4: Convert results to DataFrame
def results_to_dataframe(results):
    df = pd.DataFrame(results)
    return df


# Main function
def main():
    # Load JSON data
    file_path = (
        "/Users/kakaovx/Desktop/agent_test/rag_test_max/preprocessed_bk_requests.json"
    )
    data = load_json(file_path)

    # Prepare Chroma collection
    collection = prepare_chroma_collection(data)

    # User input
    user_input = input("Enter your query: ")

    # Perform hybrid search
    k = 5  # Number of top results to retrieve
    results = hybrid_search(collection, user_input, k)

    # Convert results to DataFrame
    df = results_to_dataframe(results)
    print(df)


if __name__ == "__main__":
    main()
