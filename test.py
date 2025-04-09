import json
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv
from tqdm import tqdm

import sys
import logging
import warnings
from datetime import datetime


# Load environment variables
load_dotenv(override=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Step 1: Load JSON data
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Step 2: Prepare Chroma collection and save/load locally
def prepare_chroma_collection(
    data,
    collection_name="contents_collection",
    openai_model="text-embedding-ada-002",
    save_path="chroma_collection.json",
):
    client = chromadb.Client()

    # Check if the collection file exists
    if os.path.exists(save_path):
        print("Loading collection from local file...")
        with open(save_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        collection = client.get_or_create_collection(name=collection_name)
        for item in saved_data:
            collection.add(
                ids=[item["id"]],
                documents=[item["document"]],
                metadatas=[item["metadata"]],
                embeddings=[item["embedding"]],
            )
        return collection

    print("Creating new collection...")
    # Create or load collection
    collection = client.get_or_create_collection(name=collection_name)

    # Add data to the collection
    saved_data = []
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
        saved_data.append(
            {
                "id": str(i),
                "document": item["contents"],
                "metadata": {"id": i, "contents": item["contents"]},
                "embedding": embedding,
            }
        )

    # Save the collection to a local file
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(saved_data, f, ensure_ascii=False, indent=4)

    return collection


# Step 3: Hybrid search (semantic + keyword)
def hybrid_search(collection, user_input, k=5):
    openai_model = "text-embedding-ada-002"

    embedding = openai.Embedding.create(
        input=user_input,
        model=openai_model,
        api_key=OPENAI_API_KEY,
    )["data"][0]["embedding"]

    # Semantic search
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

    # Save DataFrame to a CSV file
    dataframe_file = (
        "/Users/kakaovx/Desktop/agent_test/rag_test_max/results_dataframe.csv"
    )
    df.to_csv(dataframe_file, index=False, encoding="utf-8-sig")

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

    while True:
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
