import json
import pandas as pd
import chromadb
import openai
import os
from dotenv import load_dotenv
from tqdm import tqdm

import streamlit as st


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
    # client = chromadb.Client()
    client = chromadb.PersistentClient(
        path="./chroma_db_for_agit_data"
    )  # 지정된 경로에 DB 생성

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
            ids=[item["parentID"]],
            documents=[item["contents"]],
            metadatas=[
                {
                    "id": item["parentID"],
                    "url": item["url"],
                    "answer_query": item["answer_query"],
                }
            ],
            embeddings=[embedding],
        )
        saved_data.append(
            {
                "id": item["parentID"],
                "document": item["contents"],
                "metadata": {
                    "id": item["parentID"],
                    "url": item["url"],
                    "answer_query": item["answer_query"],
                },
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

    # Perform semantic search
    semantic_results = collection.query(query_embeddings=[embedding], n_results=k)

    # Perform keyword search by checking if the user input is in the document
    keyword_results = [
        {
            "document": doc,  # Extract the first element from the list
            "metadata": metadata,  # Extract the first element from the list
            "score": 1.0,  # Assign a fixed score for keyword matches
        }
        for doc, metadata in zip(
            collection.get()["documents"][0], collection.get()["metadatas"][0]
        )
        if user_input.lower() in doc.lower()  # Access the first element for comparison
    ]

    # Combine results: prioritize semantic results, then add keyword results
    combined_results = []
    seen_ids = set()

    # Add semantic results first
    for doc, metadata in zip(
        semantic_results["documents"][0], semantic_results["metadatas"][0]
    ):
        combined_results.append({"document": doc, "metadata": metadata})
        seen_ids.add(metadata["id"])

    # Add keyword results if not already in semantic results
    for result in keyword_results:
        if result["metadata"]["id"] not in seen_ids:
            combined_results.append(result)

    return combined_results[:k]


# Step 4: Convert results to DataFrame
def results_to_dataframe(results):
    df = pd.DataFrame(results)
    return df


# Main function
def main():

    # Use session state to ensure this runs only once
    if "collection" not in st.session_state:
        # Load JSON data
        file_path = "preprocessed_bk_requests.json"
        data = load_json(file_path)

        # Prepare Chroma collection
        st.session_state.collection = prepare_chroma_collection(data)

    # Retrieve the collection from session state
    collection = st.session_state.collection

    # Streamlit app
    st.title("Hybrid Search Test")

    user_input = st.text_input("Enter your query:", "")
    # Get the number of documents in the collection
    collection_size = len(collection.get()["documents"])

    # Add a slider to select the number of results, limited by the collection size
    max_results = st.number_input(
        "Enter the top-k value to retrieve:",
        min_value=1,
        max_value=collection_size,
        value=min(5, collection_size),
        step=1,
    )

    # Validate the input and show an error message if out of range
    if max_results < 1 or max_results > collection_size:
        st.error(f"Please enter a value between 1 and {collection_size}.")
    else:
        k = int(max_results)

    if st.button("Search"):
        if user_input.strip():
            # Perform hybrid search
            results = hybrid_search(collection, user_input, k)

            # Convert results to DataFrame
            df = results_to_dataframe(results)

            # Display results
            st.write("## Search Results:")
            for index, row in df.iterrows():

                # Reorder metadata keys to ensure consistent order
                ordered_metadata = {
                    "id": row["metadata"]["id"],
                    "url": row["metadata"]["url"],
                    "answer_query": row["metadata"]["answer_query"],
                }

                document = row["document"]

                with st.expander(f"### Result {index + 1}", expanded=True):
                    st.write("##### Metadata:", ordered_metadata)
                    st.text("Document: \n\n" + str(document))
        else:
            st.warning("Please enter a query to search.")


if __name__ == "__main__":
    main()
