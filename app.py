import os
import pandas as pd
import re
import requests
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from datasets import load_dataset

# Function to download the dataset from GitHub
@st.cache_resource
def download_dataset_from_github():
    github_url = "https://raw.githubusercontent.com/Mansur-Mahdee/Flower_power/refs/heads/main/data/language-of-flowers.csv"
    dataset_path = "/tmp/language-of-flowers.csv"
    
    if os.path.exists(dataset_path):
        st.write("Dataset already exists, skipping download.")
        return dataset_path

    try:
        response = requests.get(github_url)
        response.raise_for_status()
        with open(dataset_path, 'wb') as f:
            f.write(response.content)
        st.write(f"Downloaded dataset from GitHub to {dataset_path}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error during dataset download: {e}")
        return None
    
    return dataset_path

# Function to generate flower information
def generate_flower_info(flower_name, flower_info_dict, gpt2_pipeline):
    flower_description = flower_info_dict.get(flower_name, "No description available.")
    query = f"Why is the flower {flower_name} associated with the meaning '{flower_description}'? Explain the cultural or historical significance behind the {flower_name}."
    generated_info = gpt2_pipeline(query, max_length=200, truncation=True)[0]["generated_text"]
    sentences = re.split(r'(?<=\w[.!?])\s+', generated_info.strip())
    limited_output = "".join(sentences[:5])
    return flower_name, flower_description, limited_output

# Main app code
def streamlit_app():
    st.title("Language of Flowers")
    st.write("Welcome to the Language of Flowers App! Here you can learn about the meanings and cultural significance of different flowers.")

    # Download and load dataset
    dataset_path = download_dataset_from_github()
    if dataset_path:
        data = pd.read_csv(dataset_path, quotechar='"', encoding='utf-8-sig', on_bad_lines='skip')
        data.columns = data.columns.str.strip()  # Clean column names
        flower_info_dict = dict(zip(data['Flower'], data['Meaning']))
        st.write("Dataset loaded successfully.")

        # Initialize GPT-2 and Rag models for text generation
        gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        gpt2_pipeline = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer)

        # Initialize RAG models for nearest flower retrieval
        dataset = load_dataset("wiki_dpr", "psgs_w100.nq.exact", trust_remote_code=True)
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        retriever = RagRetriever.from_pretrained(
            "facebook/rag-token-nq", 
            index_name="exact", 
            use_dummy_dataset=True, 
            trust_remote_code=True  # Add trust_remote_code=True to the retriever
        )
        model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
        tokenizer.pad_token_id = 0
        rag_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

        # User input to get flower name
        flower_name = st.text_input("Enter a flower name:", "")
        if flower_name:
            flower_name = flower_name.strip().capitalize()  # Capitalize to match the dataset

            # If the flower is not in the dataset, use RAG to find the closest match
            if flower_name not in flower_info_dict:
                st.write("Exact match not found. Searching for the closest match...")
                rag_input = f"Find the closest flower name for: {flower_name}"
                # Use the RAG model to find the closest match
                rag_output = rag_pipeline(rag_input, max_length=100, truncation=True)[0]["generated_text"]
                st.write(f"Closest match found: {rag_output}")
                flower_name = rag_output.strip()  # Update flower_name with RAG output

            # Check if the flower name is now in the dictionary
            if flower_name in flower_info_dict:
                flower_name, flower_description, generated_info = generate_flower_info(flower_name, flower_info_dict, gpt2_pipeline)
                st.write(f"### Information for {flower_name}:")
                st.write(f"**Meaning**: {flower_description}")
                st.write(f"**Cultural or Historical Significance**: {generated_info}")
            else:
                st.write(f"Sorry, we still couldn't find information for the flower: {flower_name}")
    else:
        st.write("There was an issue with downloading or loading the dataset.")

if __name__ == "__main__":
    streamlit_app()
