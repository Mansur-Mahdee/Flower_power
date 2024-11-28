import os
import pandas as pd
import re
import requests
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
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

        # User input to get flower name
        flower_name = st.text_input("Enter a flower name:", "")
        if flower_name:
            flower_name = flower_name.strip().capitalize()  # Capitalize to match the dataset
            if flower_name in flower_info_dict:
                flower_name, flower_description, generated_info = generate_flower_info(flower_name, flower_info_dict, gpt2_pipeline)
                st.write(f"### Information for {flower_name}:")
                st.write(f"**Meaning**: {flower_description}")
                st.write(f"**Cultural or Historical Significance**: {generated_info}")
            else:
                st.write(f"Sorry, we don't have information on the flower: {flower_name}")
    else:
        st.write("There was an issue with downloading or loading the dataset.")

if __name__ == "__main__":
    streamlit_app()
