 
import os
import re
import subprocess
import pandas as pd
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Function to download the Kaggle dataset using API token
def download_kaggle_dataset():
    # Get Kaggle credentials from Streamlit secrets
    kaggle_username = st.secrets["username"]
    kaggle_key = st.secrets["key"]
    
    # Ensure the credentials are available in the Streamlit secrets
    if not kaggle_username or not kaggle_key:
        st.error("Kaggle API token not found in Streamlit secrets. Please add your 'kaggle.json' details.")
        return None

    # Create the directory for the Kaggle API token in the temporary directory
    os.makedirs('/tmp/.kaggle', exist_ok=True)
    
    # Write the Kaggle credentials to a JSON file
    with open("/tmp/.kaggle/kaggle.json", "w") as f:
        f.write(f'{{"username": "{kaggle_username}", "key": "{kaggle_key}"}}')
    
    # Download the dataset from Kaggle using the Kaggle API
    subprocess.run(["kaggle", "datasets", "download", "-d", "jenlooper/language-of-flowers"], check=True)
    
    # Path to the dataset (no need to unzip)
    dataset_path = "/root/.cache/kagglehub/datasets/jenlooper/language-of-flowers/versions/2/language-of-flowers.csv"

    # Check if the dataset file exists
    if not os.path.exists(dataset_path):
        st.error(f"Dataset file not found at {dataset_path}")
        return None
    
    # If the file exists, return the dataset path
    st.write("Dataset found successfully!")
    return dataset_path

# Call the function and use the dataset
dataset_path = download_kaggle_dataset()
if dataset_path:
    data = pd.read_csv(dataset_path)
    st.write(data.head())
 
# Function to get flower information based on the flower name
def generate_flower_info(flower_name, flower_info_dict):
    """
    Generate information about a flower based on the flower name.
    
    Parameters:
        flower_name (str): Name of the flower.
    
    Returns:
        tuple: Flower name, flower description, and generated information.
    """
    # Get flower description from the dataset
    flower_description = flower_info_dict.get(flower_name, "No description available.")
    
    # Use GPT-2 model if we want to generate additional details about the flower
    query = f"Why is the flower {flower_name} associated with the meaning '{flower_description}'? Explain the cultural or historical significance behind the {flower_name}."
    
    # Generate additional information about the flower using GPT-2
    generated_info = gpt2_pipeline(query, max_length=200, truncation=True)[0]["generated_text"]
    
    # Split the generated text into sentences
    sentences = re.split(r'(?<=\w[.!?])\s+', generated_info.strip())

    # Limit the output to the first 5 sentences
    limited_output = "".join(sentences[:5])

    return flower_name, flower_description, limited_output

# Load the Kaggle dataset (CSV file) when the app starts
dataset_path = download_kaggle_dataset()

if dataset_path is not None:
    # Load dataset into DataFrame
    data = pd.read_csv(dataset_path)

    # Create a dictionary to map flower names to meanings
    flower_info_dict = dict(zip(data['Flower'], data['Meaning']))

    # Initialize the RAG model components
    tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
    model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)
    tokenizer.pad_token_id = 0

    # Define the RAG pipeline for text generation
    rag_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

    # Load GPT-2 model and tokenizer for text generation
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Define GPT-2 text generation pipeline
    gpt2_pipeline = pipeline("text-generation", model=gpt2_model, tokenizer=gpt2_tokenizer)

    # Streamlit App Function
    def streamlit_app():
        st.title("Language of Flowers")

        st.write(
            "Welcome to the Language of Flowers App! Here you can learn about the meanings and cultural significance of different flowers."
        )

        # User input to get flower name
        flower_name = st.text_input("Enter a flower name:", "")

        if flower_name:
            flower_name = flower_name.strip().capitalize()  # Capitalize for matching with dataset
            if flower_name in flower_info_dict:
                flower_name, flower_description, generated_info = generate_flower_info(flower_name, flower_info_dict)
                st.write(f"### Information for {flower_name}:")
                st.write(f"**Meaning**: {flower_description}")
                st.write(f"**Cultural or Historical Significance**: {generated_info}")
            else:
                st.write(f"Sorry, we don't have information on the flower: {flower_name}")

    if __name__ == "__main__":
        streamlit_app()


