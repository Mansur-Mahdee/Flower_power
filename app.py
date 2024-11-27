 
import os
import zipfile
import pandas as pd
import re
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

# Function to extract dataset
def extract_dataset():
    # Path to the downloaded zip file and extraction directory
    zip_path = "/mount/src/flower_power/language-of-flowers.zip"
    extraction_dir = "/tmp/language_of_flowers"
    
    # Check if the zip file exists and unzip it
    if os.path.exists(zip_path):
        # Unzip the file if it's not already extracted
        if not os.path.exists(extraction_dir):
            os.makedirs(extraction_dir)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extraction_dir)
            st.write(f"Dataset unzipped to {extraction_dir}")
        else:
            st.write(f"Dataset already extracted to {extraction_dir}")
        
        # Define path to the CSV file inside the extracted folder
        dataset_path = os.path.join(extraction_dir, "language-of-flowers.csv")
        
        # Check if the CSV file exists
        if os.path.exists(dataset_path):
            st.write("Dataset file found!")
            return dataset_path
        else:
            st.error(f"CSV file not found in {extraction_dir}")
            return None
    else:
        st.error(f"Zip file not found at {zip_path}")
        return None

# Function to get flower information based on the flower name
def generate_flower_info(flower_name, flower_info_dict, gpt2_pipeline):
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

# Load the dataset and prepare necessary models and pipelines
dataset_path = extract_dataset()

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
                flower_name, flower_description, generated_info = generate_flower_info(flower_name, flower_info_dict, gpt2_pipeline)
                st.write(f"### Information for {flower_name}:")
                st.write(f"**Meaning**: {flower_description}")
                st.write(f"**Cultural or Historical Significance**: {generated_info}")
            else:
                st.write(f"Sorry, we don't have information on the flower: {flower_name}")

    if __name__ == "__main__":
        streamlit_app()

else:
    st.write("There was an issue with extracting or loading the dataset.")



