import os
import pandas as pd
import re
import requests
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from datasets import load_dataset

@st.cache_resource
def download_dataset_from_github():
    # URL of the CSV file on GitHub
    github_url = "https://github.com/Mansur-Mahdee/Flower_power/blob/main/data/language-of-flowers.csv"
    
    # Define the path to save the dataset
    dataset_path = "/tmp/language-of-flowers.csv"
    
    # Check if the dataset already exists
    if os.path.exists(dataset_path):
        st.write("Dataset already exists, skipping download.")
        return dataset_path

    # Download the CSV file from GitHub
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Raise an exception for 4xx/5xx errors

        # Save the content as a CSV file
        with open(dataset_path, 'wb') as f:
            f.write(response.content)

        st.write(f"Downloaded dataset from GitHub to {dataset_path}")
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error during dataset download: {e}")
        return None
    
    return dataset_path

# Call the function to download the dataset from GitHub
dataset_path = download_dataset_from_github()

if dataset_path:
    st.write(f"Dataset ready at {dataset_path}")
else:
    st.write("There was an issue with downloading the dataset.")


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


if dataset_path is not None:
    # Load dataset into DataFrame
    try:
        # Load the dataset and ensure proper handling of empty spaces and missing values
        data = pd.read_csv("/tmp/language-of-flowers.csv", quotechar='"', encoding='utf-8', on_bad_lines='skip')

    # Strip spaces from column names to ensure proper access
        data.columns = data.columns.str.strip()

    # Display the column names to check if the columns are correctly named
        st.write(data.columns)

    # Remove the 'Color' column if it is not necessary for the dictionary creation
        data_cleaned = data[['Flower', 'Meaning']]

    # Display the first few rows of the cleaned data to check
        st.write(data_cleaned.head())

    # Create the flower-info dictionary
        flower_info_dict = dict(zip(data_cleaned['Flower'], data_cleaned['Meaning']))

    except Exception as e:
        st.error(f"Error reading CSV: {e}")

    # Create a dictionary to map flower names to meanings
    flower_info_dict = dict(zip(data['Flower'], data['Meaning']))

    dataset = load_dataset("wiki_dpr","psgs_w100.nq.exact", trust_remote_code=True)
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

    # Load GPT-2 model and tokenizer for text generation
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
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
    st.write("There was an issue with downloading or loading the dataset.")
