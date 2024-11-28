 
import os
import zipfile
import pandas as pd
import re
import subprocess
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from datasets import load_dataset


@st.cache_resource
def download_and_extract_dataset():
    # Path to the dataset extraction directory and CSV file
    extraction_dir = "/tmp/language_of_flowers"
    dataset_path = os.path.join(extraction_dir, "language-of-flowers.csv")
    
    # Check if the dataset CSV already exists
    if os.path.exists(dataset_path):
        st.write("Dataset already exists, skipping download and extraction.")
        return dataset_path

    # If not, proceed to download the dataset
    kaggle_username = st.secrets["username"]
    kaggle_key = st.secrets["key"]
    
    if not kaggle_username or not kaggle_key:
        st.error("Kaggle API token not found in Streamlit secrets. Please add your 'kaggle.json' details.")
        return None

    # Set up Kaggle credentials
    os.makedirs('/tmp/.kaggle', exist_ok=True)
    with open("/tmp/.kaggle/kaggle.json", "w") as f:
        f.write(f'{{"username": "{kaggle_username}", "key": "{kaggle_key}"}}')

    # Download dataset from Kaggle
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "jenlooper/language-of-flowers"],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE
        )
        st.write(result.stdout.decode())  # Log the output of the download
        st.write(result.stderr.decode())  # Log any error messages if they exist
    except subprocess.CalledProcessError as e:
        st.error(f"Error during dataset download: {e.stderr.decode()}")
        return None

    # List files in the temporary directory
    download_dir = "/tmp"
    downloaded_files = os.listdir(download_dir)
    st.write("Downloaded files:")
    st.write(downloaded_files)

    # Check if the dataset is already a CSV (no need to unzip)
    for file in downloaded_files:
        if file.endswith('.csv'):
            st.write(f"CSV file found directly: {file}")
            return os.path.join(download_dir, file)

    # If no CSV found, check for a ZIP file and extract it
    zip_file_path = None
    for file in downloaded_files:
        if file.endswith('.zip'):
            zip_file_path = os.path.join(download_dir, file)
            break

    if not zip_file_path:
        st.error("No CSV or ZIP file found in the downloaded files.")
        return None

    # Extract the ZIP file if found
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_dir)
    st.write(f"Dataset extracted to {extraction_dir}")

    # Now check if the CSV file exists in the extraction directory
    if os.path.exists(dataset_path):
        st.write("Dataset file found!")
        return dataset_path
    else:
        st.error(f"CSV file not found after extraction in {extraction_dir}")
        return None


# Call the function to download or extract dataset
dataset_path = download_and_extract_dataset()

if dataset_path:
    st.write(f"Dataset ready at {dataset_path}")
else:
    st.write("There was an issue with downloading or extracting the dataset.")
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
    data = pd.read_csv(dataset_path)

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
    st.write("There was an issue with extracting or loading the dataset.")



