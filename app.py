import pandas as pd
import requests
from io import StringIO
import streamlit as st

# Replace with the raw GitHub file URL
github_raw_url = "https://raw.githubusercontent.com/Mansur-Mahdee/Flower_power/refs/heads/main/data/language-of-flowers.csv"

# Function to fetch the CSV file and load it into a DataFrame
def load_csv_from_github(url):
    try:
        # Make a request to the raw GitHub URL with appropriate headers
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Ensure we got a valid response

        # If the request is successful, we load the content into pandas
        csv_data = StringIO(response.text)  # Use StringIO to handle text as a file
        data = pd.read_csv(csv_data, quotechar='"', encoding='utf-8-sig', on_bad_lines='skip')

        # Show the raw data's column names and first few rows for debugging
        st.write("Column names:", list(data.columns))
        st.write("First few rows of the dataset:", data.head())

        # Clean column names by stripping any whitespace
        data.columns = data.columns.str.strip()
        st.write("Cleaned column names:", list(data.columns))

        # Check if 'Flower' and 'Meaning' columns exist
        if 'Flower' in data.columns and 'Meaning' in data.columns:
            flower_info_dict = dict(zip(data['Flower'], data['Meaning']))
            st.write(flower_info_dict)
        else:
            st.error("The expected 'Flower' and 'Meaning' columns are missing or named incorrectly.")

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching the CSV file: {str(e)}")
    except Exception as e:
        st.error(f"Error reading the CSV data: {str(e)}")

# Call the function with the correct GitHub raw URL
load_csv_from_github(github_raw_url)
