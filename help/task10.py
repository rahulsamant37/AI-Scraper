import logging
from typing import List, Type
from pydantic import BaseModel, create_model
import streamlit as st
from streamlit_tags import st_tags
import random
import requests

# Function to create dynamic listing model based on user input
def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    logger = logging.getLogger(__name__)
    try:
        # Create dynamic fields with string type for each input field
        field_definitions = {field: (str, ...) for field in field_names}
        model = create_model('DynamicListingModel', **field_definitions)
        logger.info(f"Dynamic listing model created with fields: {field_names}")
        return model
    except Exception as e:
        logger.error(f"Error creating dynamic listing model: {e}")
        raise

# Function to create a container model for holding multiple listings
def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    logger = logging.getLogger(__name__)
    try:
        container_model = create_model('DynamicListingsContainer', listings=(List[listing_model], ...))
        logger.info("Listings container model created")
        return container_model
    except Exception as e:
        logger.error(f"Error creating listings container model: {e}")
        raise

# Sidebar for Web Scraper Settings in Streamlit
st.sidebar.title("Web Scraper Settings")
model_choice = st.sidebar.selectbox("Select Model", ["gemini-1.5-flash", "gpt-4o-mini"])
url = st.sidebar.text_input("Enter URL", "")

# Tag-like input for fields
with st.sidebar:
    fields_input_tags = st_tags(
        label="Enter Fields to Extract:",
        text="Add more fields",            # Placeholder text for the input
        value=[],                           # Default tags, can be empty initially
        suggestions=[],                     # Suggestions, if any
        maxtags=10,                         # Maximum number of tags
        key="field_input"                   # Unique key for this widget
    )

# User-agent list
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59 Safari/537.36",
]

# Function to get data with dynamic headers and user-agent
def get_data_with_user_agent(url: str) -> str:
    headers = {
        "User-Agent": random.choice(user_agents)
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception("Failed to retrieve data")

# Button for initiating scraping
if st.sidebar.button("Scrape"):
    st.sidebar.write("Scraping initiated...")  # This will display when scraping starts
    try:
        # Create dynamic model using fields inputted by the user
        if fields_input_tags:
            dynamic_listing_model = create_dynamic_listing_model(fields_input_tags)
            # Create container model for holding multiple listings
            listings_container_model = create_listings_container_model(dynamic_listing_model)

            # Scrape the data from the entered URL
            raw_data = get_data_with_user_agent(url)
            # For now, we just display the raw data (you would typically process this further)
            st.write(raw_data)
            
            # Example of using dynamic model and container
            listings = listings_container_model(listings=[])
            st.write(listings)
    except Exception as e:
        st.error(f"Error during scraping: {e}")

# Main Title
st.markdown(
    """
    <div style='text-align: center; padding-top: 50px; font-size: 36px; color: white;'>
        <b>Universal Web Scraper <span>&#128187;</span></b>
    </div>
    """,
    unsafe_allow_html=True
)

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Sidebar background */
        .css-18e3th9 {
            background-color: #1e1e1e;
        }

        /* Sidebar content styling */
        .stSidebar {
            background-color: #303030;
            color: white;
        }

        /* Main title styling */
        .css-145kmo2 {
            background-color: #FF4B4B;
            color: white;
            border: none;
        }
    </style>
    """,
    unsafe_allow_html=True
)
