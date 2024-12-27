import streamlit as st
from streamlit_tags import st_tags_sidebar
import pandas as pd
import json
import sys
import asyncio
import os
import logging
from datetime import datetime
from scraper import fetch_html_playwright, save_raw_data, format_data, save_formatted_data, calculate_price, html_to_markdown_with_readability, create_dynamic_listing_model, create_listings_container_model, get_text_chunks

from assets import PRICING

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import re
import validators  # You'll need to install this library with: pip install validators

def validate_url(url):
    """
    Validate the URL format and check if it's a valid, accessible URL
    
    Args:
        url (str): The URL to validate
    
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check if URL is empty
    if not url:
        return False, "Please enter a URL"
    
    # Basic format validation using validators library
    if not validators.url(url):
        return False, "Invalid URL format. Please enter a valid URL (e.g., https://example.com)"
    
    # Optional: Add more specific checks
    try:
        # Additional URL validation
        parsed_url = validators.url(url)
        if not parsed_url:
            return False, "URL appears to be invalid"
        
        # Optional: You could add additional checks like:
        # - Checking for specific protocols (http/https)
        if not url.startswith(('http://', 'https://')):
            return False, "URL must start with http:// or https://"
        
        return True, ""
    
    except Exception as e:
        return False, f"URL validation error: {str(e)}"

def setup_logging():
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'streamlit_app_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

st.set_page_config(page_title="Universal Web Scraper")
logger.info("Streamlit page configuration set")

st.title("Universal Web Scraper 🦑")
st.sidebar.title("Web Scraper Settings")
model_selection = st.sidebar.selectbox("Select Model", options=list(PRICING.keys()), index=0)
logger.info(f"Model selected: {model_selection}")

url_input = st.sidebar.text_input("Enter URL")
logger.info(f"URL input received: {url_input}")
chunk_size = st.sidebar.slider("Chunk Size", min_value=1000, max_value=20000, value=10000, step=1000)
chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=100, max_value=2000, value=1000, step=100)

tags = st.sidebar.empty()
tags = st_tags_sidebar(
    label='Enter Fields to Extract:',
    text='Press enter to add a tag',
    value=[],
    suggestions=[],
    maxtags=-1,
    key='tags_input'
)
logger.info(f"Fields to extract: {tags}")

st.sidebar.markdown("---")
fields = tags
input_tokens = output_tokens = total_cost = 0

def merge_listings(all_listings):
    """Merge listings by comparing their content to avoid duplicates"""
    merged = []
    seen = set()
    
    for listing in all_listings:
        listing_values = tuple(str(v) for v in listing.values())
        if listing_values not in seen:
            seen.add(listing_values)
            merged.append(listing)
    
    return merged

def process_chunks(chunks, DynamicListingsContainer, DynamicListingModel, model_selection):
    """Process each chunk and combine results while preserving all fields"""
    logger.info(f"Processing {len(chunks)} chunks")
    all_listings = []
    total_tokens = {"input_tokens": 0, "output_tokens": 0}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Processing chunk {i+1}/{len(chunks)}")
        progress_bar.progress((i + 1) / len(chunks))
        
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        formatted_data, tokens_count = format_data(chunk, DynamicListingsContainer, DynamicListingModel, model_selection)
        if isinstance(formatted_data, str):
            try:
                formatted_data = json.loads(formatted_data)
            except json.JSONDecodeError:
                continue
                
        if hasattr(formatted_data, 'dict'):
            formatted_data = formatted_data.dict()
        chunk_listings = formatted_data.get('listings', []) if isinstance(formatted_data, dict) else []
        for listing in chunk_listings:
            for field in fields:
                if field not in listing:
                    listing[field] = ""
        
        all_listings.extend(chunk_listings)
        total_tokens["input_tokens"] += tokens_count.get("input_tokens", 0)
        total_tokens["output_tokens"] += tokens_count.get("output_tokens", 0)
    progress_bar.empty()
    status_text.empty()
    merged_listings = merge_listings(all_listings)
    combined_data = {"listings": merged_listings}
    
    return combined_data, total_tokens

def perform_scrape():
    logger.info("Starting scraping process")
    try:
        if not fields:
            st.error("Please enter at least one field to extract")
            logger.error("No fields specified for extraction")
            return None
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger.info(f"Fetching HTML from URL: {url_input}")
        raw_html = fetch_html_playwright(url_input)
        
        logger.info("Converting HTML to markdown")
        markdown = html_to_markdown_with_readability(raw_html)
        save_raw_data(markdown, timestamp)
        logger.info(f"Splitting text into chunks (size: {chunk_size}, overlap: {chunk_overlap})")
        chunks = get_text_chunks(markdown)
        
        logger.info(f"Creating dynamic models with fields: {fields}")
        DynamicListingModel = create_dynamic_listing_model(fields)
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        with st.spinner('Processing chunks...'):
            formatted_data, tokens_count = process_chunks(chunks, DynamicListingsContainer, DynamicListingModel, model_selection)
        
        logger.info("Calculating price")
        input_tokens, output_tokens, total_cost = calculate_price(tokens_count, model=model_selection)
        
        logger.info("Saving formatted data")
        df = save_formatted_data(formatted_data, timestamp)
        st.sidebar.success(f"Processed {len(chunks)} chunks")
        st.sidebar.info(f"Found {len(formatted_data['listings'])} items")
        
        logger.info("Scraping process completed successfully")
        return df, formatted_data, markdown, input_tokens, output_tokens, total_cost, timestamp
    
    except Exception as e:
        logger.error(f"Error during scraping process: {str(e)}")
        st.sidebar.error(f"An error occurred during scraping: {str(e)}")
        return None

if 'perform_scrape' not in st.session_state:
    st.session_state['perform_scrape'] = False
    logger.info("Initialized session state")

if st.sidebar.button("Scrape"):
    # Validate URL before proceeding
    is_valid, error_message = validate_url(url_input)
    
    if not is_valid:
        # Display error message if URL is invalid
        st.sidebar.error(error_message)
        logger.warning(f"Invalid URL entered: {error_message}")
        st.session_state['perform_scrape'] = False
    else:
        # Proceed with scraping if URL is valid
        logger.info("Scrape button clicked")
        with st.spinner('Please wait... Data is being scraped.'):
            st.session_state['results'] = perform_scrape()
            st.session_state['perform_scrape'] = True
            logger.info("Scraping process initiated")

if st.session_state.get('perform_scrape') and st.session_state.get('results'):
    logger.info("Displaying scraping results")
    df, formatted_data, markdown, input_tokens, output_tokens, total_cost, timestamp = st.session_state['results']
    
    st.write("Scraped Data:", df)
    st.sidebar.markdown("## Token Usage")
    st.sidebar.markdown(f"**Input Tokens:** {input_tokens}")
    st.sidebar.markdown(f"**Output Tokens:** {output_tokens}")
    st.sidebar.markdown(f"**Total Cost:** :green-background[***${total_cost:.4f}***]")
    
    logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Cost: ${total_cost:.4f}")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        json_data = json.dumps(formatted_data, indent=4)
        st.download_button("Download JSON", data=json_data, file_name=f"{timestamp}_data.json")
        logger.info("JSON download button created")
    with col2:
        try:
            if isinstance(formatted_data, str):
                data_dict = json.loads(formatted_data)
            else:
                data_dict = formatted_data
            main_data = data_dict.get('listings', [])
            df = pd.DataFrame(main_data)
            st.download_button("Download CSV", data=df.to_csv(index=False), file_name=f"{timestamp}_data.csv")
            logger.info("CSV download button created")
        except Exception as e:
            logger.error(f"Error creating CSV download button: {str(e)}")
    with col3:
        st.download_button("Download Markdown", data=markdown, file_name=f"{timestamp}_data.md")
        logger.info("Markdown download button created")

if 'results' in st.session_state:
    logger.info("Session state contains results")
    df, formatted_data, markdown, input_tokens, output_tokens, total_cost, timestamp = st.session_state['results']