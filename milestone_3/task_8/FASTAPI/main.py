import os
import json
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, create_model
from dotenv import load_dotenv

import logging
import pandas as pd

# Import the core functionality from the original script
from scraper import (
    fetch_html_playwright, 
    html_to_markdown_with_readability, 
    create_dynamic_listing_model,
    create_listings_container_model,
    format_data,
    save_formatted_data,
    calculate_price,
    setup_logging
)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Web Scraping and Data Extraction API",
    description="An API for web scraping, data extraction, and formatting",
    version="1.0.0"
)

# Configuration model for scraping request
class ScrapingRequest(BaseModel):
    url: str
    fields: List[str]
    model: Optional[str] = "gemini-1.5-flash"

# Result model to be dynamically created based on fields
def create_result_model(fields: List[str]):
    field_definitions = {field: (str, None) for field in fields}
    return create_model('ScrapingResult', **field_definitions)

@app.get("/")
def root():
    return {
        "title": "Web Scraping and Data Extraction API",
        "version": "1.0.0",
        "description": "An API for web scraping, data extraction, and formatting",
        "available_endpoints": [
            "/scrape/",
            "/models/",
            "/pricing/"
        ]
    }

@app.post("/scrape/")
async def scrape_website(request: ScrapingRequest):
    """
    Endpoint to scrape a website and extract structured data
    
    Parameters:
    - url: Website to scrape
    - fields: List of fields to extract
    - model: LLM model to use for data extraction (optional)
    
    Returns:
    - Extracted data
    - Token usage
    - Cost information
    """
    # Setup logging with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logging(timestamp)
    
    try:
        # Fetch HTML content
        logger.info(f"Fetching HTML from URL: {request.url}")
        raw_html = await fetch_html_playwright(request.url)
        
        # Convert HTML to markdown
        markdown = html_to_markdown_with_readability(raw_html)
        
        # Create dynamic models for data extraction
        DynamicListingModel = create_dynamic_listing_model(request.fields)
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        
        # Format data using selected model
        formatted_data, token_counts = format_data(
            markdown, 
            DynamicListingsContainer, 
            DynamicListingModel, 
            request.model
        )
        
        # Calculate pricing
        input_tokens, output_tokens, total_cost = calculate_price(token_counts, request.model)
        
        # Prepare response
        response = {
            "data": formatted_data,
            "tokens": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            },
            "cost": total_cost,
            "model": request.model
        }
        
        # Optional: Save data to files
        save_formatted_data(formatted_data, timestamp)
        
        return response
    
    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Additional endpoints for specific use cases

@app.get("/models/")
def list_available_models():
    """
    Endpoint to list available LLM models
    """
    return {
        "models": [
            "whisper-large-v3", 
            "gemini-1.5-flash", 
            "Llama3.1 8B", 
            "Groq Llama3.1 70b"
        ]
    }

@app.get("/pricing/")
def get_model_pricing():
    """
    Endpoint to retrieve model pricing information
    """
    from assets import PRICING  # Assuming pricing is defined in assets.py
    return PRICING

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)