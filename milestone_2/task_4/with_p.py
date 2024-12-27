import streamlit as st
import pandas as pd
from playwright.async_api import async_playwright
import asyncio
import time
from typing import List, Dict
import json
from io import BytesIO
import aiohttp
from asyncio import gather
import logging
import sys

# Use ProactorEventLoop on Windows to support subprocess execution
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'selected_store' not in st.session_state:
    st.session_state.selected_store = None
if 'scraped_data' not in st.session_state:
    st.session_state.scraped_data = []

async def get_store_names() -> List[str]:
    """Scrape all store names from the website."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto("https://dealsheaven.in/stores")
            
            stores = await page.query_selector_all("ul.store-listings li a")
            store_names = []
            for store in stores:
                name = await store.text_content()
                if name:
                    store_names.append(name.strip())
            
            await browser.close()
            return sorted(store_names)
    except Exception as e:
        logger.error(f"Error getting store names: {e}")
        return []

async def scrape_single_page(page, url: str) -> List[Dict]:
    """Scrape a single page of products."""
    try:
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        
        products = []
        product_items = await page.query_selector_all("div.product-item-detail")
        
        for item in product_items:
            try:
                product_data = {}
                
                # Extract time
                time_elem = await item.query_selector("div.time")
                product_data['time_posted'] = await time_elem.get_attribute('title') if time_elem else ''
                
                # Extract discount
                discount_elem = await item.query_selector("div.discount")
                product_data['discount'] = await discount_elem.text_content() if discount_elem else ''
                
                # Extract title
                title_elem = await item.query_selector("h3")
                product_data['title'] = await title_elem.get_attribute('title') if title_elem else ''
                
                # Extract prices
                original_price = await item.query_selector("p.price")
                special_price = await item.query_selector("p.spacail-price")
                product_data['original_price'] = await original_price.text_content() if original_price else ''
                product_data['special_price'] = await special_price.text_content() if special_price else ''
                
                # Extract shop now link
                shop_now = await item.query_selector("div.shop-now-button a")
                product_data['shop_link'] = await shop_now.get_attribute('href') if shop_now else ''
                
                # Add page number
                product_data['page_number'] = url.split('page=')[-1].split('&')[0]
                
                products.append(product_data)
            except Exception as e:
                logger.error(f"Error scraping product: {e}")
                continue
                
        return products
    except Exception as e:
        logger.error(f"Error scraping page {url}: {e}")
        return []

async def scrape_products(store: str, num_pages: int, search_query: str = None) -> List[Dict]:
    """Scrape products from the selected store using parallel processing."""
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        
        # Create a context with multiple pages
        context = await browser.new_context()
        
        # Generate URLs for all pages
        store_url = f"https://dealsheaven.in/store/{store.lower()}"
        urls = []
        for page_num in range(1, num_pages + 1):
            current_url = f"{store_url}?page={page_num}"
            if search_query:
                current_url += f"&keyword={search_query}"
            urls.append(current_url)
        
        # Create multiple pages for parallel scraping
        tasks = []
        for url in urls:
            page = await context.new_page()
            tasks.append(scrape_single_page(page, url))
        
        # Run all scraping tasks in parallel
        results = await gather(*tasks)
        
        # Flatten results
        all_products = [product for page_products in results for product in page_products]
        
        await browser.close()
        return all_products

def save_data(data: List[Dict], format: str):
    """Save scraped data in the specified format."""
    df = pd.DataFrame(data)
    
    if format == 'csv':
        return df.to_csv(index=False).encode('utf-8')
    elif format == 'excel':
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        return output.getvalue()
    elif format == 'json':
        return json.dumps(data, indent=2)

def create_streamlit_ui():
    """Create the Streamlit user interface."""
    st.title("🛍️ DealsHeaven Store Scraper")

    # Get store names
    stores = asyncio.run(get_store_names())
    store_select = st.selectbox("Select a Store", options=stores)

    # Number of pages to scrape
    num_pages = st.number_input("Number of Pages to Scrape", min_value=1, max_value=10, value=1)

    # Search functionality
    search_query = st.text_input("Search Products (Optional)")

    return store_select, num_pages, search_query

def handle_results(scraped_data, store_select):
    """Handle and display scraping results."""
    if scraped_data:
        df = pd.DataFrame(scraped_data)
        st.dataframe(df)
        
        # Download buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            csv_data = save_data(scraped_data, 'csv')
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{store_select}_products.csv",
                mime="text/csv"
            )
        
        with col2:
            excel_data = save_data(scraped_data, 'excel')
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name=f"{store_select}_products.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            json_data = save_data(scraped_data, 'json')
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name=f"{store_select}_products.json",
                mime="application/json"
            )
    else:
        st.warning("No products found!")

def heaven_main():
    """Main function to run the Streamlit application."""
    # Create UI elements
    store_select, num_pages, search_query = create_streamlit_ui()

    if st.button("Start Scraping"):
        with st.spinner("Scraping data..."):
            try:
                # Run the scraping process
                scraped_data = asyncio.run(scrape_products(store_select, num_pages, search_query))
                st.session_state.scraped_data = scraped_data
                st.session_state.selected_store = store_select
                
                # Handle and display results
                handle_results(scraped_data, store_select)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Scraping error: {e}")

if __name__ == "__main__":
    heaven_main()