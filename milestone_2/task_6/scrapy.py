import sys
import asyncio
from io import BytesIO
from playwright.async_api import async_playwright


import json
import base64
import pandas as pd
from typing import List, Dict
from asyncio import gather

import streamlit as st

# Use ProactorEventLoop on Windows to support subprocess execution
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehanceScraper:
    def __init__(self):
        self.base_url = "https://www.behance.net"
        logger.debug("BehanceScraper initialized with base_url: %s", self.base_url)
        
    async def init_browser(self):
        logger.info("Initializing browser...")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        logger.info("Browser initialized successfully.")
        
    async def close_browser(self):
        logger.info("Closing browser...")
        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()
        logger.info("Browser closed.")
        
    async def scroll_page(self, target_items):
        logger.debug("Starting page scroll to collect %d items...", target_items)
        items = []
        previous_height = 0
        retry_count = 0
        max_retries = 3

        while len(items) < target_items:
            if self.current_section == "assets":
                items = await self.page.query_selector_all("div.ProjectCoverNeue-cover-X3S")
            else:
                items = await self.page.query_selector_all("div.JobCard-jobCard-mzZ")
                
            logger.debug("Collected %d items so far...", len(items))
            current_height = await self.page.evaluate('document.documentElement.scrollHeight')
            
            if current_height == previous_height:
                retry_count += 1
                logger.warning("No new items loaded. Retry attempt %d of %d", retry_count, max_retries)
                if retry_count >= max_retries:
                    break
            else:
                retry_count = 0
                
            await self.page.evaluate('window.scrollTo(0, document.documentElement.scrollHeight)')
            await asyncio.sleep(2)
            previous_height = current_height
            
            progress = min(len(items), target_items) / target_items
            st.progress(progress)
        
        logger.info("Finished scrolling with %d items collected.", len(items))
        return items[:target_items]

    async def extract_asset_data(self, item):
        try:
            title = await item.query_selector("a.Title-title-lpJ")
            title_text = await title.text_content() if title else "N/A"
            
            creator = await item.query_selector("a.Owners-owner-EEG")
            creator_text = await creator.text_content() if creator else "N/A"
            
            price_elem = await item.query_selector("span.PaidAssetsCountBadge-count-yPz")
            price = await price_elem.text_content() if price_elem else "N/A"
            
            likes = await item.query_selector("div.Stats-stats-Q1s span")
            likes_text = await likes.text_content() if likes else "0"
            
            logger.debug("Extracted asset data: %s", {
                "title": title_text, "creator": creator_text, "price": price, "likes": likes_text
            })
            return {
                "title": title_text,
                "creator": creator_text,
                "price": price,
                "likes": likes_text
            }
        except Exception as e:
            logger.error("Error extracting asset data: %s", e)
            return {"error": str(e)}

    async def extract_job_data(self, item):
        try:
            title = await item.query_selector("h3.JobCard-jobTitle-LS4")
            title_text = await title.text_content() if title else "N/A"
            
            company = await item.query_selector("p.JobCard-company-GQS")
            company_text = await company.text_content() if company else "N/A"
            
            location = await item.query_selector("p.JobCard-jobLocation-sjd")
            location_text = await location.text_content() if location else "N/A"
            
            description = await item.query_selector("p.JobCard-jobDescription-SYp")
            description_text = await description.text_content() if description else "N/A"
            
            posted = await item.query_selector("span.JobCard-time-Cvz")
            posted_text = await posted.text_content() if posted else "N/A"
            
            logger.debug("Extracted job data: %s", {
                "title": title_text, "company": company_text, "location": location_text, "description": description_text, "posted": posted_text
            })
            return {
                "title": title_text,
                "company": company_text,
                "location": location_text,
                "description": description_text,
                "posted": posted_text
            }
        except Exception as e:
            logger.error("Error extracting job data: %s", e)
            return {"error": str(e)}

    async def scrape_data(self, section, target_items, search_query=None):
        self.current_section = section
        logger.info("Starting scrape for section: %s with target items: %d", section, target_items)
        
        if section == "assets":
            await self.page.goto(f"{self.base_url}/assets")
        else:
            await self.page.goto(f"{self.base_url}/joblist")
            
        if search_query:
            logger.debug("Performing search with query: %s", search_query)
            search_input = await self.page.query_selector('input.SearchBar-searchInput-dYr[type="search"]')
            await search_input.fill(search_query)
            await search_input.press("Enter")
            await asyncio.sleep(2)
            
        items = await self.scroll_page(target_items)
        results = []
        
        for item in items:
            if section == "assets":
                data = await self.extract_asset_data(item)
            else:
                data = await self.extract_job_data(item)
            results.append(data)
        
        logger.info("Scraping completed with %d results.", len(results))
        return results

def get_download_link(df, file_type):
    if file_type == "csv":
        data = df.to_csv(index=False)
        b64 = base64.b64encode(data.encode()).decode()
        href = f'data:file/csv;base64,{b64}'
        return href, "behance_data.csv"
    elif file_type == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        b64 = base64.b64encode(output.getvalue()).decode()
        href = f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'
        return href, "behance_data.xlsx"
    else:
        data = df.to_json(orient='records')
        b64 = base64.b64encode(data.encode()).decode()
        href = f'data:file/json;base64,{b64}'
        return href, "behance_data.json"

async def run_scraper(section, target_items, search_query):
    scraper = BehanceScraper()
    await scraper.init_browser()
    
    with st.spinner("Scraping data..."):
        results = await scraper.scrape_data(section, target_items, search_query)
    
    await scraper.close_browser()
    
    if len(results) < target_items:
        st.warning(f"Only {len(results)} items found, which is less than the requested {target_items} items.")
        
    df = pd.DataFrame(results)
    st.dataframe(df)
    
    st.markdown("### Download Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        href, filename = get_download_link(df, "csv")
        st.download_button(
            label="Download CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=filename,
            mime='text/csv'
        )
    with col2:
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name="behance_data.xlsx",
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    with col3:
        st.download_button(
            label="Download JSON",
            data=df.to_json(orient='records').encode('utf-8'),
            file_name="behance_data.json",
            mime='application/json'
        )

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
