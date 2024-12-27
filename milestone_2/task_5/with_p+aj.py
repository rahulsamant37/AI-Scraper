import logging
import streamlit as st
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import json
from io import BytesIO
import time
import base64
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use ProactorEventLoop on Windows to support subprocess execution
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

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

# Streamlit Interface
st.title("Behance Data Scraper")
section = st.selectbox("Select Section", ["assets", "jobs"])
target_items = st.number_input("Number of items to scrape", min_value=1, value=10)
search_query = st.text_input("Search Query (Optional)")

if st.button("Start Scraping"):
    asyncio.run(run_scraper(section, target_items, search_query))
