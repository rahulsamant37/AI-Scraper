import streamlit as st
import asyncio
from playwright.async_api import async_playwright
import pandas as pd
import json
from io import BytesIO
import time
import base64
import sys
# Use ProactorEventLoop on Windows to support subprocess execution
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

class BehanceScraper:
    def __init__(self):
        self.base_url = "https://www.behance.net"
        
    async def init_browser(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
        
    async def close_browser(self):
        await self.context.close()
        await self.browser.close()
        await self.playwright.stop()
        
    async def scroll_page(self, target_items):
        items = []
        previous_height = 0
        retry_count = 0
        max_retries = 3

        while len(items) < target_items:
            # Get current items
            if self.current_section == "assets":
                items = await self.page.query_selector_all("div.ProjectCoverNeue-cover-X3S")
            else:
                items = await self.page.query_selector_all("div.JobCard-jobCard-mzZ")
            
            current_height = await self.page.evaluate('document.documentElement.scrollHeight')
            
            if current_height == previous_height:
                retry_count += 1
                if retry_count >= max_retries:
                    break
            else:
                retry_count = 0
                
            await self.page.evaluate('window.scrollTo(0, document.documentElement.scrollHeight)')
            await asyncio.sleep(2)  # Wait for content to load
            previous_height = current_height
            
            # Update progress
            progress = min(len(items), target_items) / target_items
            st.progress(progress)
            
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
            
            return {
                "title": title_text,
                "creator": creator_text,
                "price": price,
                "likes": likes_text
            }
        except Exception as e:
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
            
            return {
                "title": title_text,
                "company": company_text,
                "location": location_text,
                "description": description_text,
                "posted": posted_text
            }
        except Exception as e:
            return {"error": str(e)}

    async def scrape_data(self, section, target_items, search_query=None):
        self.current_section = section
        
        # Navigate to appropriate section
        if section == "assets":
            await self.page.goto(f"{self.base_url}/assets")
        else:
            await self.page.goto(f"{self.base_url}/joblist")
            
        # Handle search if provided
        if search_query:
            search_input = await self.page.query_selector('input[type="search"]')
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
            
        return results

def get_download_link(df, file_type):
    if file_type == "csv":
        data = df.to_csv(index=False)
        b64 = base64.b64encode(data.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="behance_data.csv">Download CSV</a>'
    elif file_type == "excel":
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False)
        b64 = base64.b64encode(output.getvalue()).decode()
        return f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="behance_data.xlsx">Download Excel</a>'
    else:  # json
        data = df.to_json(orient='records')
        b64 = base64.b64encode(data.encode()).decode()
        return f'<a href="data:file/json;base64,{b64}" download="behance_data.json">Download JSON</a>'

async def main():
    st.title("Behance Data Scraper")
    
    # UI Controls
    section = st.selectbox("Select Section", ["assets", "jobs"])
    target_items = st.number_input("Number of items to scrape", min_value=1, value=10)
    search_query = st.text_input("Search Query (Optional)")
    
    if st.button("Start Scraping"):
        scraper = BehanceScraper()
        await scraper.init_browser()
        
        with st.spinner("Scraping data..."):
            results = await scraper.scrape_data(section, target_items, search_query)
        
        await scraper.close_browser()
        
        if len(results) < target_items:
            st.warning(f"Only {len(results)} items found, which is less than the requested {target_items} items.")
            
        # Display results
        df = pd.DataFrame(results)
        st.dataframe(df)
        
        # Download buttons
        st.markdown("### Download Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(get_download_link(df, "csv"), unsafe_allow_html=True)
        with col2:
            st.markdown(get_download_link(df, "excel"), unsafe_allow_html=True)
        with col3:
            st.markdown(get_download_link(df, "json"), unsafe_allow_html=True)

if __name__ == "__main__":
    asyncio.run(main())