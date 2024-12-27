import streamlit as st
import asyncio
from scrapy import BehanceScraper, heaven_main, run_scraper

def main():
    st.set_page_config(page_title="Web Scraper", layout="wide")
    
    # Main title and scraper selection
    st.title("🌐 Web Scraper")
    
    # Create tabs for different scrapers
    scraper_type = st.selectbox(
        "Select Scraper",
        ["Choose a scraper...", "Behance Scraper", "DealsHeaven Scraper"],
        index=0
    )
    
    if scraper_type == "Behance Scraper":
        st.header("Behance Scraper")
        
        # Behance scraper options
        section = st.radio("Select Section", ["assets", "jobs"])
        target_items = st.number_input("Number of Items to Scrape", min_value=1, max_value=100, value=10)
        search_query = st.text_input("Search Query (Optional)")
        
        if st.button("Start Behance Scraping"):
            asyncio.run(run_scraper(section, target_items, search_query))
            
    elif scraper_type == "DealsHeaven Scraper":
        st.header("DealsHeaven Scraper")
        heaven_main()

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            Select a scraper from the dropdown above to begin scraping data.
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()