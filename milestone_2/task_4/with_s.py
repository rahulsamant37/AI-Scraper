from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import streamlit as st
import json
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

@st.cache_data(ttl=3600)
def scrape_stores():
    url = "https://dealsheaven.in/stores"

    driver.get(url)
    time.sleep(2)

    stores = driver.find_elements(By.CSS_SELECTOR, "ul.store-listings li a")
    store_data = [{"name": store.text.strip(), "url": store.get_attribute('href').strip()} for store in stores]
    
    return store_data

def scrape_single_page(url):
    try:
        driver.get(url)
        time.sleep(2)
        
        products = driver.find_elements(By.CSS_SELECTOR, '.product-item-detail')
        
        page_data = []
        for product in products:
            product_data = {
                'Product Name': product.find_element(By.TAG_NAME, 'h3').text.strip() if product.find_elements(By.TAG_NAME, 'h3') else "N/A",
                'Current Price': product.find_element(By.CLASS_NAME, 'price').text.strip() if product.find_elements(By.CLASS_NAME, 'price') else "N/A",
                'Special Price': product.find_element(By.CLASS_NAME, 'spacail-price').text.strip() if product.find_elements(By.CLASS_NAME, 'spacail-price') else "N/A",
                'Discount': product.find_element(By.CLASS_NAME, 'discount').text.strip() if product.find_elements(By.CLASS_NAME, 'discount') else "N/A",
                'Image URL': product.find_element(By.CSS_SELECTOR, 'img').get_attribute('data-src') if product.find_elements(By.CSS_SELECTOR, 'img[data-src]') else "N/A"
            }
            page_data.append(product_data)
        
        return page_data
    except Exception as e:
        st.error(f"Error scraping page: {e}")
        return []

def scrape_store_deals(store_name, max_pages):
    try:
        url = f"https://dealsheaven.in/store/{store_name}"
        driver.get(url)
        time.sleep(2)
        
        product_data = []
        current_page = 1
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        while current_page <= max_pages:
            st.info(f"Scraping page {current_page}...")

            products = driver.find_elements(By.CSS_SELECTOR, '.product-item-detail')
            for product in products:
                product_data.append({
                    'Product Name': product.find_element(By.TAG_NAME, 'h3').text.strip() if product.find_elements(By.TAG_NAME, 'h3') else "N/A",
                    'Current Price': product.find_element(By.CLASS_NAME, 'price').text.strip() if product.find_elements(By.CLASS_NAME, 'price') else "N/A",
                    'Special Price': product.find_element(By.CLASS_NAME, 'spacail-price').text.strip() if product.find_elements(By.CLASS_NAME, 'spacail-price') else "N/A",
                    'Discount': product.find_element(By.CLASS_NAME, 'discount').text.strip() if product.find_elements(By.CLASS_NAME, 'discount') else "N/A",
                    'Image URL': product.find_element(By.CSS_SELECTOR, 'img').get_attribute('data-src') if product.find_elements(By.CSS_SELECTOR, 'img[data-src]') else "N/A"
                })

            progress_bar.progress(current_page / max_pages)
            progress_text.text(f"Scraping page {current_page} of {max_pages}...")
            

            try:
                next_button = driver.find_element(By.CSS_SELECTOR, 'ul.pagination li a[rel="next"]')
                next_button.click()
                time.sleep(2)
                current_page += 1
            except Exception as e:
                st.warning("No more pages or 'Next' button not found.")
                break
        
        progress_text.empty()
        return product_data, current_page
        
    except Exception as e:
        st.error(f"Error: {e}")
        return [], 0

def search_store_products(store_name, search_query):
    product_data = []

    encoded_query = requests.utils.quote(search_query)
    search_url = f"https://dealsheaven.in/store/{store_name}?keyword={encoded_query}"
    
    try:
        driver.get(search_url)
        time.sleep(2)
        
        products = driver.find_elements(By.CSS_SELECTOR, '.product-item-detail')

        if not products:
            search_url = f"https://dealsheaven.in/search?keyword={encoded_query}&store={store_name}"
            driver.get(search_url)
            time.sleep(2)
            products = driver.find_elements(By.CSS_SELECTOR, '.product-item-detail')

        for product in products:
            product_name = product.find_element(By.TAG_NAME, 'h3')
            if product_name and search_query.lower() in product_name.text.strip().lower():
                product_name = product_name.text.strip()
                current_price = product.find_element(By.CLASS_NAME, 'price').text.strip() if product.find_elements(By.CLASS_NAME, 'price') else "N/A"
                special_price = product.find_element(By.CLASS_NAME, 'spacail-price').text.strip() if product.find_elements(By.CLASS_NAME, 'spacail-price') else "N/A"
                discount = product.find_element(By.CLASS_NAME, 'discount').text.strip() if product.find_elements(By.CLASS_NAME, 'discount') else "N/A"
                image_url = product.find_element(By.CSS_SELECTOR, 'img').get_attribute('data-src') if product.find_elements(By.CSS_SELECTOR, 'img[data-src]') else "N/A"
                product_url = product.find_element(By.TAG_NAME, 'a').get_attribute('href') if product.find_elements(By.TAG_NAME, 'a') else "N/A"

                product_data.append({
                    'Product Name': product_name,
                    'Current Price': current_price,
                    'Special Price': special_price,
                    'Discount': discount,
                    'Image URL': image_url,
                    'Product URL': product_url
                })
        
        return product_data
        
    except Exception as e:
        st.error(f"Error during search: {e}")
        return []


def main():
    st.set_page_config(page_title="Store Deals Scraper", page_icon="🛍️", layout="wide")

    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        .main-header {
            font-size: 2.8rem;
            background: linear-gradient(45deg, #1e88e5, #1565C0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🛍️ Store Deals Scraper</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-title">⚙️ Scraping Options</div>', unsafe_allow_html=True)
        
        store_data = scrape_stores()
        store_names = [store['name'] for store in store_data]
        
        selected_store = st.selectbox("📍 Select Store:", store_names)
        search_query = st.text_input("🔍 Search Product (optional):")
        num_pages = st.number_input("📄 Number of Pages to Scrape:", min_value=1, max_value=20, value=3)

        estimated_time = (num_pages * 1.5) / min(num_pages, 5)
        st.info(f"⏱️ Estimated time: ~{estimated_time:.1f} seconds")

        if search_query:
            start_search = st.button("🔍 Search Products")
        start_scraping = st.button("🚀 Start Scraping")

    if search_query and 'start_search' in locals() and start_search:
        store_slug = [store['url'].split('/')[-1] for store in store_data if store['name'] == selected_store][0]
        
        with st.spinner(f"Searching {selected_store} for '{search_query}'..."):
            product_data = search_store_products(store_slug, search_query)
            
        if product_data:
            st.success(f"✅ Found {len(product_data)} products matching '{search_query}' in {selected_store}")
            df = pd.DataFrame(product_data)
            st.dataframe(df, use_container_width=True, height=400)

            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download CSV", csv, file_name=f"{selected_store}_search_results.csv")
            
            with col2:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)
                st.download_button("📊 Download Excel", excel_buffer, file_name=f"{selected_store}_search_results.xlsx")
            
            with col3:
                json_str = df.to_json(orient='records', indent=2)
                st.download_button("🔄 Download JSON", json_str, file_name=f"{selected_store}_search_results.json")
                
        else:
            st.warning(f"No products found matching '{search_query}' in {selected_store}")

    elif start_scraping:
        store_slug = [store['url'].split('/')[-1] for store in store_data if store['name'] == selected_store][0]
        
        with st.spinner(f"Scraping {selected_store}..."):
            start_time = datetime.now()
            product_data, available_pages = scrape_store_deals(store_slug, num_pages)
            end_time = datetime.now()
            
        if product_data:
            scraping_time = (end_time - start_time).total_seconds()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Products", len(product_data))
            
            with col2:
                st.metric("Pages Scraped", num_pages)
            
            with col3:
                st.metric("Scraping Time", f"{scraping_time:.1f}s")

            df = pd.DataFrame(product_data)
            st.dataframe(df, use_container_width=True, height=400)

            col1, col2, col3 = st.columns(3)
            
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📄 Download CSV", csv, file_name=f"{selected_store}_data.csv")
            
            with col2:
                excel_buffer = io.BytesIO()
                df.to_excel(excel_buffer, index=False)
                excel_buffer.seek(0)
                st.download_button("📊 Download Excel", excel_buffer, file_name=f"{selected_store}_data.xlsx")
            
            with col3:
                json_str = df.to_json(orient='records', indent=2)
                st.download_button("🔄 Download JSON", json_str, file_name=f"{selected_store}_data.json")

        else:
            st.error(f"No products found for {selected_store}.")

if __name__ == "__main__":
    main()
