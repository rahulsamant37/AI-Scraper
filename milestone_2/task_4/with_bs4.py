import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import json
import io
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


@st.cache_data(ttl=3600)
def scrape_stores():
    url = "https://dealsheaven.in/stores"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    store_data = [{"name": a.text.strip(), "url": a['href'].strip()} for a in soup.select("ul.store-listings li a")]
    return store_data

def scrape_single_page(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        products = soup.select('.product-item-detail')
        
        page_data = []
        for product in products:
            product_data = {
                'Product Name': product.select_one('h3').text.strip() if product.select_one('h3') else "N/A",
                'Current Price': product.select_one('.price').text.strip() if product.select_one('.price') else "N/A",
                'Special Price': product.select_one('.spacail-price').text.strip() if product.select_one('.spacail-price') else "N/A",
                'Discount': product.select_one('.discount').text.strip() if product.select_one('.discount') else "N/A",
                'Image URL': product.select_one('img[data-src]')['data-src'] if product.select_one('img[data-src]') else "N/A"
            }
            page_data.append(product_data)
        return page_data
    except Exception as e:
        st.error(f"Error scraping page: {e}")
        return []

def scrape_store_deals(store_name, max_pages):
    try:
        url = f"https://dealsheaven.in/store/{store_name}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        pagination = soup.select('ul.pagination li')
        available_pages = int(pagination[-2].text) if len(pagination) > 1 else 1
        pages_to_scrape = min(max_pages, available_pages)
        
        urls = [f"https://dealsheaven.in/store/{store_name}?page={page}" for page in range(1, pages_to_scrape + 1)]
        product_data = []
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        with ThreadPoolExecutor(max_workers=min(pages_to_scrape, 5)) as executor:
            future_to_url = {executor.submit(scrape_single_page, url): i 
                           for i, url in enumerate(urls, 1)}
            
            completed = 0
            for future in as_completed(future_to_url):
                completed += 1
                page_num = future_to_url[future]
                progress_bar.progress(completed / pages_to_scrape)
                progress_text.text(f"Scraping page {page_num} of {pages_to_scrape}...")
                page_data = future.result()
                product_data.extend(page_data)
        
        progress_text.empty()
        return product_data, available_pages
        
    except Exception as e:
        st.error(f"Error: {e}")
        return [], 0

def search_store_products(store_name, search_query):
    product_data = []

    encoded_query = requests.utils.quote(search_query)
    
    search_url = f"https://dealsheaven.in/store/{store_name}?keyword={encoded_query}"
    
    try:
        response = requests.get(search_url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        products = soup.select('.product-item-detail')
        
        if not products:
            search_url = f"https://dealsheaven.in/search?keyword={encoded_query}&store={store_name}"
            response = requests.get(search_url)
            soup = BeautifulSoup(response.content, "html.parser")
            products = soup.select('.product-item-detail')

        for product in products:
            product_name = product.select_one('h3')
            if product_name and search_query.lower() in product_name.text.strip().lower():
                product_name = product_name.text.strip()
                current_price = product.select_one('.price')
                current_price = current_price.text.strip() if current_price else "N/A"
                
                special_price = product.select_one('.spacail-price')
                special_price = special_price.text.strip() if special_price else "N/A"
                
                discount = product.select_one('.discount')
                discount = discount.text.strip() if discount else "N/A"
                
                image = product.select_one('img[data-src]')
                image_url = image['data-src'] if image else "N/A"

                product_link = product.select_one('a')
                product_url = product_link['href'] if product_link else "N/A"
                
                product_data.append({
                    'Product Name': product_name,
                    'Current Price': current_price,
                    'Special Price': special_price,
                    'Discount': discount,
                    'Image URL': image_url,
                    'Product URL': product_url
                })
                
        if not product_data:
            print(f"No matching products found for '{search_query}' in '{store_name}'")
            return []
            
        print(f"Found {len(product_data)} matching products for '{search_query}' in '{store_name}'")
        return product_data
        
    except Exception as e:
        print(f"Error during search: {e}")
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
        .stat-card {
            background: white;
            padding: 1.5rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background: linear-gradient(45deg, #1e88e5, #1565C0);
            color: white;
            font-size: 1.2rem;
            padding: 0.8rem 2rem;
            border-radius: 8px;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(30, 136, 229, 0.4);
        }
        .stProgress>div>div {
            background-color: #2962FF;
        }
        .sidebar-title {
            font-size: 1.8rem !important;
            font-weight: bold !important;
            background: linear-gradient(45deg, #1e88e5, #1565C0);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            padding: 0.5rem 0 !important;
            margin-bottom: 1rem !important;
            text-align: center !important;
            border-bottom: 2px solid #1e88e5 !important;
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
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Total Products", len(product_data))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Pages Scraped", num_pages)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                st.metric("Scraping Time", f"{scraping_time:.1f}s")
                st.markdown('</div>', unsafe_allow_html=True)

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