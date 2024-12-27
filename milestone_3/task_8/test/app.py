import streamlit as st
from streamlit_tags import st_tags_sidebar
import pandas as pd
import json
import threading
import sys
import sqlite3
import asyncio
import os
import logging
from datetime import datetime
from typing import List, Dict, Type, Any
from scraper import fetch_html_playwright, save_raw_data, format_data, save_formatted_data, calculate_price, html_to_markdown_with_readability, create_dynamic_listing_model, create_listings_container_model, get_text_chunks

from assets import PRICING

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

class WebScraperDatabase:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, db_path: str = 'web_scraper_data.db'):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(WebScraperDatabase, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, db_path: str = 'web_scraper_data.db'):
        if self._initialized:
            return
        
        self.db_path = db_path
        self._local = threading.local()
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        
        self._create_connection()
        self._create_tables()
        self._initialized = True

    def _create_connection(self):
        try:
            # Use thread-local storage for database connections
            if not hasattr(self._local, 'conn'):
                self._local.conn = sqlite3.connect(self.db_path, 
                                                   check_same_thread=False, 
                                                   isolation_level=None)  # Auto-commit mode
                self._local.conn.execute('pragma journal_mode=wal')  # Improve concurrent access
                self._local.conn.execute('PRAGMA foreign_keys = ON')  # Enable foreign key support
            
            self.conn = self._local.conn
            self.cursor = self.conn.cursor()
            print(f"Connected to SQLite database at {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def _create_tables(self):
        """
        Create necessary tables for storing web scraping data
        Safe method to create tables if they don't exist
        """
        try:
            # Table for storing raw scraping metadata
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS scrape_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    raw_data TEXT,
                    markdown_data TEXT,
                    model_used TEXT
                )
            ''')

            # Table for storing formatted listing data 
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS listings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scrape_id INTEGER,
                    data JSON,
                    FOREIGN KEY (scrape_id) REFERENCES scrape_metadata(id)
                )
            ''')

            # Table for storing token and cost information
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS token_costs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scrape_id INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_cost REAL,
                    model_name TEXT,
                    FOREIGN KEY (scrape_id) REFERENCES scrape_metadata(id)
                )
            ''')

            self.conn.commit()
            print("Database tables created successfully")
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
            self.conn.rollback()
            raise

    def insert_scrape_data(self, url, raw_data, markdown_data, model_used, formatted_data, token_counts, total_cost):
        """
        Insert scrape metadata and associated data into the database

        :param url: URL that was scraped
        :param raw_data: Raw HTML data (optional)
        :param markdown_data: Markdown converted content
        :param model_used: AI model used for extraction
        :param formatted_data: Extracted and formatted data
        :param token_counts: Dictionary with input and output tokens
        :param total_cost: Total cost of the API call
        :return: ID of the inserted scrape metadata
        """
        try:
            # Insert scrape metadata
            self.cursor.execute('''
                INSERT INTO scrape_metadata 
                (url, raw_data, markdown_data, model_used) 
                VALUES (?, ?, ?, ?)
            ''', (url, raw_data, markdown_data, model_used))

            # Get the ID of the last inserted scrape metadata
            scrape_id = self.cursor.lastrowid

        except sqlite3.Error as e:
            print(f"Error inserting scrape metadata: {e}")
            self.conn.rollback()
            raise

        try:
            # Insert token and cost information
            self.cursor.execute('''
                INSERT INTO token_costs 
                (scrape_id, input_tokens, output_tokens, total_cost, model_name) 
                VALUES (?, ?, ?, ?, ?)
            ''', (
                scrape_id, 
                token_counts.get('input_tokens', 0), 
                token_counts.get('output_tokens', 0), 
                total_cost, 
                model_used
            ))

            # Insert listings data if available
            if hasattr(formatted_data, 'listings'):
                for listing in formatted_data.listings:
                    self.cursor.execute('''
                        INSERT INTO listings 
                        (scrape_id, data) 
                        VALUES (?, ?)
                    ''', (scrape_id, json.dumps(listing)))

            # Commit all database changes
            self.conn.commit()
            print(f"Successfully inserted scrape data with ID: {scrape_id}")

            return scrape_id

        except sqlite3.Error as e:
            print(f"Error inserting detailed scrape data: {e}")
            self.conn.rollback()
            raise
        
    def retrieve_scrapes(self, limit=50, model_filter=None, start_date=None, end_date=None):
        """
        Retrieve scrape metadata with optional filtering

        :param limit: Maximum number of scrapes to retrieve
        :param model_filter: Filter by specific model
        :param start_date: Filter scrapes from this date
        :param end_date: Filter scrapes up to this date
        :return: List of scrape metadata
        """
        try:
            query = '''
                SELECT 
                    sm.id, 
                    sm.url, 
                    sm.timestamp, 
                    sm.model_used, 
                    tc.input_tokens, 
                    tc.output_tokens, 
                    tc.total_cost
                FROM scrape_metadata sm
                JOIN token_costs tc ON sm.id = tc.scrape_id
                WHERE 1=1
            '''
            params = []

            if model_filter:
                query += ' AND sm.model_used = ?'
                params.append(model_filter)

            if start_date:
                query += ' AND sm.timestamp >= ?'
                params.append(start_date)

            if end_date:
                query += ' AND sm.timestamp <= ?'
                params.append(end_date)

            query += ' ORDER BY sm.timestamp DESC LIMIT ?'
            params.append(limit)

            self.cursor.execute(query, params)
            return self.cursor.fetchall()

        except sqlite3.Error as e:
            print(f"Error retrieving scrapes: {e}")
            return []

    def get_listings_for_scrape(self, scrape_id):
        """
        Retrieve listings for a specific scrape

        :param scrape_id: ID of the scrape
        :return: List of listings
        """
        try:
            self.cursor.execute('''
                SELECT data 
                FROM listings 
                WHERE scrape_id = ?
            ''', (scrape_id,))

            # Parse JSON data back to dictionaries
            listings = [json.loads(row[0]) for row in self.cursor.fetchall()]
            return listings

        except sqlite3.Error as e:
            print(f"Error retrieving listings: {e}")
            return []

    def close_connection(self):
        """
        Safely close the database connection
        """
        try:
            if hasattr(self._local, 'conn'):
                self._local.conn.close()
                del self._local.conn
                print("Database connection closed")
        except Exception as e:
            print(f"Error closing database: {e}")

    def __del__(self):
        """
        Ensure database connection is closed when object is deleted
        """
        self.close_connection()

def integrate_database_with_scraper(url, markdown, formatted_data, token_counts, total_cost, model):
    """
    Integrate database storage with the existing web scraping workflow
    
    :param url: URL being scraped
    :param markdown: Markdown content
    :param formatted_data: Extracted data
    :param token_counts: Token usage
    :param total_cost: API call cost
    :param model: Model used
    :return: Scrape ID
    """
    try:
        db = WebScraperDatabase()
        scrape_id = db.insert_scrape_data(
            url=url,
            raw_data=None,  # You can pass raw HTML here if needed
            markdown_data=markdown,
            model_used=model,
            formatted_data=formatted_data,
            token_counts=token_counts,
            total_cost=total_cost
        )
        return scrape_id
    except Exception as e:
        print(f"Database integration error: {e}")
        raise

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

# Database interactions tab
tab1, tab2, tab3 = st.tabs(["Scraper", "Database View", "Scrape History"])

with tab1:
    st.sidebar.title("Web Scraper Settings")
    selected_model = st.sidebar.selectbox("Select Model", options=list(PRICING.keys()), index=0)
    logger.info(f"Model selected: {selected_model}")

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

    def process_chunks(chunks, DynamicListingsContainer, DynamicListingModel, selected_model):
        logger = logging.getLogger(__name__)
        all_listings = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            try:
                # Add print statements for debugging
                print(f"Chunk {i} content: {chunk[:200]}...")
                # Attempt to format data for each chunk
                formatted_data, token_counts = format_data(
                    chunk, 
                    DynamicListingsContainer, 
                    DynamicListingModel, 
                    selected_model
                )
                
                # Add print statements to verify data
                print(f"Formatted Data for Chunk {i}: {formatted_data}")
                print(f"Token Counts: {token_counts}")

                # Check if formatted_data has listings
                if formatted_data and hasattr(formatted_data, 'listings'):
                    all_listings.extend(formatted_data.listings)

                # Accumulate token counts and calculate cost
                input_tokens = token_counts.get('input_tokens', 0)
                output_tokens = token_counts.get('output_tokens', 0)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens

            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                # Print full traceback
                import traceback
                traceback.print_exc()
                continue
            
        # Create a new container with all collected listings
        final_result = DynamicListingsContainer(listings=all_listings)

        return final_result, total_input_tokens, total_output_tokens


    def perform_scrape(url_input, fields, selected_model):
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
            # Split markdown into chunks
            chunks = get_text_chunks(markdown)
            logger.info(f"Splitting text into chunks (size: 10000, overlap: 1000)")
            logger.info(f"Processing {len(chunks)} chunks")

            # Process chunks
            formatted_data, input_tokens, output_tokens = process_chunks(
                chunks, 
                DynamicListingsContainer, 
                DynamicListingModel, 
                selected_model
            )

            # Save formatted data
            df = save_formatted_data(formatted_data, timestamp)

            # Calculate cost
            total_cost = calculate_price(
                {"input_tokens": input_tokens, "output_tokens": output_tokens}, 
                selected_model
            )[-1]
            # Integrate with database
            try:
                scrape_id = integrate_database_with_scraper(
                    url_input, 
                    markdown, 
                    formatted_data, 
                    {"input_tokens": input_tokens, "output_tokens": output_tokens}, 
                    total_cost, 
                    selected_model
                )
            except Exception as e:
                logger.error(f"Database integration failed: {e}")
                scrape_id = None
            # Merge listings to remove duplicates
            if hasattr(formatted_data, 'listings'):
                formatted_data.listings = merge_listings(formatted_data.listings)
            # Return all necessary information
            results = (df, formatted_data, markdown, input_tokens, output_tokens, total_cost, timestamp, scrape_id)
            return results

        except Exception as e:
            logger.error(f"Comprehensive scraping error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


    if 'perform_scrape' not in st.session_state:
        st.session_state['perform_scrape'] = False
        logger.info("Initialized session state")

    if st.sidebar.button("Scrape"):
        # Clear any previous results
        st.session_state['results'] = None
        st.session_state['error'] = None

        # Show loading spinner
        with st.spinner("Scraping in progress... 🕸️"):
            try:
                results = perform_scrape(url_input, fields, selected_model)
                if results:
                    st.session_state['results'] = results
                    st.session_state['error'] = None
                else:
                    st.session_state['error'] = "Scraping did not return any results"
            except Exception as e:
                st.session_state['error'] = f"Scraping failed: {str(e)}"
                logger.error(f"Scraping error: {e}")

    # Display results or error message
    if st.session_state.get('error'):
        st.error(st.session_state['error'])

    if st.session_state.get('results'):
        logger.info("Displaying scraping results")
        df, formatted_data, markdown, input_tokens, output_tokens, total_cost, timestamp, scrape_id = st.session_state['results']

        st.subheader("Scraping Results")

        # Detailed Listings Display
        if formatted_data and hasattr(formatted_data, 'listings'):
            st.write(f"**Total Listings Found:** {len(formatted_data.listings)}")

            # Expandable section for all listings
            with st.expander("View All Listings"):
                for idx, listing in enumerate(formatted_data.listings, 1):
                    st.markdown(f"### Listing {idx}")
                    for key, value in listing.items():
                        st.text(f"{key.replace('_', ' ').title()}: {value}")
                    st.markdown("---")

            # Table view of listings using st.dataframe with better formatting
            st.subheader("Listings Table View")
            listings_df = pd.DataFrame(formatted_data.listings)

            # Enhanced dataframe display
            st.dataframe(
                listings_df, 
                use_container_width=True,  # Make table responsive
                hide_index=True  # Hide index column
            )

        # Token Usage and Cost Display
        st.sidebar.markdown("## Token Usage")
        st.sidebar.markdown(f"**Input Tokens:** {input_tokens}")
        st.sidebar.markdown(f"**Output Tokens:** {output_tokens}")
        st.sidebar.markdown(f"**Total Cost:** :green-background[***${total_cost:.4f}***]")

        logger.info(f"Token usage - Input: {input_tokens}, Output: {output_tokens}, Cost: ${total_cost:.4f}")

        # Download Options
        col1, col2, col3 = st.columns(3)

        with col1:
            # Convert custom object to dictionary before JSON serialization
            json_data = json.dumps({
                'listings': [listing.__dict__ for listing in formatted_data.listings]
            }, indent=4)
            st.download_button("Download JSON", data=json_data, file_name=f"{timestamp}_data.json")
            logger.info("JSON download button created")
        with col2:
            try:
                # Create DataFrame from listings
                listings_df = pd.DataFrame([listing.__dict__ for listing in formatted_data.listings])
                st.download_button("Download CSV", data=listings_df.to_csv(index=False), file_name=f"{timestamp}_data.csv")
                logger.info("CSV download button created")
            except Exception as e:
                logger.error(f"Error creating CSV download button: {str(e)}")
        with col3:
            st.download_button("Download Markdown", data=markdown, file_name=f"{timestamp}_data.md")
            logger.info("Markdown download button created")

with tab2:
    st.header("Database Interaction")
    
    # Database operations
    db = WebScraperDatabase()
    
    # Model filter
    model_filter = st.selectbox("Filter by Model", 
        options=['All'] + list(PRICING.keys()), 
        index=0
    )
    
    # Date range filter
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None)
    with col2:
        end_date = st.date_input("End Date", value=None)
    
    # Retrieve scrapes based on filters
    if st.button("Retrieve Scrapes"):
        try:
            # Prepare filters
            model = model_filter if model_filter != 'All' else None
            start = start_date.strftime('%Y-%m-%d') if start_date else None
            end = end_date.strftime('%Y-%m-%d') if end_date else None
            
            # Retrieve scrapes
            scrapes = db.retrieve_scrapes(
                limit=50, 
                model_filter=model, 
                start_date=start, 
                end_date=end
            )
            
            # Display results
            if scrapes:
                scrape_df = pd.DataFrame(scrapes, columns=[
                    'Scrape ID', 'URL', 'Timestamp', 'Model', 
                    'Input Tokens', 'Output Tokens', 'Total Cost'
                ])
                st.dataframe(scrape_df)
                
                # Allow selecting a specific scrape to view its listings
                selected_scrape = st.selectbox(
                    "Select a Scrape to View Listings", 
                    options=[f"Scrape ID: {scrape[0]}" for scrape in scrapes]
                )
                
                if selected_scrape:
                    scrape_id = int(selected_scrape.split(":")[1].strip())
                    listings = db.get_listings_for_scrape(scrape_id)
                    
                    if listings:
                        listings_df = pd.DataFrame(listings)
                        st.write("Listings for Selected Scrape:")
                        st.dataframe(listings_df)
                        
                        # Option to download listings
                        st.download_button(
                            "Download Listings CSV", 
                            data=listings_df.to_csv(index=False), 
                            file_name=f"scrape_{scrape_id}_listings.csv"
                        )
            else:
                st.warning("No scrapes found matching the selected criteria.")
        
        except Exception as e:
            st.error(f"Error retrieving scrapes: {e}")

with tab3:
    st.header("Scrape History Visualization")
    
    # Aggregate cost and token usage
    try:
        db = WebScraperDatabase()
        scrapes = db.retrieve_scrapes(limit=100)
        
        if scrapes:
            history_df = pd.DataFrame(scrapes, columns=[
                'Scrape ID', 'URL', 'Timestamp', 'Model', 
                'Input Tokens', 'Output Tokens', 'Total Cost'
            ])
            
            # Cost analysis
            st.subheader("Total Cost Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Accumulated Cost", f"${history_df['Total Cost'].sum():.2f}")
            
            with col2:
                st.metric("Average Cost per Scrape", f"${history_df['Total Cost'].mean():.4f}")
            
            # Model-wise cost distribution
            st.subheader("Cost Distribution by Model")
            model_costs = history_df.groupby('Model')['Total Cost'].sum()
            st.bar_chart(model_costs)
            
            # Token usage
            st.subheader("Token Usage")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Input Tokens", f"{history_df['Input Tokens'].sum():,}")
            
            with col2:
                st.metric("Total Output Tokens", f"{history_df['Output Tokens'].sum():,}")
            
            # Model-wise token usage
            st.subheader("Token Usage by Model")
            input_tokens = history_df.groupby('Model')['Input Tokens'].sum()
            output_tokens = history_df.groupby('Model')['Output Tokens'].sum()
            
            token_usage_df = pd.DataFrame({
                'Input Tokens': input_tokens,
                'Output Tokens': output_tokens
            })
            st.bar_chart(token_usage_df)
        
        else:
            st.warning("No scrape history available.")
    
    except Exception as e:
        st.error(f"Error visualizing scrape history: {e}")

# Close database connection when app is done
if 'db' in locals():
    db.close_connection()