import os
import random
import asyncio
import sys
import platform
import time
import sqlite3
import re
import json
import threading
import logging
from datetime import datetime
from typing import List, Dict, Type, Any

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text
import tiktoken

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, Page

from openai import OpenAI
import google.generativeai as genai
from groq import Groq

from langchain.text_splitter import RecursiveCharacterTextSplitter
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")

from assets import USER_AGENTS,PRICING,HEADLESS_OPTIONS,SYSTEM_MESSAGE,USER_MESSAGE,LLAMA_MODEL_FULLNAME,GROQ_LLAMA_MODEL_FULLNAME
load_dotenv()
def setup_logging(timestamp):
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'scraping_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

def setup_playwright():
    logger = logging.getLogger(__name__)
    playwright = browser = context = page = None
    
    try:
        try:
            playwright = sync_playwright().start()
        except Exception as e:
            logger.error(f"Failed to start Playwright: {str(e)}")
            raise RuntimeError(f"Playwright initialization failed: {str(e)}")
        
        try:
            browser = playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-gpu']
            )
        except Exception as e:
            logger.error(f"Failed to launch browser: {str(e)}")
            if playwright:
                playwright.stop()
            raise RuntimeError(f"Browser launch failed: {str(e)}")
        
        try:
            context = browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={'width': 1920, 'height': 1080}
            )
            context.set_default_timeout(30000)
            context.set_default_navigation_timeout(30000)
            page = context.new_page()
            logger.info("Playwright setup completed successfully")
            return playwright, browser, context, page
            
        except Exception as e:
            logger.error(f"Failed to create context or page: {str(e)}")
            if browser:
                browser.close()
            if playwright:
                playwright.stop()
            raise RuntimeError(f"Context/page creation failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in setup_playwright: {str(e)}")
        try:
            if page:
                page.close()
            if context:
                context.close()
            if browser:
                browser.close()
            if playwright:
                playwright.stop()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")
        raise RuntimeError(f"Failed to setup Playwright: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in setup_playwright: {str(e)}")
        if context:
            context.close()
        if browser:
            browser.close()
        if playwright:
            playwright.stop()
        raise RuntimeError(f"Failed to setup Playwright: {str(e)}")

def click_accept_cookies(page: Page):
    """Enhanced cookie consent handling with better error recovery"""
    logger = logging.getLogger(__name__)
    try:
        selectors = [
            "text=accept",
            "text=agree",
            "text=allow",
            "text=consent",
            "text=continue",
            "text=ok",
            "text=got it",
            "[aria-label*='accept' i]",
            "[aria-label*='cookie' i]",
            "#cookie-consent button",
            ".cookie-banner button",
        ]
        
        for selector in selectors:
            try:
                button = page.locator(selector).first
                if button.count() > 0:
                    button.click(timeout=5000)
                    logger.info(f"Clicked cookie consent using selector: {selector}")
                    return
            except Exception:
                continue
        
        logger.warning("No cookie consent button found")
    
    except Exception as e:
        logger.warning(f"Non-critical error handling cookies: {str(e)}")

def scroll_and_wait(page: Page) -> None:
    """Enhanced scrolling with better error handling"""
    logger = logging.getLogger(__name__)
    try:
        last_height = page.evaluate("document.body.scrollHeight")
        retries = 3
        
        while retries > 0:
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1000)
                
                new_height = page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    break
                    
                last_height = new_height
                retries = 3
            except Exception as scroll_error:
                logger.warning(f"Scroll attempt failed: {str(scroll_error)}")
                retries -= 1
                page.wait_for_timeout(1000)
        
        logger.info("Page scrolling completed")
        
    except Exception as e:
        logger.warning(f"Non-critical scrolling error: {str(e)}")

def fetch_html_playwright(url: str) -> str:
    logger = logging.getLogger(__name__)
    playwright = browser = context = page = None
    
    try:
        playwright, browser, context, page = setup_playwright()
        logger.info(f"Fetching HTML from URL: {url}")
        try:
            response = page.goto(url, wait_until='networkidle')
            if not response:
                raise RuntimeError("Navigation failed: no response received")
            if response.status >= 400:
                raise RuntimeError(f"Navigation failed: HTTP {response.status}")
        except Exception as nav_error:
            logger.error(f"Navigation failed: {str(nav_error)}")
            raise RuntimeError(f"Failed to navigate to URL: {str(nav_error)}")
        try:
            click_accept_cookies(page)
        except Exception as cookie_error:
            logger.warning(f"Cookie handling error (non-critical): {str(cookie_error)}")
        scroll_and_wait(page)
        html = page.content()
        logger.info("HTML content successfully fetched")
        return html
        
    except Exception as e:
        logger.error(f"Error fetching HTML: {str(e)}")
        raise RuntimeError(f"Failed to fetch HTML: {str(e)}")
        
    finally:
        try:
            if page:
                page.close()
            if context:
                context.close()
            if browser:
                browser.close()
            if playwright:
                playwright.stop()
            logger.info("Playwright session closed properly")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")

def clean_html(html_content):
    logger = logging.getLogger(__name__)
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        for element in soup.find_all(['header', 'footer']):
            element.decompose()
        logger.info("HTML content cleaned successfully")
        return str(soup)
    except Exception as e:
        logger.error(f"Error cleaning HTML: {e}")
        raise

def html_to_markdown_with_readability(html_content):
    logger = logging.getLogger(__name__)
    try:
        cleaned_html = clean_html(html_content)
        markdown_converter = html2text.HTML2Text()
        markdown_converter.ignore_links = False
        markdown_content = markdown_converter.handle(cleaned_html)
        logger.info("HTML successfully converted to markdown")
        return markdown_content
    except Exception as e:
        logger.error(f"Error converting HTML to markdown: {e}")
        raise

def save_raw_data(raw_data, timestamp, output_folder='output'):
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(output_folder, exist_ok=True)
        raw_output_path = os.path.join(output_folder, f'rawData_{timestamp}.md')
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            f.write(raw_data)
        logger.info(f"Raw data saved to {raw_output_path}")
        return raw_output_path
    except Exception as e:
        logger.error(f"Error saving raw data: {e}")
        raise

def remove_urls_from_file(file_path):
    logger = logging.getLogger(__name__)
    try:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        base, ext = os.path.splitext(file_path)
        new_file_path = f"{base}_cleaned{ext}"
        with open(file_path, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
        cleaned_content = re.sub(url_pattern, '', markdown_content)
        with open(new_file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        logger.info(f"URLs removed and cleaned file saved as: {new_file_path}")
        return cleaned_content
    except Exception as e:
        logger.error(f"Error removing URLs from file: {e}")
        raise

def create_dynamic_listing_model(field_names: List[str]) -> Type[BaseModel]:
    logger = logging.getLogger(__name__)
    try:
        field_definitions = {field: (str, ...) for field in field_names}
        model = create_model('DynamicListingModel', **field_definitions)
        logger.info(f"Dynamic listing model created with fields: {field_names}")
        return model
    except Exception as e:
        logger.error(f"Error creating dynamic listing model: {e}")
        raise

def create_listings_container_model(listing_model: Type[BaseModel]) -> Type[BaseModel]:
    logger = logging.getLogger(__name__)
    try:
        container_model = create_model('DynamicListingsContainer', listings=(List[listing_model], ...))
        logger.info("Listings container model created")
        return container_model
    except Exception as e:
        logger.error(f"Error creating listings container model: {e}")
        raise


def get_text_chunks(cleaned_content):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(cleaned_content)
    return chunks


def generate_system_message(listing_model: BaseModel) -> str:
    logger = logging.getLogger(__name__)
    try:
        schema_info = listing_model.model_json_schema()
        field_descriptions = []
        for field_name, field_info in schema_info["properties"].items():
            field_type = field_info["type"]
            field_descriptions.append(f'"{field_name}": "{field_type}"')
        schema_structure = ",\n".join(field_descriptions)
        system_message = f"""
        You are an intelligent text extraction and conversion assistant. Your task is to extract structured information 
                            from the given text and convert it into a pure JSON format. The JSON should contain only the structured data extracted from the text, 
                            with no additional commentary, explanations, or extraneous information. 
                            You could encounter cases where you can't find the data of the fields you have to extract or the data will be in a foreign language.
                            Please process the following text and provide the output in pure JSON format with no words before or after the JSON:
        Please ensure the output strictly follows this schema:

        {{
            "listings": [
                {{
                    {schema_structure}
                }}
            ]
        }} """
        logger.info("System message generated successfully")
        return system_message
    except Exception as e:
        logger.error(f"Error generating system message: {e}")
        raise

def format_data(data, DynamicListingsContainer, DynamicListingModel, selected_model):
    logger = logging.getLogger(__name__)
    token_counts = {}
    
    try:
        logger.info(f"Formatting data using model: {selected_model}")
        
        if selected_model == "whisper-large-v3":
            sys_message = generate_system_message(DynamicListingModel)
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": USER_MESSAGE + data}
                ],
                model=GROQ_LLAMA_MODEL_FULLNAME,
            )
            response_content = completion.choices[0].message.content
            logger.info("Groq Llama model response received")
            
            try:
                parsed_response = json.loads(response_content)
                logger.info("Response successfully parsed as JSON")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return None, token_counts
                
            token_counts = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens
            }
            logger.info(f"Groq API request completed. Input tokens: {completion.usage.prompt_tokens}, Output tokens: {completion.usage.completion_tokens}")
            return parsed_response, token_counts

        elif selected_model == "gemini-1.5-flash":
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-flash',
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": DynamicListingsContainer
                    })
            prompt = SYSTEM_MESSAGE + "\n" + USER_MESSAGE + data
            input_tokens = model.count_tokens(prompt)
            completion = model.generate_content(prompt)
            usage_metadata = completion.usage_metadata
            token_counts = {
                "input_tokens": usage_metadata.prompt_token_count,
                "output_tokens": usage_metadata.candidates_token_count
            }
            logger.info(f"Gemini API request completed. Input tokens: {usage_metadata.prompt_token_count}, Output tokens: {usage_metadata.candidates_token_count}")
            return completion.text, token_counts
        
        elif selected_model == "Llama3.1 8B":
            sys_message = generate_system_message(DynamicListingModel)
            client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

            completion = client.chat.completions.create(
                model=LLAMA_MODEL_FULLNAME,
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": USER_MESSAGE + data}
                ],
                temperature=0.7,
            )
            response_content = completion.choices[0].message.content
            logger.info("Llama model response received")
            
            try:
                parsed_response = json.loads(response_content)
                logger.info("Response successfully parsed as JSON")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return None, token_counts
                
            token_counts = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens
            }
            logger.info(f"Llama API request completed. Input tokens: {completion.usage.prompt_tokens}, Output tokens: {completion.usage.completion_tokens}")
            return parsed_response, token_counts
            
        elif selected_model == "Groq Llama3.1 70b":
            sys_message = generate_system_message(DynamicListingModel)
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

            completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": sys_message},
                    {"role": "user", "content": USER_MESSAGE + data}
                ],
                model=GROQ_LLAMA_MODEL_FULLNAME,
            )
            response_content = completion.choices[0].message.content
            logger.info("Groq Llama model response received")
            
            try:
                parsed_response = json.loads(response_content)
                logger.info("Response successfully parsed as JSON")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                return None, token_counts
                
            token_counts = {
                "input_tokens": completion.usage.prompt_tokens,
                "output_tokens": completion.usage.completion_tokens
            }
            logger.info(f"Groq API request completed. Input tokens: {completion.usage.prompt_tokens}, Output tokens: {completion.usage.completion_tokens}")
            return parsed_response, token_counts
            
        else:
            error_msg = f"Unsupported model: {selected_model}"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
    except Exception as e:
        logger.error(f"Error in format_data: {e}")
        raise

def save_formatted_data(formatted_data, timestamp, output_folder='output'):
    logger = logging.getLogger(__name__)
    try:
        os.makedirs(output_folder, exist_ok=True)
        if isinstance(formatted_data, str):
            try:
                formatted_data_dict = json.loads(formatted_data)
                logger.info("Successfully parsed string data as JSON")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {e}")
                raise ValueError("The provided formatted data is a string but not valid JSON.")
        else:
            formatted_data_dict = formatted_data.dict() if hasattr(formatted_data, 'dict') else formatted_data
        
        json_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data_dict, f, indent=4)
        logger.info(f"Formatted data saved to JSON at {json_output_path}")
        
        if isinstance(formatted_data_dict, dict):
            data_for_df = next(iter(formatted_data_dict.values())) if len(formatted_data_dict) == 1 else formatted_data_dict
        elif isinstance(formatted_data_dict, list):
            data_for_df = formatted_data_dict
        else:
            error_msg = "Formatted data is neither a dictionary nor a list, cannot convert to DataFrame"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            df = pd.DataFrame(data_for_df)
            logger.info("DataFrame created successfully")
            excel_output_path = os.path.join(output_folder, f'sorted_data_{timestamp}.xlsx')
            df.to_excel(excel_output_path, index=False)
            logger.info(f"Formatted data saved to Excel at {excel_output_path}")
            return df
        except Exception as e:
            logger.error(f"Error creating DataFrame or saving Excel: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"Error in save_formatted_data: {str(e)}")
        raise

def calculate_price(token_counts, model):
    logger = logging.getLogger(__name__)
    try:
        input_token_count = token_counts.get("input_tokens", 0)
        output_token_count = token_counts.get("output_tokens", 0)
        input_cost = input_token_count * PRICING[model]["input"]
        output_cost = output_token_count * PRICING[model]["output"]
        total_cost = input_cost + output_cost
        
        logger.info(f"Price calculation completed for model {model}")
        logger.info(f"Input tokens: {input_token_count}, Output tokens: {output_token_count}")
        logger.info(f"Total cost: ${total_cost:.4f}")
        
        return input_token_count, output_token_count, total_cost
    except Exception as e:
        logger.error(f"Error calculating price: {str(e)}")
        raise

class WebScraperDatabase:
    _local = threading.local()

    def __init__(self, db_path: str = 'web_scraper_data.db'):
        self.db_path = db_path
        self._create_connection()

    def _create_connection(self):
        try:
            # Use thread-local storage for database connections
            if not hasattr(self._local, 'conn'):
                self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self._local.conn.execute('pragma journal_mode=wal')  # Improve concurrent access
            
            self.conn = self._local.conn
            self.cursor = self.conn.cursor()
            print(f"Connected to SQLite database at {self.db_path}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            raise

    def close_connection(self):
        if hasattr(self._local, 'conn'):
            try:
                self._local.conn.close()
                del self._local.conn
            except Exception as e:
                print(f"Error closing database: {e}")

    def _create_tables(self):
        """
        Create necessary tables for storing web scraping data
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
            raise

    def insert_scrape_data(self, 
                            url: str, 
                            raw_data: str, 
                            markdown_data: str, 
                            model_used: str, 
                            formatted_data: Dict[str, Any], 
                            token_counts: Dict[str, int], 
                            total_cost: float):
        """
        Insert a complete scraping session's data into the database
        
        :param url: URL that was scraped
        :param raw_data: Raw HTML or text data
        :param markdown_data: Converted markdown data
        :param model_used: AI model used for data extraction
        :param formatted_data: Extracted and formatted data
        :param token_counts: Token usage information
        :param total_cost: Total cost of the API call
        :return: The ID of the inserted scrape metadata
        """
        try:
            # Insert metadata
            self.cursor.execute('''
                INSERT INTO scrape_metadata 
                (url, raw_data, markdown_data, model_used) 
                VALUES (?, ?, ?, ?)
            ''', (url, raw_data, markdown_data, model_used))
            scrape_id = self.cursor.lastrowid

            # Insert listings
            if formatted_data and 'listings' in formatted_data:
                for listing in formatted_data['listings']:
                    self.cursor.execute('''
                        INSERT INTO listings (scrape_id, data) 
                        VALUES (?, ?)
                    ''', (scrape_id, json.dumps(listing)))

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

            self.conn.commit()
            print(f"Scrape data inserted successfully. Scrape ID: {scrape_id}")
            return scrape_id

        except sqlite3.Error as e:
            self.conn.rollback()
            print(f"Error inserting scrape data: {e}")
            raise

    def retrieve_scrapes(self, 
                          limit: int = 10, 
                          model_filter: str = None, 
                          start_date: str = None, 
                          end_date: str = None):
        """
        Retrieve scraping sessions with optional filtering
        
        :param limit: Maximum number of records to retrieve
        :param model_filter: Filter by AI model used
        :param start_date: Earliest timestamp to retrieve from
        :param end_date: Latest timestamp to retrieve to
        :return: List of scraping session details
        """
        try:
            query = '''
                SELECT 
                    sm.id, sm.url, sm.timestamp, sm.model_used,
                    tc.input_tokens, tc.output_tokens, tc.total_cost
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
            raise

    def get_listings_for_scrape(self, scrape_id: int):
        """
        Retrieve listings for a specific scrape session
        
        :param scrape_id: ID of the scrape session
        :return: List of listings
        """
        try:
            self.cursor.execute('''
                SELECT data FROM listings 
                WHERE scrape_id = ?
            ''', (scrape_id,))
            return [json.loads(row[0]) for row in self.cursor.fetchall()]

        except sqlite3.Error as e:
            print(f"Error retrieving listings: {e}")
            raise

    def close_connection(self):
        """
        Close the database connection
        """
        if self.conn:
            self.conn.close()
            print("Database connection closed")

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

if __name__ == "__main__":
    url = 'https://webscraper.io/test-sites/e-commerce/static'
    fields=['Name of item', 'Price']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging at the start
    logger = setup_logging(timestamp)
    logger.info("Starting web scraping process")
    
    try:
        logger.info(f"Processing URL: {url}")
        raw_html = fetch_html_playwright(url)
        
        logger.info("Converting HTML to markdown")
        markdown = html_to_markdown_with_readability(raw_html)
        save_raw_data(markdown, timestamp)
        
        logger.info("Creating data models")
        DynamicListingModel = create_dynamic_listing_model(fields)
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        
        logger.info("Formatting data with gemini-1.5-flash")
        formatted_data, token_counts = format_data(markdown, DynamicListingsContainer, DynamicListingModel, "gemini-1.5-flash")
        logger.info("Data formatting completed")
        
        save_formatted_data(formatted_data, timestamp)
        
        input_tokens, output_tokens, total_cost = calculate_price(token_counts, "Groq Llama3.1 70b")
        logger.info(f"Process completed successfully")
        logger.info(f"Input token count: {input_tokens}")
        logger.info(f"Output token count: {output_tokens}")
        logger.info(f"Estimated total cost: ${total_cost:.4f}")

    except Exception as e:
        logger.error(f"An error occurred during the process: {str(e)}")
        raise
    scrape_id = integrate_database_with_scraper(
        url, 
        markdown, 
        formatted_data, 
        token_counts, 
        total_cost, 
        "gemini-1.5-flash"
    )

    # Retrieve past scrapes
    db = WebScraperDatabase()
    recent_scrapes = db.retrieve_scrapes(limit=5)
    print("Recent Scrapes:", recent_scrapes)

    # Get listings for a specific scrape
    if scrape_id:
        listings = db.get_listings_for_scrape(scrape_id)
        print("Listings:", listings)