import os
import random
import asyncio
import sys
import platform
import time
import re
import json
import logging
from datetime import datetime
from typing import List, Dict, Type

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, create_model
import html2text
import tiktoken

from dotenv import load_dotenv
from playwright.async_api import async_playwright, Page

from openai import OpenAI
import google.generativeai as genai
from groq import Groq

from langchain.text_splitter import RecursiveCharacterTextSplitter

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


async def setup_playwright():
    logger = logging.getLogger(__name__)
    try:
        playwright = await async_playwright().start()
        
        try:
            browser = await playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-gpu']
            )
        except Exception as e:
            logger.error(f"Failed to launch browser: {str(e)}")
            await playwright.stop()
            raise RuntimeError(f"Browser launch failed: {str(e)}")
        
        try:
            context = await browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport={'width': 1920, 'height': 1080}
            )
            await context.set_default_timeout(30000)
            await context.set_default_navigation_timeout(30000)
            page = await context.new_page()
            logger.info("Playwright setup completed successfully")
            return playwright, browser, context, page
            
        except Exception as e:
            logger.error(f"Failed to create context or page: {str(e)}")
            await browser.close()
            await playwright.stop()
            raise RuntimeError(f"Context/page creation failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in setup_playwright: {str(e)}")
        try:
            await playwright.stop()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {str(cleanup_error)}")
        raise RuntimeError(f"Failed to setup Playwright: {str(e)}")

async def click_accept_cookies(page: Page):
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
                if await button.count() > 0:
                    await button.click(timeout=5000)
                    logger.info(f"Clicked cookie consent using selector: {selector}")
                    return
            except Exception:
                continue
        
        logger.warning("No cookie consent button found")
    
    except Exception as e:
        logger.warning(f"Non-critical error handling cookies: {str(e)}")

async def scroll_and_wait(page: Page) -> None:
    """Enhanced scrolling with better error handling"""
    logger = logging.getLogger(__name__)
    try:
        last_height = await page.evaluate("document.body.scrollHeight")
        retries = 3
        
        while retries > 0:
            try:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(1000)
                
                new_height = await page.evaluate("document.body.scrollHeight")
                if new_height == last_height:
                    break
                    
                last_height = new_height
                retries = 3
            except Exception as scroll_error:
                logger.warning(f"Scroll attempt failed: {str(scroll_error)}")
                retries -= 1
                await page.wait_for_timeout(1000)
        
        logger.info("Page scrolling completed")
        
    except Exception as e:
        logger.warning(f"Non-critical scrolling error: {str(e)}")

async def fetch_html_playwright(url: str) -> str:
    logger = logging.getLogger(__name__)
    playwright = browser = context = page = None
    
    try:
        playwright, browser, context, page = await setup_playwright()
        logger.info(f"Fetching HTML from URL: {url}")
        try:
            response = await page.goto(url, wait_until='networkidle')
            if not response:
                raise RuntimeError("Navigation failed: no response received")
            if response.status >= 400:
                raise RuntimeError(f"Navigation failed: HTTP {response.status}")
        except Exception as nav_error:
            logger.error(f"Navigation failed: {str(nav_error)}")
            raise RuntimeError(f"Failed to navigate to URL: {str(nav_error)}")
        
        try:
            await click_accept_cookies(page)
        except Exception as cookie_error:
            logger.warning(f"Cookie handling error (non-critical): {str(cookie_error)}")
        
        await scroll_and_wait(page)
        html = await page.content()
        logger.info("HTML content successfully fetched")
        return html
        
    except Exception as e:
        logger.error(f"Error fetching HTML: {str(e)}")
        raise RuntimeError(f"Failed to fetch HTML: {str(e)}")
        
    finally:
        try:
            if page:
                await page.close()
            if context:
                await context.close()
            if browser:
                await browser.close()
            if playwright:
                await playwright.stop()
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

async def main():
    url = 'https://webscraper.io/test-sites/e-commerce/static'
    fields=['Name of item', 'Price']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging at the start
    logger = setup_logging(timestamp)
    logger.info("Starting web scraping process")
    
    try:
        logger.info(f"Processing URL: {url}")
        raw_html = await fetch_html_playwright(url)
        
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

if __name__ == "__main__":
    asyncio.run(main())