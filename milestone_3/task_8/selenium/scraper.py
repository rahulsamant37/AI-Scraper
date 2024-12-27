import os
import random
import time
import re
import json
import logging
from datetime import datetime
from typing import List, Type

import pandas as pd
from bs4 import BeautifulSoup
from pydantic import BaseModel, create_model
import html2text
import tiktoken

from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

def setup_selenium():
    logger = logging.getLogger(__name__)
    options = Options()
    user_agent = random.choice(USER_AGENTS)
    options.add_argument(f"user-agent={user_agent}")
    for option in HEADLESS_OPTIONS:
        options.add_argument(option)
    service = Service(r"./chromedriver-win64/chromedriver-win64/chromedriver.exe") 
    driver = webdriver.Chrome(service=service, options=options)
    logger.info("Selenium WebDriver setup completed")
    return driver

def click_accept_cookies(driver):
    """
    Tries to find and click on a cookie consent button. It looks for several common patterns.
    """
    logger = logging.getLogger(__name__)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//button | //a | //div"))
        )
        accept_text_variations = [
            "accept", "agree", "allow", "consent", "continue", "ok", "I agree", "got it"
        ]
        for tag in ["button", "a", "div"]:
            for text in accept_text_variations:
                try:
                    element = driver.find_element(By.XPATH, f"//{tag}[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{text}')]")
                    if element:
                        element.click()
                        logger.info(f"Clicked the '{text}' button")
                        return
                except:
                    continue

        logger.warning("No 'Accept Cookies' button found")
    
    except Exception as e:
        logger.error(f"Error finding 'Accept Cookies' button: {e}")

def fetch_html_selenium(url):
    logger = logging.getLogger(__name__)
    driver = setup_selenium()
    try:
        logger.info(f"Fetching HTML from URL: {url}")
        driver.get(url)
        time.sleep(1)
        driver.maximize_window()
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)
        html = driver.page_source
        logger.info("HTML content successfully fetched")
        return html
    except Exception as e:
        logger.error(f"Error fetching HTML: {e}")
        raise
    finally:
        driver.quit()
        logger.info("WebDriver session closed")

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
        
        if selected_model in ["gpt-4o-mini", "gpt-4o-2024-08-06"]:
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            completion = client.beta.chat.completions.parse(
                model=selected_model,
                messages=[
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": USER_MESSAGE + data},
                ],
                response_format=DynamicListingsContainer
            )
            encoder = tiktoken.encoding_for_model(selected_model)
            input_token_count = len(encoder.encode(USER_MESSAGE + data))
            output_token_count = len(encoder.encode(json.dumps(completion.choices[0].message.parsed.dict())))
            token_counts = {
                "input_tokens": input_token_count,
                "output_tokens": output_token_count
            }
            logger.info(f"OpenAI API request completed. Input tokens: {input_token_count}, Output tokens: {output_token_count}")
            return completion.choices[0].message.parsed, token_counts

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

if __name__ == "__main__":
    url = 'https://webscraper.io/test-sites/e-commerce/static'
    fields=['Name of item', 'Price']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Setup logging at the start
    logger = setup_logging(timestamp)
    logger.info("Starting web scraping process")
    
    try:
        logger.info(f"Processing URL: {url}")
        raw_html = fetch_html_selenium(url)
        
        logger.info("Converting HTML to markdown")
        markdown = html_to_markdown_with_readability(raw_html)
        save_raw_data(markdown, timestamp)
        
        logger.info("Creating data models")
        DynamicListingModel = create_dynamic_listing_model(fields)
        DynamicListingsContainer = create_listings_container_model(DynamicListingModel)
        
        logger.info("Formatting data with Groq Llama3.1 70b")
        formatted_data, token_counts = format_data(markdown, DynamicListingsContainer, DynamicListingModel, "Groq Llama3.1 70b")
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