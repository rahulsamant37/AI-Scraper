from playwright.async_api import async_playwright
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
import time
import json
import logging
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

class BehanceDetailedScraper:
    def __init__(self):
        self.projects = []

    async def scrape_main_page(self):
        """Scrape the main Behance page using Playwright with exact HTML structure matching"""
        async with async_playwright() as p:
            try:
                print("Launching browser...")
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                )
                
                page = await context.new_page()
                print("Loading Behance...")
                await page.goto("https://www.behance.net/")
                
                # Wait for projects to load using the exact class name
                print("Waiting for content to load...")
                await page.wait_for_selector(".ProjectCover-root-X6u", timeout=60000)
                
                # Scroll to load more content
                print("Scrolling to load more content...")
                for _ in range(3):
                    await page.evaluate("window.scrollBy(0, window.innerHeight)")
                    await asyncio.sleep(2)
                
                # Extract project information using the exact HTML structure
                self.projects = await page.evaluate("""
                    () => {
                        const projects = [];
                        const projectElements = document.querySelectorAll('.ProjectCover-root-X6u');
                        
                        projectElements.forEach(element => {
                            const project = {};
                            
                            // Get title from the Title-title-lpJ class
                            const titleElement = element.querySelector('.Title-title-lpJ');
                            if (titleElement) {
                                project.title = titleElement.textContent.trim();
                                project.url = titleElement.closest('a')?.href || '';
                            }
                            
                            // Get author from the Owners-owner-EEG class
                            const authorElement = element.querySelector('.Owners-owner-EEG');
                            project.author = authorElement ? authorElement.textContent.trim() : '';
                            
                            // Get stats
                            const statsElement = element.querySelector('.Stats-stats-Q1s');
                            if (statsElement) {
                                const stats = statsElement.textContent.trim().match(/\\d+[KMB]?/g) || [];
                                project.appreciations = stats[0] || '0';
                                project.views = stats[1] || '0';
                            }
                            
                            // Get image URL from the ProjectCoverNeue-image-TFB class
                            const imageElement = element.querySelector('.ProjectCoverNeue-image-TFB');
                            if (imageElement) {
                                project.image_url = imageElement.src;
                                project.image_alt = imageElement.alt;
                            }
                            
                            // Get feature flags if any
                            const featuredElement = element.querySelector('.Feature-ribbon-Tyk');
                            project.is_featured = !!featuredElement;
                            
                            // Only add projects with at least a title
                            if (project.title) {
                                projects.push(project);
                            }
                        });
                        
                        return projects;
                    }
                """)
                
                await browser.close()
                print(f"Found {len(self.projects)} projects")
                return True
                
            except Exception as e:
                logging.error(f"Error during Playwright scraping: {str(e)}")
                return False

    async def process_with_langchain(self):
        """Process project URLs with LangChain"""
        try:
            print("\nProcessing content with LangChain...")
            
            # Get URLs of featured projects
            urls = [p['url'] for p in self.projects if p['url'] and p.get('is_featured')]
            
            if urls:
                # Initialize LangChain loader
                loader = AsyncHtmlLoader(urls[:5])  # Process first 5 featured projects
                docs = await loader.load()
                
                # Transform HTML to text
                html2text = Html2TextTransformer()
                docs_transformed = html2text.transform_documents(docs)
                
                # Add processed content to respective projects
                for i, doc in enumerate(docs_transformed):
                    url = urls[i]
                    for project in self.projects:
                        if project['url'] == url:
                            project['detailed_content'] = doc.page_content[:1000] + "..."  # First 1000 chars
                            break
                
            return True
            
        except Exception as e:
            logging.error(f"Error during LangChain processing: {str(e)}")
            return False

    def save_results(self):
        """Save the scraped and processed data"""
        try:
            # Format the data for better readability
            formatted_projects = []
            for project in self.projects:
                formatted_project = {
                    'title': project.get('title', 'N/A'),
                    'author': project.get('author', 'N/A'),
                    'stats': {
                        'appreciations': project.get('appreciations', '0'),
                        'views': project.get('views', '0')
                    },
                    'featured': project.get('is_featured', False),
                    'url': project.get('url', ''),
                    'image': {
                        'url': project.get('image_url', ''),
                        'alt': project.get('image_alt', '')
                    }
                }
                if 'detailed_content' in project:
                    formatted_project['detailed_content'] = project['detailed_content']
                formatted_projects.append(formatted_project)
            
            with open('behance_detailed_projects.json', 'w', encoding='utf-8') as f:
                json.dump(formatted_projects, f, ensure_ascii=False, indent=2)
            
            print("\nData saved to behance_detailed_projects.json")
            return True
            
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")
            return False

async def main():
    scraper = BehanceDetailedScraper()
    
    print("Starting Behance scraper...")
    if await scraper.scrape_main_page():
        await scraper.process_with_langchain()
        scraper.save_results()
        
        # Print sample of processed data
        print("\n=== SAMPLE PROCESSED DATA ===")
        for idx, project in enumerate(scraper.projects[:3], 1):
            print(f"\nProject {idx}:")
            print(f"Title: {project.get('title', 'N/A')}")
            print(f"Author: {project.get('author', 'N/A')}")
            print(f"Stats: {project.get('appreciations', '0')} appreciations, {project.get('views', '0')} views")
            print(f"Featured: {'Yes' if project.get('is_featured') else 'No'}")
            print(f"URL: {project.get('url', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
