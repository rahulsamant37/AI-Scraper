import asyncio
from playwright.async_api import async_playwright
import pandas as pd

async def scrape_amazon_deals(max_pages=10):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        current_page = 1
        with pd.ExcelWriter('product_data.xlsx', engine='xlsxwriter') as writer:
            await page.goto("https://dealsheaven.in/store/amazon")

            while current_page <= max_pages:
                print(f"Scraping page {current_page}...")

                try:
                    await page.wait_for_selector('.product-item-detail', timeout=10000)

                    ad_close_button_selector = 'your_ad_close_button_selector'
                    ad_close_button = await page.query_selector(ad_close_button_selector)
                    
                    if ad_close_button:
                        await ad_close_button.click()
                        print("Closed the advertisement/modal.")
                    else:
                        print("No advertisement/modal found.")

                    products = await page.query_selector_all('.product-item-detail')
                    product_data = []
                    
                    for product in products:
                        product_name = await product.query_selector('h3')
                        product_name = await product_name.inner_text() if product_name else "N/A"
                        current_price = await product.query_selector('.price')
                        current_price = await current_price.inner_text() if current_price else "N/A"
                        special_price = await product.query_selector('.special-price')
                        special_price = await special_price.inner_text() if special_price else "N/A"
                        discount = await product.query_selector('.discount')
                        discount = await discount.inner_text() if discount else "N/A"
                        image = await product.query_selector('img[data-src]')
                        image_url = await image.get_attribute('data-src') if image else "N/A"

                        product_data.append({
                            'Product Name': product_name,
                            'Current Price': current_price,
                            'Special Price': special_price,
                            'Discount': discount,
                            'Image URL': image_url
                        })

                    print(f"Scraped page {current_page} successfully.")
                    df = pd.DataFrame(product_data)
                    df.to_excel(writer, sheet_name=f'Page_{current_page}', index=False)
                    next_page_selector = 'body > section.product-container > div > nav > nav > ul > li:nth-child(15) > a'
                    next_page_link = await page.query_selector(next_page_selector)
                    
                    if next_page_link:
                        try:
                            await next_page_link.click()
                            print(f"Clicked on the 'Next' link for page {current_page + 1}...")
                            await page.wait_for_selector('.product-item-detail', timeout=10000)
                            current_page += 1
                        except Exception as e:
                            print(f"Failed to click 'Next' button: {e}")
                            break
                    else:
                        print("No more pages or end of the list.")
                        break

                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

        await browser.close()
        print("Successfully saved the product data to product_data.xlsx")

if __name__ == "__main__":
    max_pages_to_scrape = 10
    asyncio.run(scrape_amazon_deals(max_pages=max_pages_to_scrape))