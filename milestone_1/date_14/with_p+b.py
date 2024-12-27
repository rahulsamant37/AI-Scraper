import os
import csv
import asyncio
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

output_dir = "data_2/outer_html"
os.makedirs(output_dir, exist_ok=True)

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        file = 0  
        for i in range(1, 10):
            await page.goto(f"https://dealsheaven.in/?page={i}")
            await page.wait_for_selector(".product-item-detail")
            elems = await page.query_selector_all(".product-item-detail")
            for elem in elems:
                a = await elem.inner_html()  # Get the outer HTML of the element
                with open(f"{output_dir}/{file}.html", "w", encoding="utf-8") as f:
                    f.write(a)
                file += 1
            await asyncio.sleep(2)

        await browser.close()
asyncio.run(main())

html_dir = "data_2/outer_html"
csv_file = "data_2/products_data.csv"
product_data = []

# Loop through all the saved HTML files
for filename in os.listdir(html_dir):
    if filename.endswith(".html"):
        file_path = os.path.join(html_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            soup = BeautifulSoup(content, "html.parser")
            product_details = soup.find("div", class_="product-item-detail")
            if product_details:
                product_name_tag = product_details.find("h3")
                product_name = product_name_tag.text.strip() if product_name_tag else "N/A"
                current_price_tag = product_details.find("p", class_="price")
                current_price = current_price_tag.text.strip().replace('₹', '').replace(',', '').strip() if current_price_tag else "N/A"
                special_price_tag = product_details.find("p", class_="spacail-price")
                special_price = special_price_tag.text.strip().replace('₹', '').replace(',', '').strip() if special_price_tag else "N/A"
                discount_tag = product_details.find("div", class_="discount")
                discount = discount_tag.text.strip() if discount_tag else "N/A"
                image_tag = product_details.find("img")
                image_url = image_tag['data-src'] if image_tag and 'data-src' in image_tag.attrs else "N/A"
                product_data.append((product_name, current_price, special_price, discount, image_url))

with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Product Name", "Current Price", "Special Price", "Discount", "Image URL"])
    writer.writerows(product_data)

print(f"Successfully saved the product data to {csv_file}")