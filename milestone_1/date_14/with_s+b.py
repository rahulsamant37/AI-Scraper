from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import os
import csv


if not os.path.exists("data_3/outer_html"):
    os.makedirs("data_3/outer_html")

driver = webdriver.Chrome()
file=0

for i in range(1,10):
    driver.get(f"https://dealsheaven.in/?page={i}")
    elems = driver.find_elements(By.CLASS_NAME, "product-item-detail")
    for elem in elems:
        a = elem.get_attribute("outerHTML")
        with open(f"data_3/outer_html/{file}.html","w",encoding="utf-8") as f:
            f.write(a)
        file +=1
    time.sleep(2)
driver.close()


html_dir = "data_3/outer_html"
csv_file = "data_3/products_data.csv"
product_data = []
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