from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import os
import time

options = webdriver.ChromeOptions()
options.add_argument("--headless")
driver = webdriver.Chrome(options=options)

base_url = "https://publiclibraries.com/state/"
driver.get(base_url)
time.sleep(2)
soup = BeautifulSoup(driver.page_source, "html.parser")
dropdown = soup.find("div", class_="dropdown-content")
links = dropdown.find_all("a")
os.makedirs("data_2", exist_ok=True)

for link in links:
    state_url = link['href']
    state_name = link.text.strip()
    driver.get(state_url)
    time.sleep(2) 
    state_soup = BeautifulSoup(driver.page_source, "html.parser")
    libraries_table = state_soup.find("table", id="libraries")
    if libraries_table is None:
        print(f"No table found for {state_name}")
        continue
    headers = [th.text.strip() for th in libraries_table.find_all("th")]
    rows = []
    for tr in libraries_table.find_all("tr")[1:]:
        columns = [td.text.strip() for td in tr.find_all("td")]
        if columns:
            rows.append(columns)
    df = pd.DataFrame(rows, columns=headers)
    csv_filename = f"data_2/{state_name}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"Data saved for {state_name} in {csv_filename}")
driver.quit()
