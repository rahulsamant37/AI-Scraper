import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

base_url = "https://publiclibraries.com/state/"

response = requests.get(base_url)
soup = BeautifulSoup(response.content, "html.parser")
dropdown = soup.find("div", class_="dropdown-content")
links = dropdown.find_all("a")
os.makedirs("data_1", exist_ok=True)
for link in links:
    state_url = link['href']
    state_name = link.text.strip()
    state_response = requests.get(state_url)
    state_soup = BeautifulSoup(state_response.content, "html.parser")
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
    csv_filename = f"{state_name}.csv"
    df.to_csv(f"data_1/{csv_filename}", index=False)
    print(f"Data saved for {state_name} in {csv_filename}")
