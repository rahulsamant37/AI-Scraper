import aiohttp
import asyncio
from bs4 import BeautifulSoup
import pandas as pd
import os

base_url = "https://publiclibraries.com/state/"
os.makedirs("data_3", exist_ok=True)

async def fetch_state_data(session, state_url, state_name):
    async with session.get(state_url) as response:
        state_soup = BeautifulSoup(await response.text(), "html.parser")
        libraries_table = state_soup.find("table", id="libraries")
        if libraries_table is None:
            print(f"No table found for {state_name}")
            return
        headers = [th.text.strip() for th in libraries_table.find_all("th")]
        rows = []
        for tr in libraries_table.find_all("tr")[1:]:
            columns = [td.text.strip() for td in tr.find_all("td")]
            if columns:
                rows.append(columns)
        df = pd.DataFrame(rows, columns=headers)
        csv_filename = f"data_3/{state_name}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Data saved for {state_name} in {csv_filename}")

async def main():
    async with aiohttp.ClientSession() as session:
        async with session.get(base_url) as response:
            soup = BeautifulSoup(await response.text(), "html.parser")
            dropdown = soup.find("div", class_="dropdown-content")
            links = dropdown.find_all("a")
            tasks = []
            for link in links:
                state_url = link['href']
                state_name = link.text.strip()
                tasks.append(fetch_state_data(session, state_url, state_name))
            await asyncio.gather(*tasks)
asyncio.run(main())
