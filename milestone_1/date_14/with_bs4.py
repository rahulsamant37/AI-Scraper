import os
import csv
import requests
from bs4 import BeautifulSoup

list_links=[]

if not os.path.exists("data_1"):
    os.makedirs("data_1")

url = "https://dealsheaven.in/"

response = requests.get(url)


if response.status_code == 200:
    print("Successfully fetched the website content")    
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string
    print("Webpage Title:", title)
    
    # Find and extract all links
    links = soup.find_all('a')
    for link in links:
        href = link.get('href')
        if href:  
            list_links.append(href)

    with open('data_1/links.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Link"])
        for link in list_links:
            writer.writerow([link])
    
    print(f"Successfully written {len(list_links)} links to links.csv")
else:
    print(f"Failed to retrieve content. Status code: {response.status_code}")
