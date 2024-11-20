import requests
from bs4 import BeautifulSoup
import csv
import time

# URL of the page to scrape
url = "https://kamis.kilimo.go.ke/site/market?product=1&per_page=1500"

# Headers to mimic a real browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def scrape_data():
    try:
        # Send a GET request to the website
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check if the request was successful

        # Parse the page content with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Locate the table on the page
        table = soup.find('table')
        if not table:
            print("No table found on the page.")
            return

        # Extract data from the table rows
        rows = table.find_all('tr')
        data = []

        for row in rows[1:]:  # Skipping the header row
            cols = row.find_all('td')
            data_row = [col.text.strip() for col in cols]
            data.append(data_row)

        # Save data to a CSV file
        with open('maize_data.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Commodity", "Classification", "Grade", "Sex", "Market", "Wholesale", "Retail", "Supply Volume", "County", "Date"])  # Header
            writer.writerows(data)

        print("Data scraped and saved to maize_data.csv successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")

# Run the scraping function
if __name__ == "__main__":
    scrape_data()
