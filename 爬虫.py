
import requests

def fetch_financial_news():
    url = "https://api.example.com/financial_news"  # 替换为实际的财经新闻API
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()['articles']
    else:
        print("Error fetching data")
        return []

news_data = fetch_financial_news()
print(news_data)
