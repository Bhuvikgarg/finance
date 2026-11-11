"""
Real-time News Ingestion and Sentiment NLP Analysis Engine.
Fetches live headlines from financial RSS feeds and computes sentiment scores.
"""

from typing import Dict, List, Optional
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Financial vocabulary tweaks for more accurate market sentiment scoring
FINANCIAL_LEXICON = {
    "bullish": 2.5,
    "rally": 2.2,
    "surge": 2.0,
    "outperform": 2.0,
    "growth": 1.8,
    "profitability": 2.0,
    "beat": 2.0,
    "breakout": 2.2,
    "dividend": 1.5,
    "upgrade": 2.2,
    "bearish": -2.5,
    "plunge": -2.4,
    "slump": -2.2,
    "tumble": -2.0,
    "underperform": -2.0,
    "loss": -2.0,
    "miss": -2.0,
    "downgrade": -2.2,
    "probe": -2.5,
    "fraud": -3.5,
    "investigation": -2.8,
    "sebi": -0.5,
    "fine": -1.8,
    "penalty": -2.2,
    "scam": -3.5,
    "default": -3.0,
}
_analyzer.lexicon.update(FINANCIAL_LEXICON)


def get_search_queries_for_ticker(ticker: str, company_name: Optional[str] = None) -> str:
    """Generate optimal search keywords for Indian and global equities."""
    clean_sym = ticker.replace(".NS", "").replace(".BO", "")
    
    # Specific known name mappings
    aliases = {
        "ETERNAL": "Eternal Limited OR Zomato stock NSE",
        "RELIANCE": "Reliance Industries stock NSE",
        "TCS": "Tata Consultancy Services stock NSE",
        "INFY": "Infosys stock NSE",
        "HDFCBANK": "HDFC Bank stock NSE",
        "ICICIBANK": "ICICI Bank stock NSE",
        "TATAMOTORS": "Tata Motors stock NSE",
        "BHARTIARTL": "Bharti Airtel stock NSE",
    }
    
    if clean_sym in aliases:
        return aliases[clean_sym]
    if company_name:
        return f"{company_name} OR {clean_sym} stock NSE"
    return f"{clean_sym} stock NSE"


def fetch_live_news_sentiment(
    ticker: str,
    company_name: Optional[str] = None,
    max_articles: int = 15,
) -> Dict:
    """
    Fetch recent news articles via Google News RSS and compute NLP sentiment.
    Returns:
        dict containing:
            - articles: List of article dicts with title, pub_date, link, sentiment, compound
            - average_compound: Mean sentiment score (-1.0 to +1.0)
            - sentiment_label: Overall label (Bullish, Neutral, Bearish)
            - bullish_count, bearish_count, neutral_count
            - sentiment_index: Scaled from -100 to +100
    """
    query = get_search_queries_for_ticker(ticker, company_name)
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    articles = []
    try:
        with urllib.request.urlopen(req, timeout=6) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            items = root.findall(".//item")

            for item in items[:max_articles]:
                title = item.find("title").text if item.find("title") is not None else ""
                pub_date = item.find("pubDate").text if item.find("pubDate") is not None else ""
                link = item.find("link").text if item.find("link") is not None else "#"
                source = item.find("source").text if item.find("source") is not None else "Financial News"

                # Calculate Sentiment
                scores = _analyzer.polarity_scores(title)
                compound = float(scores["compound"])

                if compound >= 0.08:
                    sentiment = "Bullish"
                    badge_color = "green"
                elif compound <= -0.08:
                    sentiment = "Bearish"
                    badge_color = "red"
                else:
                    sentiment = "Neutral"
                    badge_color = "gray"

                articles.append({
                    "title": title,
                    "pub_date": pub_date,
                    "link": link,
                    "source": source,
                    "compound": compound,
                    "sentiment": sentiment,
                    "badge_color": badge_color,
                })
    except Exception as e:
        print(f"[NewsSentiment] Warning: Failed to fetch live news for {ticker}: {e}")

    if not articles:
        return {
            "articles": [],
            "average_compound": 0.0,
            "sentiment_label": "Neutral / No Recent News",
            "bullish_count": 0,
            "bearish_count": 0,
            "neutral_count": 0,
            "sentiment_index": 0,
        }

    compounds = [a["compound"] for a in articles]
    avg_compound = float(sum(compounds) / len(compounds))

    bull_count = sum(1 for a in articles if a["sentiment"] == "Bullish")
    bear_count = sum(1 for a in articles if a["sentiment"] == "Bearish")
    neu_count = sum(1 for a in articles if a["sentiment"] == "Neutral")

    if avg_compound >= 0.10:
        overall_label = "🟢 Bullish News Flow"
    elif avg_compound <= -0.10:
        overall_label = "🔴 Bearish News Flow"
    else:
        overall_label = "⚪ Neutral / Mixed News Flow"

    return {
        "articles": articles,
        "average_compound": avg_compound,
        "sentiment_label": overall_label,
        "bullish_count": bull_count,
        "bearish_count": bear_count,
        "neutral_count": neu_count,
        "sentiment_index": int(round(avg_compound * 100)),
    }


if __name__ == "__main__":
    res = fetch_live_news_sentiment("ETERNAL.NS")
    print("Sentiment Label:", res["sentiment_label"])
    print("Sentiment Index:", res["sentiment_index"])
    print(f"Articles Found: {len(res['articles'])}")
    for a in res["articles"][:3]:
        print(f"  [{a['sentiment']}] {a['title']}")
