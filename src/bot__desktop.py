#!/usr/bin/env python3
"""
Financial Analysis Bot – Desktop App (Eel, Edge, fast, crypto fix)
"""

import os
import re
import json
import time
import hashlib
import logging
import sys
import concurrent.futures
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Dict, Any, Optional, Tuple
from io import BytesIO

import eel
import requests
import yfinance as yf
from ddgs import DDGS
from bs4 import BeautifulSoup

# Optional imports
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# ================= CONFIG =================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "openai/gpt-oss-120b:free"
CURRENT_DATE = datetime.now().strftime("%d.%m.%Y")

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
PRICE_CACHE_DIR = os.path.join(CACHE_DIR, "prices")
NEWS_CACHE_DIR = os.path.join(CACHE_DIR, "news")
GLOBAL_NEWS_CACHE_DIR = os.path.join(CACHE_DIR, "global_news")
TREND_CACHE_DIR = os.path.join(CACHE_DIR, "trend")
DETECTOR_CACHE_DIR = os.path.join(CACHE_DIR, "detector")
STATS_CACHE_DIR = os.path.join(CACHE_DIR, "stats")
FOREX_CACHE_DIR = os.path.join(CACHE_DIR, "forex")

for d in [CACHE_DIR, PRICE_CACHE_DIR, NEWS_CACHE_DIR, GLOBAL_NEWS_CACHE_DIR,
          TREND_CACHE_DIR, DETECTOR_CACHE_DIR, STATS_CACHE_DIR, FOREX_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

CACHE_TTL_PRICE = 60
CACHE_TTL_TREND = 3600
CACHE_TTL_GLOBAL_NEWS = 7200
CACHE_TTL_NEWS = 600
CACHE_TTL_DETECTOR = 86400
CACHE_TTL_STATS = 3600
CACHE_TTL_FOREX = 300

MAX_TOKENS_ANALYSIS = 1500
MAX_TOKENS_DETECTOR = 180

# Общий таймаут сбора данных (не более 12 секунд)
CONTEXT_TIMEOUT = 12

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# ================= CACHE =================
def get_cache_key(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()

def get_cached(cache_dir: str, key: str, ttl: int) -> Optional[Any]:
    cache_file = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                if time.time() - data.get("timestamp", 0) < ttl:
                    return data.get("value")
        except:
            pass
    return None

def set_cached(cache_dir: str, key: str, value: Any):
    cache_file = os.path.join(cache_dir, f"{key}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump({"timestamp": time.time(), "value": value}, f)
    except:
        pass

# ================= LLM (soft errors) =================
def call_llm(prompt: str, max_tokens: int, temperature: float = 0.3) -> str:
    if not OPENROUTER_API_KEY:
        return "ERROR: API ключ не задан. Проверьте, что переменная OPENROUTER_API_KEY установлена."
    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=30
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            if content is None:
                return "ERROR: Ассистент вернул пустой ответ. Попробуйте задать вопрос иначе."
            content = re.sub(r'\*\*?|__|#{1,6}\s|`{1,3}|\|', '', content)
            return content.strip()
        elif resp.status_code == 429:
            return "ERROR: К сожалению, на сегодня запросы закончились. Продолжайте завтра после 00:00."
        elif resp.status_code == 401:
            return "ERROR: Неверный API-ключ. Проверьте ключ OpenRouter."
        elif resp.status_code == 500:
            return "ERROR: Внутренняя ошибка сервера OpenRouter. Попробуйте позже."
        else:
            return f"ERROR: Сервер ответил кодом {resp.status_code}. Попробуйте позже."
    except requests.exceptions.ConnectionError:
        return "ERROR: Не удалось подключиться к серверу. Проверьте интернет-соединение."
    except requests.exceptions.Timeout:
        return "ERROR: Сервер долго не отвечает. Повторите попытку позже."
    except Exception as e:
        return f"ERROR: Непредвиденная ошибка: {str(e)[:100]}"

# ================= FOREX (Frankfurter + fallback) =================
def get_forex_rate_frankfurter(from_currency: str, to_currency: str = "RUB") -> Optional[float]:
    try:
        url = f"https://api.frankfurter.app/latest?from={from_currency}&to={to_currency}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            rate = data['rates'].get(to_currency)
            if rate:
                return rate
    except:
        pass
    return None

def get_forex_rate_exchange(from_currency: str, to_currency: str = "RUB") -> Optional[float]:
    try:
        url = f"https://api.exchangerate-api.com/v4/latest/{from_currency}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            rate = data['rates'].get(to_currency)
            if rate:
                return rate
    except:
        pass
    return None

def get_any_currency_rate(from_currency: str) -> Optional[float]:
    cache_key = get_cache_key("currency", from_currency)
    cached = get_cached(PRICE_CACHE_DIR, cache_key, CACHE_TTL_PRICE)
    if cached:
        return cached
    rate = get_forex_rate_frankfurter(from_currency, "RUB")
    if not rate:
        rate = get_forex_rate_exchange(from_currency, "RUB")
    if not rate:
        try:
            resp = requests.get("https://www.cbr-xml-daily.ru/daily_json.js", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if "USD" in data["Valute"]:
                    usd_rate = data["Valute"]["USD"]["Value"]
                    resp2 = requests.get(f"https://api.exchangerate-api.com/v4/latest/{from_currency}", timeout=5)
                    if resp2.status_code == 200:
                        data2 = resp2.json()
                        if "USD" in data2["rates"]:
                            rate = usd_rate / data2["rates"]["USD"]
        except:
            pass
    if rate:
        set_cached(PRICE_CACHE_DIR, cache_key, rate)
        return rate
    return None

# ================= PRICE API (крипта через CoinGecko + Yahoo) =================
def get_usd_rub() -> Optional[float]:
    try:
        resp = requests.get("https://www.cbr-xml-daily.ru/daily_json.js", timeout=5)
        if resp.status_code == 200:
            rate = resp.json()["Valute"]["USD"]["Value"]
            if rate and 50 < rate < 200:
                return rate
    except:
        pass
    try:
        resp = requests.get("https://api.exchangerate-api.com/v4/latest/USD", timeout=5)
        if resp.status_code == 200:
            rate = resp.json()["rates"]["RUB"]
            if rate and 50 < rate < 200:
                return rate
    except:
        pass
    return None

def get_eur_rub() -> Optional[float]:
    try:
        resp = requests.get("https://www.cbr-xml-daily.ru/daily_json.js", timeout=5)
        if resp.status_code == 200:
            return resp.json()["Valute"]["EUR"]["Value"]
    except:
        pass
    return None

def get_cny_rub() -> Optional[float]:
    cache_key = get_cache_key("cny_rub")
    cached = get_cached(PRICE_CACHE_DIR, cache_key, CACHE_TTL_PRICE)
    if cached:
        return cached
    try:
        resp = requests.get("https://www.cbr-xml-daily.ru/daily_json.js", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            for code in ["CNY", "CNYK"]:
                if code in data["Valute"]:
                    rate = data["Valute"][code]["Value"]
                    set_cached(PRICE_CACHE_DIR, cache_key, rate)
                    return rate
    except:
        pass
    return None

def get_moex_price(ticker: str) -> Optional[float]:
    cache_key = get_cache_key("moex", ticker)
    cached = get_cached(PRICE_CACHE_DIR, cache_key, CACHE_TTL_PRICE)
    if cached:
        return cached
    try:
        urls = [
            f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/tqbr/securities/{ticker}.json",
            f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{ticker}.json",
        ]
        for url in urls:
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                continue
            data = resp.json()
            if "marketdata" in data and data["marketdata"]["data"]:
                cols = data["marketdata"]["columns"]
                if "LAST" in cols:
                    idx = cols.index("LAST")
                    for row in data["marketdata"]["data"]:
                        if len(row) > idx and row[idx] is not None:
                            price = float(row[idx])
                            if price > 0:
                                set_cached(PRICE_CACHE_DIR, cache_key, price)
                                return price
        return None
    except:
        return None

def get_crypto_price_coingecko(coin_id: str) -> Optional[float]:
    """
    Быстрый запрос к CoinGecko API v3 для получения цены в USD.
    coin_id: например 'bitcoin', 'ethereum', 'solana'
    """
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return float(data[coin_id]['usd'])
    except:
        pass
    return None

# Соответствие тикеров Yahoo -> CoinGecko ID
CRYPTO_COINGECKO_MAP = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
    "DOGE-USD": "dogecoin",
    "XRP-USD": "ripple",
}

def get_yahoo_price(ticker: str) -> Optional[float]:
    cache_key = get_cache_key("yahoo", ticker)
    cached = get_cached(PRICE_CACHE_DIR, cache_key, CACHE_TTL_PRICE)
    if cached:
        return cached
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="1d", timeout=5)
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            set_cached(PRICE_CACHE_DIR, cache_key, price)
            return price
    except Exception as e:
        logger.error(f"Yahoo error for {ticker}: {e}")
        return None

def get_crypto_price(ticker: str) -> Optional[float]:
    """
    Для криптовалют сначала пытаемся CoinGecko, затем Yahoo.
    """
    cache_key = get_cache_key("crypto", ticker)
    cached = get_cached(PRICE_CACHE_DIR, cache_key, CACHE_TTL_PRICE)
    if cached:
        return cached

    # 1. CoinGecko
    coin_id = CRYPTO_COINGECKO_MAP.get(ticker)
    if coin_id:
        price = get_crypto_price_coingecko(coin_id)
        if price:
            set_cached(PRICE_CACHE_DIR, cache_key, price)
            return price

    # 2. Yahoo (запасной)
    price = get_yahoo_price(ticker)
    if price:
        set_cached(PRICE_CACHE_DIR, cache_key, price)
        return price
    return None

def get_price_trend(ticker: str, market: str, days: int = 30) -> Optional[Dict[str, Any]]:
    if market not in ("yahoo", "crypto"):
        return None
    cache_key = get_cache_key("trend", ticker, market, days)
    cached = get_cached(TREND_CACHE_DIR, cache_key, CACHE_TTL_TREND)
    if cached:
        return cached
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period=f"{days}d", timeout=5)
        if not hist.empty and len(hist) >= 5:
            prices = hist["Close"].tolist()
            trend = {
                "current": round(prices[-1], 2),
                "week_ago": round(prices[-7], 2) if len(prices) >= 7 else round(prices[0], 2),
                "month_ago": round(prices[0], 2),
                "high": round(max(prices), 2),
                "low": round(min(prices), 2),
                "change_percent": round(((prices[-1] - prices[0]) / prices[0]) * 100, 1)
            }
            set_cached(TREND_CACHE_DIR, cache_key, trend)
            return trend
    except:
        pass
    return None

# ================= STATISTICS =================
def fetch_fed_rate() -> Optional[Dict[str, Any]]:
    cache_key = get_cache_key("fed_rate")
    cached = get_cached(STATS_CACHE_DIR, cache_key, CACHE_TTL_STATS)
    if cached:
        return cached
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            lines = resp.text.strip().split('\n')
            if len(lines) >= 2:
                last_line = lines[-1].split(',')
                latest_rate = round(float(last_line[1]), 2)
                result = {"rate": latest_rate, "source": "FRED", "date": last_line[0]}
                set_cached(STATS_CACHE_DIR, cache_key, result)
                return result
    except:
        pass
    return None

def fetch_cbr_key_rate() -> Optional[Dict[str, Any]]:
    cache_key = get_cache_key("cbr_rate")
    cached = get_cached(STATS_CACHE_DIR, cache_key, CACHE_TTL_STATS)
    if cached:
        return cached
    try:
        url = "https://cbr.ru/hd_base/KeyRate/"
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, 'html.parser')
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:
                if re.search(r'\d{2}\.\d{2}\.\d{4}', cells[0].text):
                    rate_text = cells[1].text.strip().replace(',', '.')
                    rate = float(re.search(r'[\d.]+', rate_text).group())
                    result = {"rate": rate, "source": "CBR", "date": cells[0].text.strip()}
                    set_cached(STATS_CACHE_DIR, cache_key, result)
                    return result
    except:
        pass
    return None

def fetch_rosstat_inflation() -> Optional[Dict[str, Any]]:
    cache_key = get_cache_key("rosstat_inflation")
    cached = get_cached(STATS_CACHE_DIR, cache_key, CACHE_TTL_STATS)
    if cached:
        return cached
    if PANDAS_AVAILABLE:
        try:
            url = "https://www.cbr.ru/Queries/UniDbQuery/DownloadExcel/98957?Posted=True&mode=1"
            resp = requests.get(url, timeout=5)
            df = pd.read_excel(BytesIO(resp.content), skiprows=3)
            for idx, row in df.iterrows():
                if isinstance(row.iloc[0], str) and 'инфляци' in row.iloc[0].lower():
                    inflation = float(row.iloc[1])
                    result = {
                        "inflation": round(inflation, 2),
                        "source": "CBR",
                        "period": str(row.iloc[0])
                    }
                    set_cached(STATS_CACHE_DIR, cache_key, result)
                    return result
        except:
            pass
    try:
        url = "https://rosstat.gov.ru/storage/mediabank/infl_2025.html"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            tables = soup.find_all('table')
            inflation_data = []
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        period = cells[0].get_text(strip=True)
                        value_text = cells[1].get_text(strip=True).replace(',', '.').replace('%', '')
                        if any(word in period.lower() for word in ['янв', 'фев', 'мар', 'апр', 'май', 'июн', 'июл', 'авг', 'сен', 'окт', 'ноя', 'дек', 'год']):
                            try:
                                value = float(re.search(r'[\d.,]+', value_text).group().replace(',', '.'))
                                inflation_data.append({"period": period, "value": value})
                            except:
                                continue
            if inflation_data:
                recent = inflation_data[:4]
                result = {
                    "inflation": recent[0]["value"],
                    "source": "Rosstat",
                    "period": recent[0]["period"],
                    "history": recent
                }
                set_cached(STATS_CACHE_DIR, cache_key, result)
                return result
    except Exception as e:
        logger.error(f"Rosstat parse error: {e}")
    return None

# ================= DETECTOR =================
def detect_intent(query: str) -> Dict[str, Any]:
    cache_key = get_cache_key("intent", query.lower())
    cached = get_cached(DETECTOR_CACHE_DIR, cache_key, CACHE_TTL_DETECTOR)
    if cached:
        return cached

    prompt = f"""Определи параметры запроса. Верни ТОЛЬКО JSON.

Запрос: {query}

JSON:
{{
    "type": "financial" или "general",
    "ticker": "тикер или null",
    "market": "forex/crypto/moex/yahoo/null",
    "need_search": true/false,
    "need_fed": true/false,
    "need_cbr": true/false,
    "need_inflation": true/false
}}

Правила:
- Криптовалюты: всегда с суффиксом -USD (BTC-USD, ETH-USD, SOL-USD)
- Акции: стандартный тикер (AAPL, TSLA, SBER, GAZP)
- Валюты: USD, EUR, CNY, GBP, CHF, JPY
- need_fed = true если речь о долларе, биткоине, крипте, ФРС, США
- need_cbr = true если речь о рубле, России
- need_inflation = true если запрос об инфляции, ценах

Только JSON."""

    response = call_llm(prompt, MAX_TOKENS_DETECTOR, 0.1)
    if response.startswith("ERROR"):
        return {"type": "general", "ticker": None, "market": None, "need_search": False,
                "need_fed": False, "need_cbr": False, "need_inflation": False}

    try:
        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            set_cached(DETECTOR_CACHE_DIR, cache_key, result)
            return result
    except:
        pass

    return {"type": "general", "ticker": None, "market": None, "need_search": False,
            "need_fed": False, "need_cbr": False, "need_inflation": False}

# ================= NEWS =================
def search_google_news_rss(query: str) -> Optional[str]:
    if not FEEDPARSER_AVAILABLE:
        return None
    try:
        from urllib.parse import quote
        encoded_query = quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)
        news_items = []
        for entry in feed.entries[:5]:
            title = entry.get('title', '')
            if ' - ' in title:
                title = title.split(' - ')[0]
            if title and len(title) > 10:
                news_items.append(title)
        return "\n".join(news_items) if news_items else None
    except:
        return None

def search_global_news(ticker: str) -> str:
    cache_key = get_cache_key("global_news", ticker)
    cached = get_cached(GLOBAL_NEWS_CACHE_DIR, cache_key, CACHE_TTL_GLOBAL_NEWS)
    if cached:
        return cached
    search_term = ticker.replace("-USD", "").lower()
    queries = [f"{search_term} news analysis", f"{search_term} market today", "global financial news"]
    info = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                try:
                    results = list(ddgs.text(q, max_results=2))
                    for r in results:
                        title = r.get("title", "")
                        if title and len(title) > 10:
                            info.append(title)
                except:
                    continue
    except:
        pass
    if not info:
        rss_news = search_google_news_rss(search_term)
        if rss_news:
            info = rss_news.split('\n')
    info = list(dict.fromkeys(info))
    result = "\n".join(info[:8]) if info else "Новости не найдены"
    set_cached(GLOBAL_NEWS_CACHE_DIR, cache_key, result)
    return result

def search_recent_news(ticker: str, days: int = 14) -> str:
    cache_key = get_cache_key("recent_news", ticker, days)
    cached = get_cached(NEWS_CACHE_DIR, cache_key, CACHE_TTL_NEWS)
    if cached:
        return cached
    search_term = ticker.replace("-USD", "").lower()
    queries = [f"{search_term} now", f"{search_term} price today"]
    all_news = []
    try:
        with DDGS() as ddgs:
            for q in queries:
                try:
                    results = list(ddgs.news(q, max_results=4, region='ru-ru'))
                    for r in results:
                        title = r.get("title", "")
                        date = r.get("date", "")[:10] if r.get("date") else ""
                        if title and len(title) > 10:
                            all_news.append(f"[{date or 'без даты'}] {title}")
                except:
                    continue
    except:
        pass
    if not all_news:
        rss_news = search_google_news_rss(search_term)
        if rss_news:
            all_news = [f"[новость] {line}" for line in rss_news.split('\n')]
    all_news = list(dict.fromkeys(all_news))
    result = "\n".join(all_news[:6]) if all_news else "Свежих новостей не найдено"
    set_cached(NEWS_CACHE_DIR, cache_key, result)
    return result

# ================= GET PRICE =================
def get_price(ticker: str, market: str) -> Tuple[Optional[float], Optional[str]]:
    if market == "forex":
        if ticker == "USD": return get_usd_rub(), "RUB"
        elif ticker == "EUR": return get_eur_rub(), "RUB"
        elif ticker == "CNY": return get_cny_rub(), "RUB"
        else:
            rate = get_any_currency_rate(ticker)
            return rate, "RUB" if rate else None
    elif market == "moex":
        price = get_moex_price(ticker)
        return price, "RUB" if price else None
    elif market == "crypto":
        # Используем быстрый Coingecko + Yahoo
        price = get_crypto_price(ticker)
        return price, "USD" if price else None
    elif market == "yahoo":
        price = get_yahoo_price(ticker)
        return price, "USD" if price else None
    return None, None

# ================= BUILD CONTEXT (параллельный с общим таймаутом) =================
def build_context(query: str) -> Dict[str, Any]:
    intent = detect_intent(query)
    if intent.get("type") == "general":
        return {"type": "general", "query": query}

    ticker = intent.get("ticker")
    market = intent.get("market")
    need_search = intent.get("need_search", True)
    need_fed = intent.get("need_fed", False)
    need_cbr = intent.get("need_cbr", False)
    need_inflation = intent.get("need_inflation", False)

    results = {"price": None, "trend": None, "usd": None, "global_news": "", "recent_news": ""}
    stats = {"fed_rate": None, "cbr_rate": None, "inflation": None}

    with ThreadPoolExecutor(max_workers=6) as executor:
        future_map = {}
        if ticker and market:
            future_map['price'] = executor.submit(get_price, ticker, market)
        if ticker and market in ("yahoo", "crypto"):
            future_map['trend'] = executor.submit(get_price_trend, ticker, market, 30)
        future_map['usd'] = executor.submit(get_usd_rub)
        if need_search and ticker:
            future_map['global_news'] = executor.submit(search_global_news, ticker)
            future_map['recent_news'] = executor.submit(search_recent_news, ticker, 14)
        if need_fed:
            future_map['fed_rate'] = executor.submit(fetch_fed_rate)
        if need_cbr:
            future_map['cbr_rate'] = executor.submit(fetch_cbr_key_rate)
        if need_inflation:
            future_map['inflation'] = executor.submit(fetch_rosstat_inflation)

        done, not_done = wait(future_map.values(), timeout=CONTEXT_TIMEOUT)
        for f in not_done:
            f.cancel()

        for key, future in future_map.items():
            if future.done() and not future.cancelled():
                try:
                    val = future.result()
                    if key in ('fed_rate', 'cbr_rate', 'inflation'):
                        stats[key] = val
                    else:
                        results[key] = val
                except:
                    pass

    price_info = results.get('price')
    price = price_info[0] if price_info and isinstance(price_info, tuple) else None
    currency = price_info[1] if price_info and isinstance(price_info, tuple) else None

    return {
        "type": "financial",
        "ticker": ticker,
        "market": market,
        "price": price,
        "currency": currency or "RUB",
        "trend": results.get('trend'),
        "usd_rate": results.get('usd'),
        "global_news": results.get('global_news', ''),
        "recent_news": results.get('recent_news', ''),
        "fed_rate": stats.get('fed_rate'),
        "cbr_rate": stats.get('cbr_rate'),
        "inflation": stats.get('inflation'),
        "need_search": need_search,
        "current_date": CURRENT_DATE
    }

# ================= GENERATE ANALYSIS =================
def generate_analysis(query: str, ctx: Dict[str, Any]) -> str:
    if ctx["type"] == "general":
        prompt = f"Кратко ответь на русском: {ctx.get('query', query)}"
        result = call_llm(prompt, MAX_TOKENS_ANALYSIS, 0.5)
        if result.startswith("ERROR"):
            return f"⚠️ Не удалось получить ответ: {result}"
        return result

    price_str = f"{ctx['price']} {ctx['currency']}" if ctx['price'] else "не найдена"
    trend_info = ""
    if ctx.get('trend'):
        t = ctx['trend']
        trend_info = f"""
Динамика за 30 дней:
- Текущая: {t['current']}
- Неделю назад: {t['week_ago']}
- Месяц назад: {t['month_ago']}
- Изменение: {t['change_percent']}%
- Диапазон: {t['low']} - {t['high']}"""

    stats_block = ""
    if ctx.get('fed_rate'):
        stats_block += f"\n- Ставка ФРС: {ctx['fed_rate']['rate']}%"
    if ctx.get('cbr_rate'):
        stats_block += f"\n- Ключевая ставка ЦБ РФ: {ctx['cbr_rate']['rate']}%"
    if ctx.get('inflation'):
        stats_block += f"\n- Инфляция в РФ: {ctx['inflation']['inflation']}%"
    if stats_block:
        stats_block = "=== СТАТИСТИКА ===" + stats_block

    # Предупреждаем, чтобы не выдумывал числа
    prompt = f"""Ты финансовый аналитик. Сегодня {ctx['current_date']}.  
Опирайся ТОЛЬКО на данные ниже. Не придумывай цифры, не упоминай источники, которых нет в блоке новостей.

Актив: {ctx['ticker']}
Текущая цена: {price_str}
{trend_info}
USD/RUB: {ctx['usd_rate'] if ctx['usd_rate'] else 'не найден'}

{stats_block}

=== ГЛОБАЛЬНЫЕ НОВОСТИ (ключевые мировые события) ===
{ctx['global_news'][:1200] if ctx['global_news'] else 'Нет данных'}

=== СВЕЖИЕ НОВОСТИ (последние заголовки, влияющие на актив) ===
{ctx['recent_news'][:800] if ctx['recent_news'] else 'Не найдены'}

Напиши развёрнутый анализ на русском языке (250-400 слов) в формате:
Текущая ситуация: (текущая цена, тренд)
Глобальный контекст: (выбери 1-2 важнейшие глобальные новости и объясни их влияние)
Статистика: (ставки, инфляция – как они влияют)
Факторы влияния: (институциональные покупки, ETF, геополитика)
Прогноз: (краткий вывод на основе данных)

Обязательно используй информацию из ВСЕХ новостных блоков. Не придумывай новые новости."""

    result = call_llm(prompt, MAX_TOKENS_ANALYSIS, 0.3)
    if result.startswith("ERROR"):
        return f"⚠️ Не удалось выполнить анализ: {result}"
    return result
# ================= EEL EXPOSED FUNCTION (fast, with progress and input lock) =================
@eel.expose
def process_query(query: str) -> str:
    def _process():
        eel.updateStatus("Определяю запрос...")
        ctx = build_context(query)
        eel.updateStatus("Анализирую данные...")
        analysis = generate_analysis(query, ctx)
        eel.updateStatus("Готово")
        if ctx["type"] == "general":
            return analysis
        price_str = f"{ctx['price']} {ctx['currency']}" if ctx.get('price') else "не найдена"
        usd_str = f"{ctx['usd_rate']}" if ctx.get('usd_rate') else "не найден"
        return f"Актив: {ctx['ticker']}\nЦена: {price_str}\nUSD/RUB: {usd_str}\n\n{analysis}"

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_process)
        try:
            return future.result(timeout=60)
        except concurrent.futures.TimeoutError:
            return "⚠️ Обработка заняла слишком много времени. Попробуйте позже."
        except Exception as e:
            return f"Ошибка: {str(e)}"

# ================= LAUNCH =================
if __name__ == '__main__':
    if not OPENROUTER_API_KEY:
        print("❌ Переменная окружения OPENROUTER_API_KEY не задана.")
        sys.exit(1)

    web_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
    if not os.path.exists(os.path.join(web_path, 'index.html')):
        web_path = os.path.dirname(os.path.abspath(__file__))
    eel.init(web_path)
    print("✅ Финансовый AI-ассистент запущен как десктопное приложение.")
    try:
        eel.start('index.html', size=(1100, 700), mode='edge')
    except Exception:
        eel.start('index.html', size=(1100, 700), mode='chrome')
if __name__ == "__main__":
    main()
