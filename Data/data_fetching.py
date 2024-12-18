# data_fetching.py

import requests
import pandas as pd
import datetime
from datetime import timedelta
from config import BASE_URL, HEADERS, generate_headers
import logging

def get_intraday_data(symbol, interval):
    """
    Fetch intraday historical data for the given symbol.
    """
    end = datetime.datetime.utcnow()
    start = end - timedelta(days=1)
    params = {
        'symbol': symbol,
        'interval': interval,
        'start': start.strftime('%Y-%m-%dT%H:%M'),
        'end': end.strftime('%Y-%m-%dT%H:%M'),
        'session_filter': 'open'
    }
    response = requests.get(f"{BASE_URL}/markets/timesales", params=params, headers=HEADERS)
    #print(f"Intraday Data Raw response for {symbol}: {response.text}")
    data = response.json()
    if 'series' in data and 'data' in data['series']:
        df = pd.DataFrame(data['series']['data'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.astype(float)
        return df
    else:
        print("No historical data found.")
        return None
    
def get_intraday_5min(symbol, interval="5min"):
    """
    Fetch intraday historical data for the given symbol.
    """
    end = datetime.datetime.utcnow()
    start = end - timedelta(days=1)
    params = {
        'symbol': symbol,
        'interval': interval,
        'start': start.strftime('%Y-%m-%dT%H:%M'),
        'end': end.strftime('%Y-%m-%dT%H:%M'),
        'session_filter': 'open'
    }
    response = requests.get(f"{BASE_URL}/markets/timesales", params=params, headers=HEADERS)
    #print(f"Intraday Data Raw response for {symbol}: {response.text}")
    data = response.json()
    if 'series' in data and 'data' in data['series']:
        df = pd.DataFrame(data['series']['data'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.astype(float)
        return df
    else:
        print("No historical data found.")
        return None

    
def get_hourly_data(symbol, interval="15min"):
    """
    Fetch hourly historical data for the given symbol.
    """
    end = datetime.datetime.utcnow()
    start = end - timedelta(days=7)  # Fetch the last 7 days of hourly data
    params = {
        'symbol': symbol,
        'interval': interval,
        'start': start.strftime('%Y-%m-%dT%H:%M'),
        'end': end.strftime('%Y-%m-%dT%H:%M'),
        'session_filter': 'open'
    }
    response = requests.get(f"{BASE_URL}/markets/timesales", params=params, headers=HEADERS)
    if response.status_code != 200:
        logging.error(f"Failed to fetch hourly data for {symbol}: {response.status_code} - {response.text}")
        return None
    try:
        data = response.json()
    except ValueError as e:
        logging.error(f"Error parsing JSON response for {symbol}: {e}")
        logging.error(f"Response content: {response.text}")
        return None
    if 'series' in data and 'data' in data['series']:
        df = pd.DataFrame(data['series']['data'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.astype(float)
        return df
    else:
        print("No hourly data found.")
        return None

def get_historical_data(symbol, days=90):
    """
    Fetch historical stock data for the given number of days using the Tradier API.
    :param symbol: Stock symbol (e.g., 'AAPL').
    :param days: Number of days of historical data to fetch.
    :return: DataFrame with historical stock data containing columns:
             'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    # Tradier API endpoint for historical market data
    endpoint = f"https://api.tradier.com/v1/markets/history"

    # Calculate the date range
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=days)

    # API request parameters
    params = {
        'symbol': symbol,
        'start': start_date.strftime('%Y-%m-%d'),
        'end': end_date.strftime('%Y-%m-%d'),
        'interval': 'daily'
    }

    # Send the GET request
    response = requests.get(endpoint, headers=HEADERS, params=params)
    #print(f"Historical Data Response Text: {response.text}")  # Print detailed error message

    # Handle errors in response
    if response.status_code != 200:
        print(f"Error: Unable to fetch data for {symbol}. HTTP Status Code: {response.status_code}")
        print(f"Response Text: {response.text}")
        return None

    data = response.json()

    # Check if the response contains the expected fields
    if 'history' not in data or 'day' not in data['history']:
        print(f"No historical data available for {symbol}. Response: {data}")
        return None

    # Convert the data to a DataFrame
    historical_data = pd.DataFrame(data['history']['day'])

    # Ensure the DataFrame has the required columns
    if not {'date', 'open', 'high', 'low', 'close', 'volume'}.issubset(historical_data.columns):
        print("Error: Missing required fields in the historical data.")
        return None

    historical_data['date'] = pd.to_datetime(historical_data['date'])
    historical_data = historical_data.set_index('date')

    return historical_data

def fetch_crypto_data(symbol, interval):
    """
    Fetch cryptocurrency data for the specified symbol and interval from Coinbase API.
    :param symbol: Cryptocurrency pair (e.g., 'BTC-USD').
    :param interval: Time interval for candlestick data (e.g., '1m', '5m').
    :return: DataFrame containing candlestick data or None if fetching fails.
    """
    base_url = "https://api.exchange.coinbase.com/products"
    granularity_map = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}

    if interval not in granularity_map:
        raise ValueError(f"Unsupported interval: {interval}. Supported intervals: {list(granularity_map.keys())}")

    granularity = granularity_map[interval]

    # Coinbase allows a maximum of 300 candles
    max_time_range = granularity * 300  # Maximum time range in seconds

    # Calculate valid start and end times
    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(seconds=max_time_range)

    params = {
        'start': start_time.isoformat() + 'Z',
        'end': end_time.isoformat() + 'Z',
        'granularity': granularity
    }

    request_path = f"/products/{symbol}/candles"
    url = f"{base_url}/{symbol}/candles"

    try:
        headers = generate_headers(request_path)
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data, columns=['time', 'low', 'high', 'open', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert epoch to datetime
                df.set_index('time', inplace=True)
                df = df.sort_index()  # Ensure data is in chronological order
                return df
            else:
                logging.error(f"Received empty data for {symbol}.")
                return None
        else:
            logging.error(f"Failed to fetch crypto data for {symbol}. Status: {response.status_code}, Response: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error fetching crypto data: {e}")
        return None

   
def fetch_stock_price(symbol):
    """
    Fetch the current stock price using the data fetching function.
    :return: Current price of the stock or None if unavailable.
    """
    intraday_data = get_intraday_data(symbol, interval='1min')
    if intraday_data is not None and not intraday_data.empty:
        logging.info(f"Current Price: {intraday_data['close'].iloc[-1]}")
        return intraday_data['close'].iloc[-1]
    else:
        return None

def fetch_crypto_price(symbol):
    """
    Fetch the current price of a cryptocurrency from Coinbase API.
    :param symbol: Cryptocurrency pair (e.g., 'BTC-USD').
    :return: Current price or None if unavailable.
    """
    base_url = f"https://api.exchange.coinbase.com/products/{symbol}/ticker"

    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            data = response.json()
            logging.info(f"Current Price: {data['price']}")
            return float(data['price'])
        else:
            logging.error(f"Failed to fetch crypto price for {symbol}: {response.text}")
            return None
    except Exception as e:
        logging.error(f"Error fetching crypto price: {e}")
        return None
            
def get_options_chain(symbol, expiration):
    """
    Fetch options chain for a given symbol and expiration date.
    """
    params = {
        'symbol': symbol,
        'expiration': expiration,
        'greeks': 'true'
    }
    response = requests.get(f"{BASE_URL}/markets/options/chains", params=params, headers=HEADERS)
    data = response.json()
    if 'options' in data and 'option' in data['options']:
        options = data['options']['option']
        df = pd.DataFrame(options)
        # Normalize the 'greeks' column
        if 'greeks' in df.columns:
            greeks_df = pd.json_normalize(df['greeks'])
            df = pd.concat([df.drop(columns=['greeks']), greeks_df], axis=1)
        else:
            print("Greeks data is missing in the API response.")
        return df
    else:
        print("No options data found.")
        return None

def get_option_quote(symbol):
    """
    Fetch the latest quote for a specific option symbol.
    """
    params = {'symbols': symbol, 'greeks': 'false'}
    response = requests.get(f"{BASE_URL}/markets/quotes", params=params, headers=HEADERS)
    data = response.json()
    if 'quotes' in data and 'quote' in data['quotes']:
        quote = data['quotes']['quote']
        if isinstance(quote, list):
            quote = quote[0]
        return float(quote.get('bid', 0)), float(quote.get('ask', 0))
    else:
        print(f"No quote data found for {symbol}.")
        return None, None