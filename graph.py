#graph.py

import pandas as pd
import datetime
import time
from utils import is_market_open
from Data.chart import live_chart
from Data.data_fetching import get_historical_data, get_intraday_data
from Strategies.swing_trading import fetch_crypto_data
from queue import Queue
import threading
import logging

def fetch_data(symbol, symbol_type):
    try:
        if symbol_type == 'crypto':
            data = fetch_crypto_data(symbol, interval='1m')
        elif symbol_type == 'stock':
            #logging.info("Fetching historical and intraday data for stock.")
            
            # Fetch historical and intraday data
            historical_data = get_historical_data(symbol, days=90)
            intraday_data = get_intraday_data(symbol, interval='1min')
            
            # Log the shape of data returned
            if historical_data is not None:
                logging.info(f"Historical Data Shape: {historical_data.shape}")
                # Convert historical data index to datetime64[ns] with time at 00:00:00
                historical_data.index = pd.to_datetime(historical_data.index, errors='coerce').normalize()
            else:
                logging.error("No historical data fetched.")
            
            if intraday_data is not None:
                logging.info(f"Intraday Data Shape: {intraday_data.shape}")
                # Ensure intraday data index is datetime64[ns]
                intraday_data.index = pd.to_datetime(intraday_data.index, errors='coerce')
            else:
                logging.error("No intraday data fetched.")
            
            # Concatenate data, ensuring consistent datetime indices
            data_frames = [df for df in [historical_data, intraday_data] if df is not None and not df.empty]
            if data_frames:
                data = pd.concat(data_frames).drop_duplicates().sort_index()
            else:
                logging.error("No valid data frames to concatenate.")
                return None

        # Ensure data is valid
        if data is None or data.empty or len(data) < 20:
            logging.error("Fetched data is insufficient for processing.")
            return None

        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logging.error(f"Data missing required columns: {missing_columns}")
            return None

        #logging.info(f"Fetched data shape: {data.shape}")
        return data
    except Exception as e:
        logging.error(f"Error in fetch_data: {e}", exc_info=True)
        return None
        
if __name__ == "__main__":
    
    # Ask for input
    symbol = input("Enter symbol: ")
    while symbol == "":
        symbol_key = input('Invalid input. Symbol cannot be empty').strip().lower()

    # 's' for stock or 'c' for crypto
    symbol_key = input('Enter "s" for stock or "c" for crypto: ').strip().lower()

    # Validate input
    while symbol_key not in ("s", "c"):
        symbol_key = input('Invalid input. You must enter "s" for stock or "c" for crypto: ').strip().lower()

    # Determine symbol type and model paths
    if symbol_key == "c":
        symbol_type = "crypto"
    else:
        symbol_type = "stock"
    
    # Fetch initial data
    candles_df = fetch_data(symbol, symbol_type)
    if candles_df is None or candles_df.empty:
        logging.error("Failed to fetch initial data. Exiting.")
        exit
        
    """
        Candlestick Charting
    """
    # Initialize shared queue for live chart updates
    shared_queue = Queue()
    # Add initial data to the queue
    shared_queue.put(candles_df)
    # Start live chart in a separate thread
    chart_thread = threading.Thread(target=live_chart, args=(shared_queue, symbol), daemon=True)
    chart_thread.start()
    
    while True:  # Continuous trading loop for crypto or restricted loop for stocks
        try:
            if symbol_type == 'stock' and not is_market_open():
                logging.info("Market is closed. Skipping trading.")
                time.sleep(300)  # Wait for 5 minutes
                continue

            # Fetch new data
            new_data = fetch_data(symbol, symbol_type)
            
            if new_data is not None and not new_data.empty:
                shared_queue.put(new_data)
                
            else:
                logging.error("Failed to fetch data or data is empty.")

            time.sleep(5)  # Adjust interval for updates
            
        except Exception as e:
            logging.error(f"Error during trading loop: {e}")