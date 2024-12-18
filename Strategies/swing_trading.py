import pandas as pd
import requests
import os
import logging
from utils import is_market_open
import time
import datetime
from Data.ml_trading import load_dnn_model, predict_signal, train_dnn
from Data.data_fetching import get_intraday_data, get_historical_data, fetch_crypto_data, fetch_crypto_price, fetch_stock_price
from Data.technical_indicators import calculate_RSI
import json
'''
    Optional: Discord bot
'''
from config import notify_discord

import warnings
# Suppress the specific UserWarning
warnings.filterwarnings("ignore", message="X does not have valid feature names, but MinMaxScaler was fitted with feature names")
warnings.filterwarnings("ignore", message="Starting a Matplotlib GUI outside of the main thread will likely fail.")
warnings.filterwarnings("ignore", message="A value is trying to be set on a copy of a slice from a DataFrame.")

class SwingTradingStrategy:
    def __init__(self, symbol, strategy_params, symbol_type, dnn_model_path, sgd_model_path):
        self.symbol = symbol
        self.symbol_type = symbol_type  # 'stock' or 'crypto'
        self.strategy_params = strategy_params
        self.dnn_model_path = dnn_model_path
        self.sgd_model_path = sgd_model_path
        self.dnn_model = None
        self.sgd_model = None
        self.capital = 10000  # Total capital
        self.positions = []  # To track open trades
        self.total_profit = 0.0  # Total profit across all trades
        self.total_trades = 0  # Count of total trades executed
        
        # Ensure the tradelog directory exists
        os.makedirs("tradelog", exist_ok=True)
        os.makedirs("Models", exist_ok=True)
        
        date_traded = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Set log file path to the tradelog folder
        if symbol_type == 'crypto':
            log_file = os.path.join("tradelog", f"{date_traded}_{symbol}.log")
        else:
            log_file = os.path.join("tradelog", f"{date_traded}_{symbol}.log")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,  # Set logging level
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),  # Log to file
                logging.StreamHandler()        # Log to console
            ]
        )
        
    def load_or_train_models(self, candles_df):
        """
        Load models if they exist; otherwise, train them.
        """
        if candles_df is None or candles_df.empty:
            logging.error("No data available to train models.")
            return
        
        # Log Shape of Data
        logging.info(f"Data for training models: {candles_df.shape}")
        
        # Load DNN model
        logging.info(f"path: {self.dnn_model_path}")
        self.dnn_model = load_dnn_model(self.dnn_model_path)
        #self.dnn_model = train_dnn(candles_df, self.dnn_model_path, self.symbol_type)
        
        if self.dnn_model == None:
            logging.info("Model not found. Training new model...")
            self.dnn_model = train_dnn(candles_df, self.dnn_model_path)
        
        logging.info("Loaded DNN model.")
        
    def load_fetch_counter(self):
        try:
            with open(f"JSON/{self.dnn_model_path}_fetch_counter.json", "r") as f:
                counter_data = json.load(f)
                return counter_data.get("fetch_count", 0)
        except (FileNotFoundError, json.JSONDecodeError):
            return 0  # Default to 0 if file doesn't exist or is invalid


    def save_fetch_counter(self, count):
        with open(f"JSON/{self.dnn_model_path}_fetch_counter.json", "w") as f:
            json.dump({"fetch_count": count}, f)

                    
    def fetch_data(self):
        try:
            if self.symbol_type == 'crypto':
                data = fetch_crypto_data(self.symbol, interval='1m')
            elif self.symbol_type == 'stock':
                #logging.info("Fetching historical and intraday data for stock.")
                
                # Fetch historical and intraday data
                historical_data = get_historical_data(self.symbol, days=90)
                intraday_data = get_intraday_data(self.symbol, interval='1min')
                
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


    def fetch_price(self):
        """
        Fetch the current price based on the symbol type.
        :return: Current price or None if unavailable.
        """
        if self.symbol_type == 'crypto':
            return fetch_crypto_price(self.symbol)
        elif self.symbol_type == 'stock':
            return fetch_stock_price(self.symbol)
        else:
            raise ValueError(f"Unsupported symbol type: {self.symbol_type}")
    
    
    def analyze_data(self, candles_df):
        
        #print(candles_df)
        """
        Use both models to analyze data and determine the final trading action.
        """
        # Get predictions from both models
        signal = predict_signal(candles_df, self.dnn_model, 'dnn')

        # Combine predictions numerically
        final_signal = ""

        # Decide final action based on combined signal
        if signal == 1:  # Positive overall signal
            final_signal = 'buy'
        elif signal == -1:
            final_signal = 'hold'
        else:  # Neutral or opposing signals
            final_signal = 'sell'

        logging.info(f"RSI: {round(calculate_RSI(candles_df['close'], period=14).iloc[-1])} Final: {final_signal}")
        return final_signal, candles_df['close'].iloc[-1]


    def execute_trade(self, signal, price):
        """
        Execute a buy or sell trade based on the signal.
        """
        if signal == 'buy':
            if self.symbol_type == 'crypto':
                amount = round(float(self.capital / price), 8)  # Fractional trading for crypto
            elif self.symbol_type == 'stock':
                amount = int(self.capital / price)  # Whole shares for stocks
            else:
                raise ValueError(f"Unsupported symbol type: {self.symbol_type}")

            if amount > 0:
                target_price = float(price * (1 + self.strategy_params['scalping_profit_target']))
                stop_loss_price = float(price * (1 - self.strategy_params['scalping_stop_loss']))
                trade = {
                    'type': 'buy',
                    'price': float(price),
                    'shares': float(amount),
                    'target_price': target_price,
                    'stop_loss_price': stop_loss_price
                }
                self.positions.append(trade)
                self.capital -= price * amount
                self.total_trades += 1
                logging.info(f"BUY EXECUTED: {amount} units at ${price:.3f}. "
                            f"Target=${target_price:.3f}, Stop-Loss=${stop_loss_price:.3f}")
                
                # Discord Notif
                notify_discord(f"**{self.symbol}** BUY EXECUTED: {round(amount)} units at ${price:.3f}  Target=${target_price:.3f}, Stop-Loss=${stop_loss_price:.3f}")
                
        elif signal == 'sell':
            for trade in self.positions[:]:
                if trade['type'] == 'buy':
                    # Calculate profit/loss for this trade
                    profit_loss = float((price - trade['price']) * trade['shares'])
                    percent_change = float(((price - trade['price']) / trade['price']) * 100)
                    profit = float((price - trade['price']) * trade['shares'])

                    # Update capital and total profit
                    self.total_profit += profit
                    self.capital += price * trade['shares']
                    self.positions.remove(trade)

                    logging.info(f"SELL EXECUTED: {trade['shares']} units at ${price:.2f}.  "
                                f"Profit/Loss: ${profit_loss:.2f} ({percent_change:.2f}%).\n")
                    
                    # Discord Notif
                    notify_discord(f"**{self.symbol}** SELL EXECUTED: {round(trade['shares'])} units at ${price:.3f}. "
                                f"Profit/Loss: ${profit_loss:.3f} ({percent_change:.2f}%).")
                    
        elif signal == 'hold':
            logging.info(f"Holding positions")
        


    def check_positions(self, current_price):
        """
        Check all open positions to see if they meet target or stop-loss criteria.
        """
        logging.info(f"Current Price: ${float(current_price):.5f}.")
        
        for trade in self.positions[:]:
            if trade['type'] == 'buy':
                # Calculate current profit/loss
                current_profit_loss = float((current_price - trade['price']) * trade['shares'])
                percent_change = float(((current_price - trade['price']) / trade['price']) * 100)

                # Log the current profit/loss
                logging.info(
                    f"Position: {trade['shares']:.2f} shares bought at ${trade['price']:.2f}. "
                    f"Profit/Loss: ${current_profit_loss:.2f} ({percent_change:.2f}%).\n"
                )
                
                if current_price >= trade['target_price']:
                    logging.info(f"Target met for {trade['shares']} shares at ${current_price:.5f}. Consider Selling.")
                    self.execute_trade('sell', current_price)
                elif current_price <= trade['stop_loss_price']:
                    logging.info(f"Stop-loss hit for {trade['shares']} shares at ${current_price:.2f}. Selling.")
                    self.execute_trade('sell', current_price)

        
    def run(self):
        """
        Run the trading strategy with live chart visualization.
        """
        fetch_count = self.load_fetch_counter()
        retrain_interval = 50  # Retrain DNN model after 50 fetches

        while self.symbol_type == 'stock' and not is_market_open():
            logging.info("Market is closed. Skipping trading.")
            time.sleep(300)  # Wait for 5 minutes
            continue
        
        logging.info(f"Starting {self.symbol_type.capitalize()} Swing Trading Strategy for {self.symbol}")
        
        # Fetch initial data
        candles_df = self.fetch_data()
        if candles_df is None or candles_df.empty:
            logging.error("Failed to fetch initial data. Exiting.")
            return
        
        #print(f"candles_df: {candles_df}")
        
    
        self.load_or_train_models(candles_df)
        
        
        while True:  # Continuous trading loop for crypto or restricted loop for stocks
            try:
                if self.symbol_type == 'stock' and not is_market_open():
                    logging.info("Market is closed. Skipping trading.")
                    time.sleep(300)  # Wait for 5 minutes
                    continue

                # Fetch new data
                new_data = self.fetch_data()
                
                #print(f"index 0: {new_data.index[0]}")
                
                if new_data is not None and not new_data.empty:
                    
                    #logging.info(f"Fetched new data:\n{new_data}")
                                 
                    fetch_count += 1
                    self.save_fetch_counter(fetch_count)
                    
                    # Ensure new_data and candles_df use integer indices
                    
                    candles_df = pd.concat([candles_df, new_data]).drop_duplicates().sort_index()
                    if len(candles_df) > 500:
                        candles_df = candles_df.iloc[-1000:]
                    
                    #logging.info(f"candles_df:\n{candles_df}")
                    
                    '''
                    # Retrain DNN after a specified number of fetches
                    if fetch_count % retrain_interval == 0:
                        logging.info(f"Retraining DNN model after {fetch_count} fetches.")
                        self.dnn_model = train_dnn(candles_df, self.dnn_model_path, self.symbol_type)
                        logging.info("DNN model retrained successfully.")
                    '''
                    
                    # Analyze data and execute trades
                    signal, decision_price = self.analyze_data(candles_df)
                    if signal in ['buy', 'hold', 'sell'] and decision_price is not None:
                        self.execute_trade(signal, decision_price)
                        
                    current_price = new_data['close'].iloc[-1]
                    self.check_positions(current_price)

                    # Log current stats
                    logging.info(f"Current Capital: ${self.capital:.2f}, Total Profit: ${self.total_profit:.2f}, Total Trades: {self.total_trades} \n")
                    for trade in self.positions:
                        logging.info(f"Open Trade: {trade}\n")
                        
                    
                else:
                    logging.error("Failed to fetch data or data is empty.")

                time.sleep(10)  # Adjust interval for updates
                
            except Exception as e:
                logging.error(f"Error during trading loop: {e}")