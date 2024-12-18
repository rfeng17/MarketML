# chart.py

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from Data.technical_indicators import (
    calculate_RSI,
    calculate_parabolic_sar
)

def live_chart(shared_queue, symbol):
    """
    Plot a live chart with indicators, updating every second.
    :param candles_df: Initial DataFrame with 'open', 'high', 'low', 'close', 'volume' columns.
    :param symbol: Trading symbol for data fetching.
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    candles_df = pd.DataFrame()
    
    # Helper function to calculate indicators
    def calculate_indicators(data):
        data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        data['SMA'] = data['close'].rolling(window=20).mean()
        data['Upper Band'] = data['SMA'] + 2 * data['close'].rolling(window=20).std()
        data['Lower Band'] = data['SMA'] - 2 * data['close'].rolling(window=20).std()
        data['EMA12'] = data['close'].ewm(span=12).mean()
        data['EMA26'] = data['close'].ewm(span=26).mean()
        data['MACD'] = data['EMA12'] - data['EMA26']
        data['Signal Line'] = data['MACD'].ewm(span=9).mean()
        data['SAR'] = calculate_parabolic_sar(data)
        data['RSI'] = calculate_RSI(data['close'], period=14)
        return data

    def convert_to_eastern_time(data):
        """
        Convert the time index of a DataFrame to Eastern Time (ET).
        :param data: DataFrame with a datetime index.
        :return: DataFrame with the index converted to ET.
        """
        eastern = pytz.timezone('US/Eastern')
        data = data.copy()  # Avoid modifying the original DataFrame
        data.index = data.index.tz_localize(pytz.utc).tz_convert(eastern)
        return data

    def update(frame):
        nonlocal candles_df

        try:
            if not shared_queue.empty():
                candles_df = shared_queue.get_nowait()

            if candles_df.empty:
                return  # Skip update if no data

            # Prepare data
            updated_data = candles_df.iloc[-50:].copy()
            updated_data = convert_to_eastern_time(updated_data)
            updated_data = calculate_indicators(updated_data)

            # Clear previous plots
            ax[0].clear()
            ax[1].clear()

            # Plot candlesticks on the first subplot
            for i in range(len(updated_data)):
                color = 'green' if updated_data['close'].iloc[i] >= updated_data['open'].iloc[i] else 'red'
                ax[0].plot(
                    [updated_data.index[i], updated_data.index[i]],
                    [updated_data['low'].iloc[i], updated_data['high'].iloc[i]],
                    color='black', linewidth=0.5
                )
                ax[0].plot(
                    [updated_data.index[i], updated_data.index[i]],
                    [updated_data['open'].iloc[i], updated_data['close'].iloc[i]],
                    color=color, linewidth=3
                )

            # Plot indicators
            ax[0].plot(updated_data.index, updated_data['VWAP'], label='VWAP', color='blue', linewidth=1)
            ax[0].plot(updated_data.index, updated_data['Upper Band'], label='Upper Band', color='orange', linestyle='--')
            ax[0].plot(updated_data.index, updated_data['Lower Band'], label='Lower Band', color='orange', linestyle='--')
            ax[0].scatter(updated_data.index, updated_data['SAR'], label='Parabolic SAR', color='purple', s=10)

            # Format first subplot
            ax[0].set_title(f'{symbol} Live Chart (Eastern Time)')
            ax[0].set_xlabel('Time (ET)')
            ax[0].set_ylabel('Price')
            ax[0].legend(loc='upper left')
            ax[0].grid()

            # Plot RSI on the second subplot
            ax[1].plot(updated_data.index, updated_data['RSI'], label='RSI', color='green', linewidth=1)
            ax[1].axhline(70, color='red', linestyle='--', linewidth=0.8)  # Overbought line
            ax[1].axhline(30, color='blue', linestyle='--', linewidth=0.8)  # Oversold line

            # Format second subplot
            ax[1].set_title('RSI')
            ax[1].set_xlabel('Time (ET)')
            ax[1].set_ylabel('RSI')
            ax[1].legend(loc='upper left')
            ax[1].grid()

        except Exception as e:
            print(f"Error in live chart update: {e}")


    # Set up animation
    ani = FuncAnimation(fig, update, interval=1000, cache_frame_data=False)  # Update every 1000 milliseconds (1 second)
    plt.show()