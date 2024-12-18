# technical_indicators.py

import pandas as pd
import numpy as np
import logging

def calculate_RSI(series, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    RSI = 100 - (100 / (1 + rs))
    return RSI

def calculate_bollinger_bands(series, period=20, num_std=2):
    """
    Calculate Bollinger Bands.
    """
    rolling_mean = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper_band = rolling_mean + (num_std * rolling_std)
    lower_band = rolling_mean - (num_std * rolling_std)
    return rolling_mean, upper_band, lower_band

def calculate_MACD(series, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD).
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    MACD = ema_fast - ema_slow
    signal_line = MACD.ewm(span=signal_period, adjust=False).mean()
    return MACD, signal_line

def calculate_moving_average(series, window=5):
    """
    Calculate the moving average.
    :param series: Pandas Series of prices.
    :param window: Lookback period for moving average.
    :return: Pandas Series of moving average values.
    """
    return series.rolling(window=window).mean()

def calculate_ema(series, period):
    """
    Calculate the Exponential Moving Average (EMA) for a given time series.
    
    :param series: Pandas Series of prices.
    :param span: The span of the EMA, which determines the smoothing factor.
    :return: Pandas Series of EMA values.
    """
    if series.empty:
        raise ValueError("The input series for EMA calculation is empty.")

    return series.ewm(span=period, adjust=False).mean()

def is_breakout(data, threshold=0.005):
    """
    Detect a breakout from the current price relative to the moving average.
    :param data: DataFrame containing 'Close' prices and moving averages.
    :param threshold: Percentage above/below the moving average to signal a breakout.
    :return: Boolean indicating a breakout condition.
    """
    moving_avg = data['close'].rolling(window=5).mean()
    breakout_up = data['close'].iloc[-1] > (1 + threshold) * moving_avg.iloc[-1]
    breakout_down = data['close'].iloc[-1] < (1 - threshold) * moving_avg.iloc[-1]
    return breakout_up, breakout_down

def calculate_vwap(data):
    """
    Calculate the VWAP for intraday data.

    :param data: DataFrame containing 'High', 'Low', 'Close', and 'Volume'.
    :return: DataFrame with an additional 'VWAP' column.
    """
    # Ensure data contains the necessary columns
    if not {'high', 'low', 'close', 'volume'}.issubset(data.columns):
        raise ValueError("Data must contain 'high', 'low', 'close', and 'volume' columns.")

    data = data.copy()
    
    # Calculate typical price
    data['Typical Price'] = (data['high'] + data['low'] + data['close']) / 3

    # Calculate cumulative values
    data['Cumulative TP * Volume'] = (data['Typical Price'] * data['volume']).cumsum()
    data['Cumulative Volume'] = data['volume'].cumsum()

    # Calculate VWAP
    data['VWAP'] = data['Cumulative TP * Volume'] / data['Cumulative Volume']

    return data

# Candlestick pattern detection functions
def is_hammer(data):
    """
    Identify hammer patterns for the given DataFrame slice or row.
    :param data: DataFrame slice with 'open', 'close', 'high', 'low'.
    :return: Boolean scalar indicating if the last candlestick is a hammer.
    """
    if len(data) < 1:
        return False  # Not enough data to evaluate

    body = abs(data['close'] - data['open'])  # Body size
    lower_shadow = abs(data['low'] - data[['open', 'close']].min(axis=1))  # Lower shadow
    upper_shadow = abs(data['high'] - data[['open', 'close']].max(axis=1))  # Upper shadow

    # Hammer criteria: lower shadow > 2 * body, upper shadow < body
    is_hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
    return is_hammer.iloc[-1]  # Return scalar boolean for the last row


def is_doji(data):
    """
    Identify if the candlestick pattern is a Doji.
    """
    if data.empty or len(data) != 1:
        return 0  # Default to 0 for invalid or insufficient data

    try:
        open_price = data['open'].iloc[0]
        close_price = data['close'].iloc[0]
        body_size = abs(open_price - close_price)
        avg_body_size = abs(data['high'] - data['low']).mean() * 0.1
        return int(body_size <= avg_body_size)
    except Exception as e:
        logging.error(f"Error in is_doji: {e}")
        return 0


def is_bullish_engulfing(data):
    """
    Detect if the latest candlestick forms a Bullish Engulfing pattern.

    :param data: DataFrame containing 'open', 'high', 'low', 'close'.
    Must have at least two rows (previous and current candle).
    :return: Boolean scalar indicating if the pattern is detected.
    """
    if len(data) < 2:
        return False  # Not enough data to detect pattern

    prev_row = data.iloc[-2]
    curr_row = data.iloc[-1]

    return (
        prev_row['close'] < prev_row['open'] and
        curr_row['close'] > curr_row['open'] and
        curr_row['close'] > prev_row['open'] and
        curr_row['open'] < prev_row['close']
    )


def is_bearish_engulfing(data):
    """
    Detect if the latest candlestick forms a Bearish Engulfing pattern.

    :param data: DataFrame containing 'open', 'high', 'low', 'close'.
    Must have at least two rows (previous and current candle).
    :return: Boolean scalar indicating if the pattern is detected.
    """
    if len(data) < 2:
        return False  # Not enough data to detect pattern

    prev_row = data.iloc[-2]
    curr_row = data.iloc[-1]

    return (
        prev_row['close'] > prev_row['open'] and
        curr_row['close'] < curr_row['open'] and
        curr_row['close'] < prev_row['open'] and
        curr_row['open'] > prev_row['close']
    )


def calculate_atr(data, period=14):
    """
    Calculate the Average True Range (ATR) for the given data.
    :param data: DataFrame containing 'High', 'Low', and 'Close' columns.
    :param period: The period over which to calculate the ATR.
    :return: Series containing ATR values.
    """
    # Ensure data contains the necessary columns
    if not {'high', 'low', 'close'}.issubset(data.columns):
        raise ValueError("Data must contain 'high', 'low', and 'close' columns.")

    # Work on a copy of the data to avoid SettingWithCopyWarning
    #data = data.copy()
    
    # Calculate True Range (TR)
    data['High-Low'] = data['high'] - data['low']
    data['High-Close'] = (data['high'] - data['close'].shift()).abs()
    data['Low-Close'] = (data['low'] - data['close'].shift()).abs()
    data['True Range'] = data[['High-Low', 'High-Close', 'Low-Close']].max(axis=1)

    # Calculate ATR
    atr = data['True Range'].rolling(window=period, min_periods=1).mean()

    # Clean up intermediate columns
    data.drop(['High-Low', 'High-Close', 'Low-Close', 'True Range'], axis=1, inplace=True)

    return atr

def calculate_parabolic_sar(data, af=0.02, af_max=0.2):
    """
    Calculate the Parabolic SAR for the given data.
    :param data: DataFrame containing 'high' and 'low' columns.
    :param af: Acceleration factor, typically starts at 0.02.
    :param af_max: Maximum acceleration factor, typically capped at 0.2.
    :return: Series containing Parabolic SAR values or NaN if data is insufficient.
    """
    if not {'high', 'low'}.issubset(data.columns):
        raise ValueError("Data must contain 'high' and 'low' columns.")

    if data.empty or len(data) < 2:
        logging.error("Insufficient data for Parabolic SAR calculation.")
        return pd.Series([float('nan')] * len(data), index=data.index)

    # Initialize variables
    sar = [data['low'].iloc[0]]  # SAR starts at the first Low
    ep = data['high'].iloc[0]  # EP (Extreme Price) starts at the first High
    af_current = af  # Acceleration Factor starts at the initial value
    uptrend = True  # Assume an initial uptrend

    for i in range(1, len(data)):
        prev_sar = sar[-1]
        prev_ep = ep

        if uptrend:
            sar_current = prev_sar + af_current * (prev_ep - prev_sar)
            ep = max(prev_ep, data['high'].iloc[i])
            if data['low'].iloc[i] < sar_current:  # Trend reversal
                uptrend = False
                sar_current = prev_ep
                ep = data['low'].iloc[i]
                af_current = af
        else:
            sar_current = prev_sar + af_current * (prev_ep - prev_sar)
            ep = min(prev_ep, data['low'].iloc[i])
            if data['high'].iloc[i] > sar_current:  # Trend reversal
                uptrend = True
                sar_current = prev_ep
                ep = data['high'].iloc[i]
                af_current = af

        # Adjust acceleration factor
        if uptrend:
            af_current = min(af_current + af, af_max) if ep > prev_ep else af_current
        else:
            af_current = min(af_current + af, af_max) if ep < prev_ep else af_current

        sar.append(sar_current)

    return pd.Series(sar, index=data.index)

def calculate_adl(candles_df):
    """
    Calculate Accumulation/Distribution Line (ADL) for a dataset.

    :param candles_df: DataFrame containing 'high', 'low', 'close', and 'volume'.
    :return: Latest ADL value.
    """
    try:
        money_flow_multiplier = (
            ((candles_df['close'] - candles_df['low']) - (candles_df['high'] - candles_df['close'])) /
            ((candles_df['high'] - candles_df['low']).replace(0, 1))  # Avoid division by zero
        )
        money_flow_volume = money_flow_multiplier * candles_df['volume']
        adl = money_flow_volume.cumsum()
        return adl.iloc[-1]
    except Exception as e:
        logging.error(f"Error calculating ADL: {e}")
        return None


def calculate_vold_ratio(candles_df):
    """
    Calculate the Volume Difference (VOLD) Ratio:
    Measures the ratio of advancing (up day) volume versus declining (down day) volume.

    :param candles_df: DataFrame containing 'close' and 'volume' columns.
    :return: VOLD ratio as a float.
    """
    try:
        # Ensure necessary columns exist
        if 'close' not in candles_df.columns or 'volume' not in candles_df.columns:
            raise ValueError("DataFrame must contain 'close' and 'volume' columns")

        # Identify advancing (up) and declining (down) days
        advancing_days = candles_df['close'] > candles_df['close'].shift(1)
        declining_days = candles_df['close'] < candles_df['close'].shift(1)

        # Calculate advancing and declining volumes
        advancing_volume = candles_df.loc[advancing_days, 'volume'].sum()
        declining_volume = candles_df.loc[declining_days, 'volume'].sum()

        # Handle edge cases
        if advancing_volume == 0 and declining_volume == 0:
            return 0.0  # Neutral market
        elif advancing_volume == 0:
            return -float('inf')  # Extreme bearish case
        elif declining_volume == 0:
            return float('inf')  # Extreme bullish case

        # Calculate the VOLD ratio
        vold_ratio = advancing_volume / declining_volume

        # Return rounded result
        return round(vold_ratio, 2)

    except Exception as e:
        logging.error(f"Error in calculate_vold_ratio: {e}", exc_info=True)
        return None



def calculate_ticks(candles_df):
    """
    Calculate Ticks (difference between up and down closes).

    :param candles_df: DataFrame containing 'close'.
    :return: Ticks value.
    """
    try:
        up_ticks = (candles_df['close'].diff() > 0).sum()
        down_ticks = (candles_df['close'].diff() < 0).sum()
        ticks = up_ticks - down_ticks
        return ticks
    except Exception as e:
        logging.error(f"Error calculating Ticks: {e}")
        return None