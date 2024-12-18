# patterns.py

import pandas as pd

def detect_patterns(candles_df):
    """
    Detects candlestick patterns from the given DataFrame of candlestick data.

    :param candles_df: DataFrame containing 'open', 'high', 'low', 'close' columns.
    :return: DataFrame with detected patterns.
    """
    patterns = []

    for i in range(2, len(candles_df)):
        prev_prev = candles_df.iloc[i - 2]  # Two bars back
        prev = candles_df.iloc[i - 1]  # One bar back
        current = candles_df.iloc[i]  # Current bar

        def evaluate_pattern(case):
            match case:
                case '2-1-2 Bullish Reversal':
                    return (
                        prev_prev['close'] < prev_prev['open'] and  # Previous-2 is red (2d)
                        prev['high'] <= prev['low'] and            # Previous-1 is an inside bar (1)
                        current['close'] > current['open']         # Current is green (2u)
                    )
                case '2-1-2 Bearish Reversal':
                    return (
                        prev_prev['close'] > prev_prev['open'] and  # Previous-2 is green (2u)
                        prev['high'] <= prev['low'] and            # Previous-1 is an inside bar (1)
                        current['close'] < current['open']         # Current is red (2d)
                    )
                case '2-2 Bullish Reversal':
                    return (
                        prev_prev['close'] < prev_prev['open'] and  # Previous-2 is red (2d)
                        prev['close'] > prev['open'] and           # Previous-1 is green (2u)
                        current['close'] > current['open']         # Current is green (2u)
                    )
                case '2-2 Bearish Reversal':
                    return (
                        prev_prev['close'] > prev_prev['open'] and  # Previous-2 is green (2u)
                        prev['close'] < prev['open'] and           # Previous-1 is red (2d)
                        current['close'] < current['open']         # Current is red (2d)
                    )
                case '1-2-2 Bullish Reversal':
                    return (
                        prev_prev['high'] <= prev_prev['low'] and   # Previous-2 is an inside bar (1)
                        prev['close'] < prev['open'] and           # Previous-1 is red (2d)
                        current['close'] > current['open']         # Current is green (2u)
                    )
                case '1-2-2 Bearish Reversal':
                    return (
                        prev_prev['high'] <= prev_prev['low'] and   # Previous-2 is an inside bar (1)
                        prev['close'] > prev['open'] and           # Previous-1 is green (2u)
                        current['close'] < current['open']         # Current is red (2d)
                    )
                case '3-2 Bullish Reversal':
                    return (
                        prev_prev['high'] > prev_prev['low'] and    # Previous-2 is an outside bar (3)
                        prev['close'] > prev['open'] and           # Previous-1 is green (2u)
                        current['close'] > current['open']         # Current is green (2u)
                    )
                case '3-2 Bearish Reversal':
                    return (
                        prev_prev['high'] > prev_prev['low'] and    # Previous-2 is an outside bar (3)
                        prev['close'] < prev['open'] and           # Previous-1 is red (2d)
                        current['close'] < current['open']         # Current is red (2d)
                    )
                case '1 Bar Rev Strat Bullish Reversal':
                    return (
                        prev['close'] < prev['open'] and           # Previous-1 is red (2d)
                        current['close'] > current['open'] and     # Current is green (2u)
                        current['close'] > prev['high']            # Current close exceeds previous high
                    )
                case '1 Bar Rev Strat Bearish Reversal':
                    return (
                        prev['close'] > prev['open'] and           # Previous-1 is green (2u)
                        current['close'] < current['open'] and     # Current is red (2d)
                        current['close'] < prev['low']             # Current close falls below previous low
                    )
                case '2-2-2 Bullish Randy Jackson':
                    return (
                        prev_prev['close'] < prev_prev['open'] and  # Previous-2 is red (2d)
                        prev['close'] > prev['open'] and           # Previous-1 is green (2u)
                        current['close'] > current['open']         # Current is green (2u)
                    )
                case '2-2-2 Bearish Randy Jackson':
                    return (
                        prev_prev['close'] > prev_prev['open'] and  # Previous-2 is green (2u)
                        prev['close'] < prev['open'] and           # Previous-1 is red (2d)
                        current['close'] < current['open']         # Current is red (2d)
                    )
                case '2-1-2 Bullish Continuation':
                    return (
                        prev_prev['close'] > prev_prev['open'] and  # Previous-2 is green (2u)
                        prev['high'] <= prev['low'] and            # Previous-1 is an inside bar (1)
                        current['close'] > current['open']         # Current is green (2u)
                    )
                case '2-1-2 Bearish Continuation':
                    return (
                        prev_prev['close'] < prev_prev['open'] and  # Previous-2 is red (2d)
                        prev['high'] <= prev['low'] and            # Previous-1 is an inside bar (1)
                        current['close'] < current['open']         # Current is red (2d)
                    )
                case '2-2 Bullish Continuation':
                    return (
                        prev_prev['close'] > prev_prev['open'] and  # Previous-2 is green (2u)
                        prev['close'] > prev['open'] and           # Previous-1 is green (2u)
                        current['close'] > current['open']         # Current is green (2u)
                    )
                case '2-2 Bearish Continuation':
                    return (
                        prev_prev['close'] < prev_prev['open'] and  # Previous-2 is red (2d)
                        prev['close'] < prev['open'] and           # Previous-1 is red (2d)
                        current['close'] < current['open']         # Current is red (2d)
                    )
                case '3-2-2 Bullish Reversal':
                    return (
                        prev_prev['high'] > prev_prev['low'] and    # Previous-2 is an outside bar (3)
                        prev['close'] < prev['open'] and           # Previous-1 is red (2d)
                        current['close'] > current['open']         # Current is green (2u)
                    )
                case '3-2-2 Bearish Reversal':
                    return (
                        prev_prev['high'] > prev_prev['low'] and    # Previous-2 is an outside bar (3)
                        prev['close'] > prev['open'] and           # Previous-1 is green (2u)
                        current['close'] < current['open']         # Current is red (2d)
                    )
                case '3-1-2 Bullish Reversal':
                    return (
                        prev_prev['high'] > prev_prev['low'] and    # Previous-2 is an outside bar (3)
                        prev['high'] <= prev['low'] and            # Previous-1 is an inside bar (1)
                        current['close'] > current['open']         # Current is green (2u)
                    )
                case '3-1-2 Bearish Reversal':
                    return (
                        prev_prev['high'] > prev_prev['low'] and    # Previous-2 is an outside bar (3)
                        prev['high'] <= prev['low'] and            # Previous-1 is an inside bar (1)
                        current['close'] < current['open']         # Current is red (2d)
                    )
                case 'Bullish Doji':
                    # Doji criteria + Downtrend
                    body_size = abs(current['close'] - current['open'])
                    candle_range = current['high'] - current['low']
                    downtrend = prev['close'] < prev_prev['close'] and current['low'] < prev['low']
                    return (
                        candle_range > 0 and  # Avoid division by zero
                        body_size / candle_range < 0.1 and  # Standard Doji criteria
                        downtrend  # Must appear after a downtrend
                    )
                case 'Bearish Doji':
                    # Doji criteria + Uptrend
                    body_size = abs(current['close'] - current['open'])
                    candle_range = current['high'] - current['low']
                    uptrend = prev['close'] > prev_prev['close'] and current['high'] > prev['high']
                    return (
                        candle_range > 0 and  # Avoid division by zero
                        body_size / candle_range < 0.1 and  # Standard Doji criteria
                        uptrend  # Must appear after an uptrend
                    )
                case 'Bullish Hammer':
                    body_size = abs(current['close'] - current['open'])
                    lower_shadow = current['open'] - current['low'] if current['close'] > current['open'] else current['close'] - current['low']
                    upper_shadow = current['high'] - max(current['open'], current['close'])
                    return (
                        lower_shadow > 2 * body_size and  # Long lower shadow
                        upper_shadow < 0.5 * body_size and  # Small or no upper shadow
                        body_size / (current['high'] - current['low']) < 0.5  # Small real body
                    )
                case 'Bullish Engulfing':
                    return (
                        prev['close'] < prev['open'] and  # Previous candle is red
                        current['close'] > current['open'] and  # Current candle is green
                        current['open'] <= prev['close'] and  # Current open is below or equal to previous close
                        current['close'] >= prev['open']  # Current close is above or equal to previous open
                    )
                case 'Bearish Engulfing':
                    return (
                        prev['close'] > prev['open'] and  # Previous candle is green
                        current['close'] < current['open'] and  # Current candle is red
                        current['open'] >= prev['close'] and  # Current open is above or equal to previous close
                        current['close'] <= prev['open']  # Current close is below or equal to previous open
                    )               
                case _:  # Default case
                    return False

        # Define all pattern names to check
        pattern_names = [
            '2-1-2 Bullish Reversal', '2-1-2 Bearish Reversal',
            '2-2 Bullish Reversal', '2-2 Bearish Reversal',
            '3-2-2 Bullish Reversal', '3-2-2 Bearish Reversal',
            '3-1-2 Bullish Reversal', '3-1-2 Bearish Reversal',
            '1-2-2 Bullish Reversal', '1-2-2 Bearish Reversal',
            '3-2 Bullish Reversal', '3-2 Bearish Reversal',
            '1 Bar Rev Strat Bullish Reversal', '1 Bar Rev Strat Bearish Reversal',
            '2-2-2 Bullish Randy Jackson', '2-2-2 Bearish Randy Jackson',
            '2-1-2 Bullish Continuation', '2-1-2 Bearish Continuation',
            '2-2 Bullish Continuation', '2-2 Bearish Continuation',
            'Bullish Doji', 'Bearish Doji', 'Bullish Hammer',
            'Bullish Engulfing', 'Bearish Engulfing'
        ]

        for pattern_name in pattern_names:
            if evaluate_pattern(pattern_name):
                patterns.append((i, pattern_name))
                break

    # Create a DataFrame for the patterns
    patterns_df = pd.DataFrame(patterns, columns=['Index', 'Pattern'])
    return patterns_df