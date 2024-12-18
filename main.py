#!/usr/bin/env python3

from Strategies.swing_trading import SwingTradingStrategy
import datetime


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
        dnn_model_path = f"Models/dnn_cryptoscalper.keras"
    else:
        symbol_type = "stock"
        dnn_model_path = f"Models/dnn_stockscalper.keras"
    
    #dnn_model_path = f"Models/dnn_model_{symbol}.keras"
    #dnn_model_path = f"Models/dnn_scalper.keras"
    sgd_model_path = f"Models/sgd_model_{symbol}.pkl"

    # Initialize and run the strategy
    strategy_params = {
        'oversold_level': 30,
        'overbought_level': 70,
        'breakout_threshold': 0.005,
        'scalping_threshold': 0.01,
        'scalping_profit_target': 0.02,
        'scalping_stop_loss': 0.01
    }

    swing_trader = SwingTradingStrategy(symbol, strategy_params, symbol_type, dnn_model_path, sgd_model_path)
    swing_trader.run()