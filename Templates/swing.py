from Strategies.stock_0dte_strategy import Stock0dteStrategy
from Strategies.swing_trading import SwingTradingStrategy
import datetime

#Swing trading
if __name__ == "__main__":
    
    symbol = 'FOXO'
    
    strategy_params = {
        'oversold_level': 30,
        'overbought_level': 70,
        'breakout_threshold': 0.005,
        'scalping_threshold': 0.01,
        'scalping_profit_target': 0.02,
        'scalping_stop_loss': 0.01
        #'atr_multiplier': 2,
    }
    swing_trader = SwingTradingStrategy(symbol, strategy_params, "stock")
    swing_trader.run()

