# utils.py

import datetime
from config import MARKET_OPEN, MARKET_CLOSE

def is_market_open():
    now = datetime.datetime.now()
    market_open = now.replace(hour=MARKET_OPEN[0], minute=MARKET_OPEN[1], second=0, microsecond=0)
    market_close = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1], second=0, microsecond=0)
    return market_open <= now <= market_close
