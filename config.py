# config.py

import os
import jwt
from cryptography.hazmat.primitives import serialization
import time
import secrets
from dotenv import load_dotenv, dotenv_values 

load_dotenv() 

# API Configuration
API_TOKEN = os.getenv("TRADIER_API")  # Replace with your actual Tradier API token

BASE_URL = 'https://api.tradier.com/v1'
HEADERS = {
    'Authorization': f'Bearer {API_TOKEN}',
    'Accept': 'application/json'
}
DB_PATH = 'Data/Database/options_data.db'

#Coinbase
COIN_API_KEY = os.getenv("COIN_BASE_API_KEY")  # Replace with your actual Coinbase API
COIN_API_SECRET = os.getenv("COIN_BASE_API_SECRET")

# Define the request
request_method = 'GET'
request_host = "api.coinbase.com"
#request_path = "/api/v3/brokerage/products/"


def notify_discord(message):
    """
        Optional: for sending out trade alerts on discord
    """
    notif = "@everyone" # Change who to ping
    with open("trade_notifications.txt", "a") as file:
        file.write(notif + " " + message + "\n")
        
def build_jwt(uri):
    private_key_bytes = COIN_API_SECRET.encode('utf-8')
    private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
    jwt_payload = {
        'sub': COIN_API_KEY,
        'iss': "cdp",
        'nbf': int(time.time()),
        'exp': int(time.time()) + 120,
        'uri': uri,
    }
    jwt_token = jwt.encode(
        jwt_payload,
        private_key,
        algorithm='ES256',
        headers={'kid': COIN_API_KEY, 'nonce': secrets.token_hex()},
    )
    return jwt_token

# Generate the required headers for authenticated requests
def generate_headers(request_path):
    """
    Generate headers dynamically for each API request.
    :param request_path: The API endpoint path (e.g., '/products/BTC-USD/candles').
    :return: Dictionary containing headers.
    """
    try:
        uri = f"{request_method} {request_host}{request_path}"
        jwt_token = build_jwt(uri)
        return {
            'Authorization': f'Bearer {jwt_token}',
            'CB-ACCESS-KEY': COIN_API_KEY,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
    except Exception as e:
        print(f"Error generating headers: {e}")
        return {}

# Market hours (Eastern Time)
MARKET_OPEN = (9, 30)
MARKET_CLOSE = (16, 0)

# Create 'tradelog' directory if it doesn't exist
LOG_DIR = 'tradelog'
os.makedirs(LOG_DIR, exist_ok=True)
