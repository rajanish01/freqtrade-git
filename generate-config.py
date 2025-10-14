import json
import os

print("🔍 Checking environment variables...")
required_vars = ["BINANCE_API_KEY", "BINANCE_SECRET_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
missing_vars = []

for var in required_vars:
    value = os.getenv(var)
    if value:
        print(f"✅ {var}: {'*' * len(value)}")
    else:
        print(f"❌ {var}: NOT SET")
        missing_vars.append(var)

if missing_vars:
    print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
    print("⚠️  Continuing with default values...")

# Check database URL
db_url = os.getenv("DATABASE_URL")
if db_url:
    print(f"✅ DATABASE_URL: {'*' * 40}")
    if "postgresql" in db_url:
        print("✅ PostgreSQL database detected")
    else:
        print("⚠️  Non-PostgreSQL database detected")
else:
    print("❌ DATABASE_URL: NOT SET - using SQLite")

# Define default values and read from env vars
config = {
    "$schema": "https://schema.freqtrade.io/schema.json",
    "stake_currency": "USDT",
    "stake_amount": "unlimited",
    "tradable_balance_ratio": 0.99,
    "dry_run": True,
    "dry_run_wallet": 1000,
    "cancel_open_orders_on_exit": True,
    "trading_mode": "spot",
    "margin_mode": "isolated",
    "unfilledtimeout": {
        "entry": 110,
        "exit": 180,
        "exit_timeout_count": 0,
        "unit": "minutes"
    },
    "order_types": {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 5,
        "stoploss_price_type": "mark",
        "stoploss_on_exchange_limit_ratio": 0.99
    },
    "entry_pricing": {
        "price_side": "other",
        "use_order_book": True,
        "order_book_top": 1,
        "price_last_balance": 0.0,
        "check_depth_of_market": {
            "enabled": False,
            "bids_to_ask_delta": 1
        }
    },
    "exit_pricing": {
        "price_side": "other",
        "use_order_book": True,
        "order_book_top": 1
    },
    "exchange": {
        "name": "binance",
        "key": os.getenv("BINANCE_API_KEY", ""),
        "secret": os.getenv("BINANCE_SECRET_KEY", ""),
        "ccxt_config": {
            "options": {
                "defaultType": "spot"
            },
            "enableRateLimit": True
        },
        "ccxt_async_config": {
            "options": {
                "defaultType": "spot"
            },
            "enableRateLimit": True
        },
        "pair_whitelist": [
            "1INCH/USDT",
            "AAVE/USDT",
            "ADA/USDT",
            "ALGO/USDT",
            "ANKR/USDT",
            "APE/USDT",
            "ARB/USDT",
            "ATOM/USDT",
            "AVAX/USDT",
            "AXS/USDT",
            "BCH/USDT",
            "BNB/USDT",
            "CAKE/USDT",
            "CHZ/USDT",
            "COMP/USDT",
            "CRV/USDT",
            "DIA/USDT",
            "DOGE/USDT",
            "DYDX/USDT",
            "DOT/USDT",
            "EGLD/USDT",
            "ENA/USDT",
            "ENJ/USDT",
            "ENS/USDT",
            "ETH/USDT",
            "ETC/USDT",
            "FET/USDT",
            "FIL/USDT",
            "GALA/USDT",
            "GRT/USDT",
            "HBAR/USDT",
            "HOT/USDT",
            "ICP/USDT",
            "IOTA/USDT",
            "JASMY/USDT",
            "LINK/USDT",
            "LTC/USDT",
            "MASK/USDT",
            "NEAR/USDT",
            "OP/USDT",
            "ONDO/USDT",
            "OM/USDT",
            "QNT/USDT",
            "POL/USDT",
            "RENDER/USDT",
            "RUNE/USDT",
            "SAND/USDT",
            "SNX/USDT",
            "SOL/USDT",
            "SUI/USDT",
            "SUSHI/USDT",
            "THETA/USDT",
            "TRX/USDT",
            "UMA/USDT",
            "UNI/USDT",
            "VET/USDT",
            "XAI/USDT",
            "XLM/USDT",
            "XRP/USDT",
            "YFI/USDT",
            "ZIL/USDT",
            "ZRX/USDT"
        ],
        "pair_blacklist": [

        ]
    },
    "pairlists": [
        {
            "method": "StaticPairList"
        }
    ],
    "telegram": {
        "enabled": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
        "notification_settings": {
            "status": "on",
            "warning": "on",
            "startup": "on",
            "entry": "on",
            "entry_fill": "on",
            "exit": {
                "roi": "on",
                "emergency_exit": "on",
                "force_exit": "on",
                "exit_signal": "on",
                "trailing_stop_loss": "on",
                "stop_loss": "on",
                "stoploss_on_exchange": "off",
                "custom_exit": "on"
            },
            "exit_fill": "on",
            "entry_cancel": "on",
            "exit_cancel": "on",
            "protection_trigger": "off",
            "protection_trigger_global": "on",
            "show_candle": "off"
        },
        "reload": True,
        "balance_dust_level": 0.01
    },
    "api_server": {
        "enabled": True,
        "listen_ip_address": "0.0.0.0",
        "listen_port": int(os.getenv("PORT", "8080")),
        "verbosity": "error",
        "enable_openapi": False,
        "jwt_secret_key": os.getenv("JWT_SECRET_KEY", "default-secret-key"),
        "ws_token": os.getenv("WS_TOKEN", "default-ws-token"),
        "CORS_origins": os.getenv('CORS_ORIGINS', '').split(',') if os.getenv('CORS_ORIGINS') else [],
        "username": os.getenv("USERNAME", "admin"),
        "password": os.getenv("PASSWORD", "admin")
    },
    "bot_name": "freqtrade",
    "initial_state": "running",
    "force_entry_enable": False,
    "db_url": os.getenv("DATABASE_URL", "sqlite:///tradesv3.sqlite"),
    "strategy": "StrongUptrend",
    "strategy_path": "user_data/strategies/",
    "recursive_strategy_search": True
}

# Ensure output directory exists
os.makedirs("/freqtrade/user_data", exist_ok=True)

# Write config
try:
    with open("/freqtrade/user_data/config.json", "w") as f:
        json.dump(config, f, indent=4)
    print("✅ config.json generated.")
except Exception as e:
    print(f"❌ Error generating config: {e}")
    exit(1)
