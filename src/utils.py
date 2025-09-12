import os, yaml, pytz
from datetime import datetime
from dateutil import tz

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def now_london():
    return datetime.now(pytz.timezone("Europe/London"))

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_env_list(key, default=None):
    raw = os.getenv(key, "")
    if not raw and default is not None:
        return default
    return [s.strip() for s in raw.split(",") if s.strip()]
