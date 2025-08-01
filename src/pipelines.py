import pandas as pd
from pathlib import Path
import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def day_month_encoding(id):
    month_int= {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    month = month_int[id["month"]]
    day = id["day"]
    try:
        return datetime.date(2024, month, day).timetuple().tm_yday
    except ValueError:
        return datetime.date(2024, month, min(day, 29)).timetuple().tm_yday
