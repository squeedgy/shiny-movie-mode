import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

df = pd.read_csv("../../data/raw/top_english_movies.csv")

def convert_votes_to_numeric(votes_str):
    multiplier = 1
    if 'K' in votes_str:
        multiplier = 1000
    elif 'M' in votes_str:
        multiplier = 1000000
    return float(re.sub(r'[^0-9.]', '', votes_str)) * multiplier

df['user_votes'] = df['user_votes'].apply(convert_votes_to_numeric)

print(df.head())
