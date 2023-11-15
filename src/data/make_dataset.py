import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

df = pd.read_csv("../../data/raw/top_english_movies.csv")

print("Data Types:\n", df.dtypes)

#convert string values with 'K' or 'M' to numeric
def convert_votes_to_numeric(votes_str):
    if isinstance(votes_str, float):
        return int(votes_str)
    multiplier = 1
    if 'K' in votes_str:
        multiplier = 1000
    elif 'M' in votes_str:
        multiplier = 1000000
    return int(float(re.sub(r'[^0-9.]', '', votes_str)) * multiplier)

df['user_votes'] = df['user_votes'].apply(convert_votes_to_numeric)

print(df.head())

df_aggregated = df.resample('Y').mean()

#check for missing values
missing_values = df.isnull().sum()

print("Missing Values:\n", missing_values)

percentage_missing = (missing_values / len(df)) * 100
print("\nPercentage of Missing Values:\n", percentage_missing)

#display the rows that are missing
missing_rows = df[df.isnull().any(axis=1)]
print("Rows with Missing Values:\n", missing_rows)

#group by movie name,
top_movies = df.groupby('movie_name')['user_votes'].sum().sort_values(ascending=False).head(5)

#plot for the top 5 movies based on user votes
plt.figure(figsize=(12, 6))
sns.barplot(x=top_movies.values, y=top_movies.index, palette="viridis")
plt.title('Top 5 Movies Based on User Votes')
plt.xlabel('Total User Votes')
plt.ylabel('Movie Name')
plt.tight_layout()  #adjust layout so text fits
plt.show()

#KDE plot for user vote distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['user_votes'], kde=True, bins=30, color='skyblue', edgecolor='black')
plt.title('Distribution of User Votes')
plt.xlabel('User Votes')
plt.ylabel('Frequency')
plt.show()

aggregated_data = df.groupby('movie_year').agg({
    'movie_rating': 'sum',
    'user_votes': 'sum',
}).reset_index()

#sum total number of user votes per year
aggregated_data['total_user_votes'] = df.groupby('movie_year')['user_votes'].sum().values

#add a column and round the values
aggregated_data['mean_movie_rating'] = df.groupby('movie_year')['movie_rating'].mean().round(2).values

#save only the necessary columns
aggregated_data[['movie_year', 'total_user_votes', 'mean_movie_rating']].to_csv(
    "../../data/raw/aggregated_results.csv", index=False
)

print(aggregated_data)