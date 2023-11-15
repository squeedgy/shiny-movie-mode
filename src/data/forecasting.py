import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

file_path = "../../data/raw/aggregated_results.csv"

df_aggregated = pd.read_csv(file_path)

df_aggregated['movie_year'] = pd.to_datetime(df_aggregated['movie_year'], format='%Y')
df_aggregated.set_index('movie_year', inplace=True)

p, d, q = 5, 1, 0

from statsmodels.tsa.arima.model import ARIMA

train_size = int(len(df_aggregated) * 0.8)
train, test = df_aggregated.iloc[:train_size], df_aggregated.iloc[train_size:]

model_rating = ARIMA(train['mean_movie_rating'], order=(p, d, q))
model_fit_rating = model_rating.fit()

print(model_fit_rating.summary())

forecast_rating = model_fit_rating.forecast(steps=len(test))

model_votes = ARIMA(train['total_user_votes'], order=(p, d, q))
model_fit_votes = model_votes.fit()

print(model_fit_votes.summary())

forecast_votes = model_fit_votes.forecast(steps=len(test))

#plot
plt.figure(figsize=(10, 6))
plt.plot(df_aggregated['mean_movie_rating'], label='Original Time Series (Mean Movie Rating)')
plt.plot(forecast_rating, color='green', label='Forecasted Mean Movie Rating')
plt.plot(test['mean_movie_rating'], color='blue', label='Test Set Mean Movie Rating')
plt.title('ARIMA Model - Forecast vs. Test Set (Mean Movie Rating)')
plt.xlabel('Year')
plt.ylabel('Mean Movie Rating')
plt.legend()
plt.show()

#plot
plt.plot(df_aggregated['total_user_votes'], label='Original Time Series (Total User Votes)')
plt.plot(forecast_votes, color='green', label='Forecasted Total User Votes')
plt.plot(test['total_user_votes'], color='blue', label='Test Set Total User Votes')
plt.title('ARIMA Model - Forecast vs. Test Set (Total User Votes)')
plt.xlabel('Year')
plt.ylabel('Total User Votes')
plt.legend()
plt.show()
