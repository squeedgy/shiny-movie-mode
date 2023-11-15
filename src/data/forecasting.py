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

model = ARIMA(train['mean_movie_rating'], order=(p, d, q))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=len(test))

plt.figure(figsize=(10, 6))
plt.plot(df_aggregated['mean_movie_rating'], label='Original Time Series')
plt.plot(forecast, color='green', label='Forecasted Values')
plt.plot(test['mean_movie_rating'], color='blue', label='Test Set')
plt.title('ARIMA Model - Forecast vs. Test Set')
plt.xlabel('Year')
plt.ylabel('Mean Movie Rating')
plt.legend()
plt.show()