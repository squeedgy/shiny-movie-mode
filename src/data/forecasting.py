import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Replace this with the actual file path where your aggregated data is stored
file_path = "../../data/raw/aggregated_results.csv"

# Read the aggregated results from the CSV file
df_aggregated = pd.read_csv(file_path)

# Convert 'movie_year' to datetime format and set it as the index
df_aggregated['movie_year'] = pd.to_datetime(df_aggregated['movie_year'], format='%Y')
df_aggregated.set_index('movie_year', inplace=True)

# Choose the order (p, d, q) for ARIMA
p, d, q = 5, 1, 0

# Import ARIMA class
from statsmodels.tsa.arima.model import ARIMA

# Split the data into training and testing sets
train_size = int(len(df_aggregated) * 0.8)  # Use 80% of the data for training
train, test = df_aggregated.iloc[:train_size], df_aggregated.iloc[train_size:]

# ARIMA model
model = ARIMA(train['mean_movie_rating'], order=(p, d, q))
model_fit = model.fit()

# Summary of the model
print(model_fit.summary())

# Forecast using the fitted model
forecast = model_fit.forecast(steps=len(test))

# Plot the original time series, forecasted values, and the actual test set values
plt.figure(figsize=(10, 6))
plt.plot(df_aggregated['mean_movie_rating'], label='Original Time Series')
plt.plot(forecast, color='green', label='Forecasted Values')
plt.plot(test['mean_movie_rating'], color='blue', label='Test Set')
plt.title('ARIMA Model - Forecast vs. Test Set')
plt.xlabel('Year')
plt.ylabel('Mean Movie Rating')
plt.legend()
plt.show()
