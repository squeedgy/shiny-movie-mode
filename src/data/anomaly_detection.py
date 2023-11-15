import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

#Load the data
file_path = "../../data/raw/aggregated_results.csv"

df_aggregated = pd.read_csv(file_path)

df_aggregated['movie_year'] = pd.to_datetime(df_aggregated['movie_year'], format='%Y')
df_aggregated.set_index('movie_year', inplace=True)

feature_for_detection = 'movie_rating'

#train model
model = IsolationForest(contamination=0.05)
model.fit(df_aggregated[['mean_movie_rating']])

# Predict anomalies (1 for normal, -1 for anomaly)
predictions = model.predict(df_aggregated[['mean_movie_rating']])

# Add a column for anomaly predictions to the dataframe
df_aggregated['anomaly'] = predictions

#visualize
plt.figure(figsize=(10, 6))
plt.plot(df_aggregated['mean_movie_rating'], label='Original Time Series')
plt.scatter(df_aggregated.index[df_aggregated['anomaly'] == -1], df_aggregated['mean_movie_rating'][df_aggregated['anomaly'] == -1], color='red', label='Anomalies')
plt.title('Isolation Forest Anomaly Detection')
plt.xlabel('Year')
plt.ylabel('mean_movie_rating')  # Update this label if needed
plt.legend()
plt.show()

#analyze
anomalies = df_aggregated[df_aggregated['anomaly'] == -1]
print(anomalies)
