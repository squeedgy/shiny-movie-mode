**Overview**:
This project aims to explore historical data of movies, forecasting future trends in movie ratings, and anomalies.

**Research Question**:
    Can we predict future trends in movie ratings and identify anomalies based on historical data?

**Dataset**
    movie_name
    movie_year
    movie_rating
    user_votes

    #Data Preperation/Cleaning
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("../../data/raw/top_english_movies.csv")

    #Exploratory Data Analysis
    print(df.info())
    print(df.describe())

    # Data Cleaning and Preparation
    df['user_votes'] = df['user_votes'].apply(convert_votes_to_numeric)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.histplot(df['user_votes'], kde=True)
    plt.title('Distribution of User Votes')
    plt.xlabel('User Votes')
    plt.ylabel('Frequency')
    plt.show()

    #Time-Series Analysis and Forecasting
    result = seasonal_decompose(df_aggregated['movie_rating'], model='additive', period=1)
    result.plot()
    plt.show()

    model = ARIMA(df_aggregated['movie_rating'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=len(test))

    #Anomaly Detection
    threshold = 2 * df_aggregated['movie_rating'].std()
    anomalies = df_aggregated[abs(df_aggregated['movie_rating'] - df_aggregated['movie_rating'].mean()) > threshold]

**Results**:
    The project offers insights into the dynamics of the English movie industry.

**Getting Started**:
    git clone https://github.com/squeedgy/shiny-movie-mode.git
    git cd movie-mode-ranking

    pip install -r requirements.txt


