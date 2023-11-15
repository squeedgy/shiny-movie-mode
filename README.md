--This project aims to explore historical data of movies, forecasting future trends in movie ratings, and detecting anomalies. In this project, we delve into patterns that have emerged over the years in movie ratings and votes, predicting future trends, and identifying significant deviations.

--Research Question
    Can we predict future trends in movie ratings and identify anomalies based on historical data?

--Dataset
    movie_name
    movie_year
    movie_rating
    user_votes

    --Data Preperation/Cleaning
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv("../../data/raw/top_english_movies.csv")

    --Exploratory Data Analysis

    --Time-Series Analysis and Forecasting

    --Anomaly Detection

--Results:
    The project culminates in a model that can forecast movie rating trends and identify anomalous years, offering insights into the dynamics of the English movie industry.

Getting Started:
    git clone https://github.com/squeedgy/shiny-movie-mode.git
    git cd movie-mode-ranking

    pip install -r requirements.txt


