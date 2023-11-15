import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt

def analyze_movie_rating(file_path="../../data/raw/aggregated_results.csv"):

    df_aggregated = pd.read_csv(file_path)

    #convert movie_year and datetime and set as index
    df_aggregated['movie_year'] = pd.to_datetime(df_aggregated['movie_year'], format='%Y')
    df_aggregated.set_index('movie_year', inplace=True)

    result = seasonal_decompose(df_aggregated['mean_movie_rating'], model='additive', period=1)

    #plot
    result.plot()
    plt.show()

    return df_aggregated, result

if __name__ == "__main__":
    analyze_movie_rating()
