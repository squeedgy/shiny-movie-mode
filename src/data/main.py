from movie_raiting import analyze_movie_rating

def main():
    #call function
    df_movie_rating, result_movie_rating = analyze_movie_rating()

    print("\nMovie Rating Dataset:")
    print(df_movie_rating)

if __name__ == "__main__":
    main()