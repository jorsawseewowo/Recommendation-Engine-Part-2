import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

ratings_df = pd.read_csv('ratings.csv')
ratings_df.head()

movies_df = pd.read_csv('movies.csv')
movies_df.head()

movies_df['year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False)
movies_df.head()

movies_df['title'] = movies_df['title'].str.replace(r'\((\d{4})\)', '')
movies_df.head()

movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

movies_df['genres'] = movies_df.genres.str.split('|')
movies_df.head()

movies_copy = movies_df.copy()

for index, row in movies_df.iterrows():
    for genre in row['genres']:
        movies_copy.at[index, genre] = 1
movies_copy.head()

movies_copy = movies_copy.fillna(0)
movies_copy.head()

ratings_df.head()

ratings_df = ratings_df.drop(['timestamp'], axis=1)
ratings_df.head()

user_input = [
    {'title' : 'Jumanji', 'rating' : 8.5},
    {'title' : 'Toy Story', 'rating' : 4.5},
    {'title' : 'Grand Slam', 'rating' : 5.6},
    {'title' : 'Zero', 'rating' : 7}
]

movies_input = pd.DataFrame(user_input)
movies_input

input_id = movies_df[movies_df['title'].isin(movies_input['title'].tolist())]

movies_input = pd.merge(input_id, movies_input)
movies_input

movies_input = movies_input.drop(['genres','year'], axis=1)
movies_input

movies_user = movies_copy[movies_copy['movieId'].isin(movies_input['movieId'].tolist())]
movies_user

movies_user = movies_user.reset_index(drop=True)
movies_user

UserGenreTable = movies_user.drop(['movieId', 'title', 'genres', 'year'], axis=1)
UserGenreTable

UserProfile = UserGenreTable.transpose().dot(movies_input['rating'])
UserProfile

GenreTable = movies_copy.set_index(movies_copy['movieId'])
GenreTable

GenreTable = GenreTable.drop(['movieId', 'title', 'genres', 'year'], axis=1)
GenreTable.head()

Recommendation_df = ((GenreTable*UserProfile).sum(axis=1))/UserProfile.sum()
Recommendation_df.head()

Recommendation_df = Recommendation_df.sort_values(ascending=False)
Recommendation_df.head()

RecommendationTable = movies_df.loc[movies_df['movieId'].isin(Recommendation_df.head(20).keys())]
RecommendationTable