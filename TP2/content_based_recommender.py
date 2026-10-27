import argparse

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Content-based recommender using movie genres and cosine similarity.
    A user profile is built from the genres of the movies they rated,
    then compared to every movie they have not watched.
    """

    def __init__(self, path_movies, path_ratings):
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.center = True      # weight genres by (rating - user mean)
        self.df_movies = None
        self.df_ratings = None
        self.genre_mat = None   # movies x genres (0/1), indexed by movieId

    def set_params(self, center):
        self.center = center

    def _prep_data(self):
        self.df_movies = pd.read_csv(
            self.path_movies, usecols=['movieId', 'title', 'genres'])
        self.df_ratings = pd.read_csv(
            self.path_ratings, usecols=['userId', 'movieId', 'rating'])

        # one-hot encode genres: "Action|Sci-Fi" -> Action=1, Sci-Fi=1
        genres = self.df_movies.set_index('movieId').genres
        self.genre_mat = genres.str.get_dummies(sep='|')
        self.genre_mat = self.genre_mat.drop(
            columns=['(no genres listed)'], errors='ignore')

    def _user_profile(self, user_id):
        """
        Weighted sum of the genre vectors of the movies the user rated.
        Returns (profile vector, set of watched movieIds).
        """
        rated = self.df_ratings[self.df_ratings.userId == user_id]
        if rated.empty:
            raise ValueError('Unknown userId: {}'.format(user_id))

        weights = rated.rating.values.astype(float)
        if self.center:
            centered = weights - weights.mean()
            # if every rating is identical, centering gives a null profile
            if np.any(centered != 0):
                weights = centered

        movies = rated.movieId.values
        keep = np.isin(movies, self.genre_mat.index)
        vectors = self.genre_mat.loc[movies[keep]].values
        profile = (weights[keep][:, None] * vectors).sum(axis=0)
        return profile, set(movies)

    def recommend(self, user_id, n_recommendations=10):
        if self.genre_mat is None:
            self._prep_data()

        profile, watched = self._user_profile(user_id)
        if not profile.any():
            print('Empty profile for user', user_id)
            return []

        # candidates = movies the user has not watched
        candidates = self.genre_mat[~self.genre_mat.index.isin(watched)]
        sims = cosine_similarity(profile.reshape(1, -1),
                                 candidates.values).ravel()

        top = np.argsort(sims)[::-1][:n_recommendations]
        titles = self.df_movies.set_index('movieId').title
        return [(titles[candidates.index[i]], float(sims[i])) for i in top]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run content-based (genre, cosine) movie recommender')
    parser.add_argument('--movies', default='ml-latest-small/movies.csv')
    parser.add_argument('--ratings', default='ml-latest-small/ratings.csv')
    parser.add_argument('--user_id', type=int, required=True)
    parser.add_argument('--top_n', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    recommender = ContentBasedRecommender(args.movies, args.ratings)
    recommender.set_params(center=True)
    results = recommender.recommend(args.user_id, args.top_n)
    print('Recommendations for user {}:'.format(args.user_id))
    for i, (title, score) in enumerate(results, 1):
        print('{}: {} (cosine similarity {:.3f})'.format(i, title, score))