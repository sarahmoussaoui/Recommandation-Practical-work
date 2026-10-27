import os
import argparse

import numpy as np
import pandas as pd

NEIGHBORS = 30 # number of neighbours used for prediction
MIN_COMMON = 3 # min co-rated movies to trust a similarity

class UserBasedRecommender:
    """
    User-based collaborative filtering recommender using Pearson similarity.
    Given a user, it returns the movies they have NOT watched that they are
    most likely to enjoy, based on the ratings of similar users.
    """

    def __init__(self, path_movies, path_ratings):
        self.path_movies = path_movies
        self.path_ratings = path_ratings
        self.k = NEIGHBORS             # number of neighbours used for prediction
        self.min_common = MIN_COMMON    # min co-rated movies to trust a similarity
        self.titles = None      # movieId -> title
        self.user_ids = None    # row index -> userId
        self.movie_ids = None   # column index -> movieId
        self.R = None           # ratings matrix (users x movies), 0 = unrated
        self.M = None           # boolean mask, True where a rating exists
        self.user_mean = None   # mean rating of each user

    def set_params(self, k, min_common):
        self.k = k
        self.min_common = min_common

    def _prep_data(self):
        df_movies = pd.read_csv(
            self.path_movies, usecols=['movieId', 'title'])
        df_ratings = pd.read_csv(
            self.path_ratings, usecols=['userId', 'movieId', 'rating'])
        self.titles = df_movies.set_index('movieId').title

        # user-movie matrix: rows = users, columns = movies (NaN = not rated)
        mat = df_ratings.pivot(index='userId', columns='movieId',
                               values='rating')
        self.user_ids = list(mat.index)
        self.movie_ids = list(mat.columns)
        self.M = ~np.isnan(mat.values)
        self.R = np.nan_to_num(mat.values, nan=0.0)
        self.user_mean = self.R.sum(axis=1) / self.M.sum(axis=1)

    def _pearson_with_all(self, u):
        """
        Pearson correlation between user row u and every other user,
        computed only on the movies both users have rated.
        """
        mask = self.M[u] & self.M                        # co-rated movies
        n = mask.sum(axis=1)
        n_safe = np.maximum(n, 1)

        # means computed on the common movies only
        mu_u = (self.R[u] * mask).sum(axis=1) / n_safe
        mu_v = (self.R * mask).sum(axis=1) / n_safe

        a = (self.R[u] - mu_u[:, None]) * mask
        b = (self.R - mu_v[:, None]) * mask

        num = (a * b).sum(axis=1)
        den = np.sqrt((a ** 2).sum(axis=1)) * np.sqrt((b ** 2).sum(axis=1))
        sim = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

        sim[n < self.min_common] = 0.0                   # not enough overlap
        sim[u] = 0.0                                     # exclude the user
        return sim

    def recommend(self, user_id, n_recommendations=10):
        if self.R is None:
            self._prep_data()
        if user_id not in self.user_ids:
            raise ValueError('Unknown userId: {}'.format(user_id))
        u = self.user_ids.index(user_id)

        sim = self._pearson_with_all(u)

        # keep the k most similar users with a positive correlation
        neighbours = np.argsort(sim)[::-1][:self.k]
        neighbours = neighbours[sim[neighbours] > 0]
        if len(neighbours) == 0:
            print('No similar users found for user', user_id)
            return []

        s = sim[neighbours][:, None]
        rated = self.M[neighbours]
        centered = (self.R[neighbours]
                    - self.user_mean[neighbours][:, None]) * rated

        # weighted average of the neighbours' deviations from their mean
        num = (s * centered).sum(axis=0)
        den = (s * rated).sum(axis=0)
        pred = np.full(len(self.movie_ids), -np.inf)
        ok = den > 0
        pred[ok] = self.user_mean[u] + num[ok] / den[ok]

        pred[self.M[u]] = -np.inf                        # drop watched movies
        top = np.argsort(pred)[::-1][:n_recommendations]
        top = [i for i in top if np.isfinite(pred[i])]

        return [(self.titles.get(self.movie_ids[i], str(self.movie_ids[i])),
                 float(np.clip(pred[i], 0.5, 5.0))) for i in top]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run user-based (Pearson) movie recommender')
    parser.add_argument('--movies', default='ml-latest-small/movies.csv')
    parser.add_argument('--ratings', default='ml-latest-small/ratings.csv')
    parser.add_argument('--user_id', type=int, required=True)
    parser.add_argument('--top_n', type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    recommender = UserBasedRecommender(args.movies, args.ratings)
    recommender.set_params(k=30, min_common=5)
    results = recommender.recommend(args.user_id, args.top_n)
    print('Recommendations for user {}:'.format(args.user_id))
    for i, (title, score) in enumerate(results, 1):
        print('{}: {} (predicted rating {:.2f})'.format(i, title, score))