"""The recommender module.

This module is responsible for running the full pipeline for netflix movies
recommendations given a dataset which contains descriptions of the movies.
"""
import pandas as pd
from .extractor import Extractor
from .preprocessor import Preprocessor


class Recommender:
    """Recommender class is responsible for fitting and retrieving data.

    Attributes:
    csv_location: A string representing the location of the csv to load
    metric: The comparsion metric, `cosine` or `euclidean`
    """
    def __init__(self, csv_location, metric='cosine'):
        self.df = pd.read_csv(csv_location)
        self.metric = metric

    def fit(self):
        self.preprocess()
        self.featurize()

    def preprocess(self):
        self.preprocessor = Preprocessor(self.df)
        self.preprocessor.preprocess()

    def featurize(self):
        self.extractor = Extractor(self.preprocessor.updated_df, self.metric)
        self.extractor.extract()

    def recommend(self, movie_title, top_k=10):
        movie_idx = self.df[self.df['Title'] == movie_title].index[0]
        # this is a Series object
        comparison_row = self.extractor.comparison_df.iloc[movie_idx]
        comparison_row = comparison_row.sort_values(ascending=False)[
            1:(top_k + 1)]
        similar_movies_indices = comparison_row.index.to_list()
        movies_df = self.df.iloc[similar_movies_indices]
        return movies_df
