"""The extractor module

This module performs tf-idf vectorization given a corpus.
It also computes a metric matrix (cosine or eulidean) in order to
compare movies.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


class Extractor:
    """Extracts features using tf-idf.

    Additionally it will compute a comparison matrix and perist it, so
    that it can be later used for TOP-K neighbors.

    Attributes:
        df: The input dataframe
        column_name: The string name of the output column
        metric: The name of the metric for comparison matrix. Either
          cosine or euclidean
    """

    def __init__(self, df, column_name='processed_description',
                 metric='cosine'):
        self.sentences = df[column_name]
        self.metric = metric
        # built defines if the extraction process
        # is finished or not.
        self.built = False
        # initialize tf-idf vectorizer
        self.vectorizer = TfidfVectorizer()

    def extract(self):
        """Runs the extraction process"""
        if self.built is True:
            return

        # vectorize sentences
        self.vectorize()
        # compute and cache comparison matrix
        self.compute_comparison_matrix()

        self.built = True

    def vectorize(self):
        """Vectorizes sentenses"""
        self.tfidf_matrix = self.vectorizer.fit_transform(self.sentences)

    def compute_comparison_matrix(self):
        """Compute similarity/distance matrix

        Supports cosine-similarity and euclidean-distance.
        """
        comparison_matrix = []

        if self.metric == 'cosine':
            comparison_matrix = cosine_similarity(self.tfidf_matrix)
        elif self.metric == 'euclidean':
            comparison_matrix = euclidean_distances(self.tfidf_matrix)
        self.comparison_df = pd.DataFrame(comparison_matrix)
