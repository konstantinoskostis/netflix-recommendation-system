"""A set of utility functions"""

import numpy as np


def select_title(df, column='Title'):
    """Select a random title from the input dataframe"""
    titles = df[column].to_list()
    index = np.random.randint(len(titles))
    return titles[index]
