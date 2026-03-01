# src/data/interaction_matrix.py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class InteractionMatrixBuilder:
    def __init__(self, min_users=5, cutoff_time=None):
        self.min_users = min_users
        self.cutoff_time = cutoff_time
        self.user_map = None
        self.item_map = None
        self.items = None
        self.users = None

    def _filter_items(self, df):
        item_counts = df.groupby('asin')['reviewerID'].nunique()
        popular_items = item_counts[item_counts >= self.min_users].index
        return df[df['asin'].isin(popular_items)].copy()

    def build(self, df):
        if self.cutoff_time:
            df = df[df['unixReviewTime'] < self.cutoff_time].copy()

        df = self._filter_items(df)

        user_cat = df['reviewerID'].astype('category')
        item_cat = df['asin'].astype('category')

        self.users = user_cat.cat.categories
        self.items = item_cat.cat.categories
        self.user_map = {u: i for i, u in enumerate(self.users)}
        self.item_map = {it: i for i, it in enumerate(self.items)}

        matrix = csr_matrix(
            (np.ones(len(df)), (user_cat.cat.codes, item_cat.cat.codes)),
            shape=(len(self.users), len(self.items))
        )
        return matrix