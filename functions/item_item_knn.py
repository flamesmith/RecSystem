# **Item-Item knn consists of several steps:**
# 1. Filter to only the products purchased by at least N people.
# 2. Compute from the filtered items the item-item similarity (before the cutoff date). Store the results in a matrix.
# 3. Take a user id and select all the purchases of the user before the cutoff date.
# 4. Convert the user purchases to a vector with 1s and 0s to be the same size as similarity matrix.
# 5. Multiply user vector with similarity matrix and order the scores in descending order for top N most similar items. Remove the items the user has already purchased.

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class ItemItemRecommenderKnn:
    def __init__(self, k=50, min_users=5, shrinkage=10):
        self.k = k
        self.min_users = min_users
        self.shrinkage = shrinkage
        self.S_knn = None
        self.items = None
        self.item_to_idx = None

    def _filter_items(self, df):
        """Removes items with low interaction counts to improve signal-to-noise ratio."""
        item_counts = df.groupby('asin')['reviewerID'].nunique()
        popular_items = item_counts[item_counts >= self.min_users].index
        return df[df['asin'].isin(popular_items)].copy()

    def fit(self, df, cutoff_time=None):
        """Trains the similarity matrix using items purchased before cutoff_time."""
        # 1. Temporal filtering
        if cutoff_time:
            df = df[df['unixReviewTime'] < cutoff_time].copy()

        # 2. Popularity filtering
        df = self._filter_items(df)

        # 3. Create categorical mappings for sparse matrix indexing
        user_cat = df['reviewerID'].astype('category')
        item_cat = df['asin'].astype('category')
        self.items = item_cat.cat.categories
        self.item_to_idx = {item: i for i, item in enumerate(self.items)}

        # 4. Build Sparse User-Item Matrix (Rows: Users, Cols: Items)
        X = csr_matrix(
            (np.ones(len(df)), (user_cat.cat.codes, item_cat.cat.codes)),
            shape=(len(user_cat.cat.categories), len(self.items))
        )

        # 5. Compute Item-Item Cosine Similarity with Shrinkage
        # Norms are the square root of the diagonal of X.T @ X (item popularity)
        item_norms = np.array(np.sqrt(X.sum(axis=0))).flatten()
        S = X.T @ X  # Dot products (shared users between items)

        # Apply shrinkage to dampen similarity between items with very few users
        rows, cols = S.nonzero()
        S.data = S.data / (item_norms[rows] * item_norms[cols] + self.shrinkage)

        # 6. Prune matrix to Top-K neighbors per item
        self.S_knn = self._top_k_filter(S, self.k)
        return self

    def _top_k_filter(self, S, k):
        """Optimized pruning: keeps only the K strongest similarities for each item."""
        S = S.tocsr()
        new_rows, new_cols, new_data = [], [], []

        for i in range(S.shape[0]):
            start, end = S.indptr[i], S.indptr[i+1]
            row_data = S.data[start:end]
            row_indices = S.indices[start:end]

            if len(row_data) > k:
                # Use argpartition for O(n) performance instead of O(n log n)
                idx = np.argpartition(row_data, -k)[-k:]
                new_rows.extend([i] * k)
                new_cols.extend(row_indices[idx])
                new_data.extend(row_data[idx])
            else:
                new_rows.extend([i] * len(row_data))
                new_cols.extend(row_indices)
                new_data.extend(row_data)

        return csr_matrix((new_data, (new_rows, new_cols)), shape=S.shape)

    def recommend(self, user_history, top_n=10):
        """Generates recommendations by scoring items against user history."""
        if self.S_knn is None:
            raise ValueError("Model is not trained. Call .fit() first.")

        # Map user's item IDs to internal matrix indices
        indices = [self.item_to_idx[i] for i in user_history if i in self.item_to_idx]

        if not indices:
            return [] # Cold-start user or all items were filtered out

        # Create binary user vector and compute scores via matrix-vector multiplication
        x_u = np.zeros(len(self.items), dtype=np.float32)
        x_u[indices] = 1.0
        scores = self.S_knn.T.dot(x_u)

        # Ensure we don't recommend items the user has already interacted with
        scores[indices] = -1.0

        # Sort and return top item IDs
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [self.items[i] for i in top_indices if scores[i] > 0]

    def save(self, path):
        """Serialize the trained model to disk using joblib.

        Persists the pruned similarity matrix S_knn and item index mappings.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a previously saved ItemItemRecommenderKnn from disk."""
        return joblib.load(path)