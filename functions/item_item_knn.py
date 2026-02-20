# **Item-Item knn consists of several steps:**
# 1. Filter to only the products purchased by at least N people.
# 2. Compute from the filtered items the item-item similarity (before the cutoff date). Store the results in a matrix.
# 3. Take a user id and select all the purchases of the user before the cutoff date.
# 4. Convert the user purchases to a vector with 1s and 0s to be the same size as similarity matrix.
# 5. Multiply user vector with similarity matrix and order the scores in descending order for top N most similar items. Remove the items the user has already purchased.

import numpy as np
from scipy.sparse import csr_matrix


class ItemItemRecommenderKnn:
    def __init__(self, k=50, shrinkage=10):
        self.k = k
        self.shrinkage = shrinkage
        self.S_knn = None
        self.items = None
        self.item_to_idx = None

    def fit(self, matrix, item_map, items):
        """
        Trains the similarity matrix from a pre-built sparse user-item matrix.

        Args:
            matrix:   sparse csr_matrix (n_users, n_items) from InteractionMatrixBuilder
            item_map: dict mapping item_id -> matrix column index
            items:    ordered list/index of item_ids (matches matrix columns)
        """
        self.items = items
        self.item_to_idx = item_map

        # Compute Item-Item Cosine Similarity with Shrinkage
        item_norms = np.array(np.sqrt(matrix.sum(axis=0))).flatten()
        S = matrix.T @ matrix

        rows, cols = S.nonzero()
        S.data = S.data / (item_norms[rows] * item_norms[cols] + self.shrinkage)

        self.S_knn = self._top_k_filter(S, self.k)
        return self

    def _top_k_filter(self, S, k):
        """Optimized pruning: keeps only the K strongest similarities for each item."""
        S = S.tocsr()
        new_rows, new_cols, new_data = [], [], []

        for i in range(S.shape[0]):
            start, end = S.indptr[i], S.indptr[i + 1]
            row_data = S.data[start:end]
            row_indices = S.indices[start:end]

            if len(row_data) > k:
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

        indices = [self.item_to_idx[i] for i in user_history if i in self.item_to_idx]

        if not indices:
            return []

        x_u = np.zeros(len(self.items), dtype=np.float32)
        x_u[indices] = 1.0
        scores = self.S_knn.T.dot(x_u)

        scores[indices] = -1.0

        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [self.items[i] for i in top_indices if scores[i] > 0]