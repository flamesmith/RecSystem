# Bayesian Personalized Ranking (BPR) using the implicit package
#
# This is a wrapper around the implicit library's BPR implementation.
# It provides the same interface as the custom PyTorch BPRRecommender
# so both can be used interchangeably.
#
# Note: The implicit library has not been updated since 2023.
# This version is intended for fast prototyping and benchmarking.

import numpy as np
from implicit.bpr import BayesianPersonalizedRanking


class BPRImplicitRecommender:
    def __init__(self, factors=64, lr=0.01, reg=0.01, epochs=10):
        """
        Initializes the BPR recommender using the implicit package.

        Args:
            factors: Number of latent dimensions for user/item embeddings.
            lr:      Learning rate for SGD.
            reg:     L2 regularization strength.
            epochs:  Number of training iterations over the data.
        """
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

        # The implicit BPR model instance
        self.model = None

        # These are populated after fit()
        self.user_factors = None
        self.item_factors = None
        self.items = None
        self.item_to_idx = None
        self.users = None
        self.user_to_idx = None

    def fit(self, matrix, item_map, items, user_map, users):
        """
        Trains user and item embeddings using the implicit BPR implementation.

        Args:
            matrix:   Sparse csr_matrix (n_users, n_items) from InteractionMatrixBuilder.
            item_map: Dict mapping item_id -> matrix column index.
            items:    Ordered list/index of item_ids (matches matrix columns).
            user_map: Dict mapping user_id -> matrix row index.
            users:    Ordered list/index of user_ids (matches matrix rows).

        Returns:
            self (for method chaining).
        """
        # Step 1: Store mappings for later use in recommend()
        self.items = items
        self.item_to_idx = item_map
        self.users = users
        self.user_to_idx = user_map

        # Step 2: Initialize the implicit BPR model with hyperparameters
        self.model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.lr,
            regularization=self.reg,
            iterations=self.epochs
        )

        # Step 3: Train the model
        # implicit expects a user-item matrix in CSR format (same as InteractionMatrixBuilder outputs)
        self.model.fit(matrix)

        # Step 4: Extract learned embeddings for use in recommend()
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors

        return self

    def recommend(self, user_id, user_history, top_n=10):
        """
        Generates top-N item recommendations for a user.

        Args:
            user_id:      The user's ID (e.g., reviewerID string).
            user_history: List of item IDs the user has already interacted with.
            top_n:        Number of recommendations to return.

        Returns:
            List of recommended item IDs, ordered by predicted relevance.
        """
        if self.item_factors is None:
            raise ValueError("Model is not trained. Call .fit() first.")

        # Step 1: Get the user's latent factor vector
        if user_id in self.user_to_idx:
            # Known user — use their learned embedding
            user_idx = self.user_to_idx[user_id]
            user_vector = self.user_factors[user_idx]
        else:
            # Unknown user — approximate by averaging item embeddings from their history
            known_items = [self.item_to_idx[i] for i in user_history if i in self.item_to_idx]
            if not known_items:
                return []
            user_vector = self.item_factors[known_items].mean(axis=0)

        # Step 2: Score all items by computing dot product with user vector
        scores = self.item_factors @ user_vector

        # Step 3: Exclude items the user has already interacted with
        exclude_indices = [self.item_to_idx[i] for i in user_history if i in self.item_to_idx]
        scores[exclude_indices] = -np.inf

        # Step 4: Return the top-N highest scoring items
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [self.items[i] for i in top_indices]
