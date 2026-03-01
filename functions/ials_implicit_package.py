# Implicit Alternating Least Squares (IALS) using the implicit package
#
# IALS learns user and item embeddings from implicit feedback (clicks, views,
# purchases) by factorizing a weighted Preference matrix P, where the weights
# come from a Confidence matrix C = 1 + alpha * R (R = interaction counts).
#
# The algorithm alternates between:
#   - Fixing item vectors V → solving for each user vector U (closed-form ALS)
#   - Fixing user vectors U → solving for each item vector V (closed-form ALS)
#
# This wrapper provides the same interface as BPRImplicitRecommender so both
# can be used interchangeably in your pipeline.

import joblib
import numpy as np
from implicit.als import AlternatingLeastSquares


class IALSImplicitRecommender:
    def __init__(self, factors=64, regularization=0.01, alpha=40, epochs=15, use_gpu=False):
        """
        Initializes the IALS recommender using the implicit package.

        Args:
            factors:        Number of latent dimensions for user/item embeddings.
                            Higher = more expressive, but slower and more prone to overfitting.
            regularization: L2 regularization strength (λ in the ALS closed-form solution).
                            Prevents embeddings from growing too large.
            alpha:          Confidence scaling parameter (α).
                            Converts raw interaction counts R_ui into confidence values:
                                C_ui = 1 + α * R_ui
                            Higher α = interactions are weighted more heavily vs. non-interactions.
            epochs:         Number of full ALS passes (alternating U and V updates).
            use_gpu:        Whether to use GPU acceleration (requires CUDA + implicit[gpu]).
        """
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.epochs = epochs
        self.use_gpu = use_gpu

        # The implicit ALS model instance — created in fit()
        self.model = None

        # Populated after fit() — used for scoring in recommend()
        self.user_factors = None   # shape: (n_users, factors)
        self.item_factors = None   # shape: (n_items, factors)
        self.items = None          # ordered list of item IDs (maps index → item_id)
        self.item_to_idx = None    # dict: item_id → matrix column index
        self.users = None          # ordered list of user IDs (maps index → user_id)
        self.user_to_idx = None    # dict: user_id → matrix row index

    def fit(self, matrix, item_map, items, user_map, users):
        """
        Trains user and item embeddings using the implicit ALS implementation.

        The implicit library expects the interaction matrix in CSR format and
        internally constructs:
            Preference matrix P: P_ui = 1 if R_ui > 0, else 0
            Confidence matrix C: C_ui = 1 + alpha * R_ui

        It then minimizes the weighted squared loss:
            L = Σ_u Σ_i C_ui * (P_ui - U_u · V_i)² + λ(||U||² + ||V||²)

        Args:
            matrix:   Sparse csr_matrix (n_users, n_items) from InteractionMatrixBuilder.
                      Cell values should be interaction counts (not just 0/1),
                      so that the Confidence matrix C = 1 + alpha * R is meaningful.
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

        # Step 2: Initialize the implicit ALS model with hyperparameters
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,           # scales R_ui into confidence C_ui = 1 + alpha * R_ui
            iterations=self.epochs,
            use_gpu=self.use_gpu,
            calculate_training_loss=False
        )

        # Step 3: Train the model
        # implicit's ALS internally scales the matrix by alpha to build C,
        # so we pass the raw interaction count matrix directly (not pre-scaled).
        self.model.fit(matrix)

        # Step 4: Extract learned embeddings for scoring in recommend()
        self.user_factors = self.model.user_factors  # shape: (n_users, factors)
        self.item_factors = self.model.item_factors  # shape: (n_items, factors)

        return self

    def recommend(self, user_id, user_history, top_n=10):
        """
        Generates top-N item recommendations for a user.

        Scoring: score(u, i) = U_u · V_i  (dot product of user and item vectors)

        Items already in the user's history are excluded from recommendations.

        For unknown users (not seen during training), we approximate their
        user vector by averaging the item vectors of their interaction history.
        This is consistent with the ALS intuition: a user vector is a weighted
        average of the item vectors they interacted with.

        Args:
            user_id:      The user's ID (e.g., reviewerID string).
            user_history: List of item IDs the user has already interacted with.
            top_n:        Number of recommendations to return.

        Returns:
            List of recommended item IDs, ordered by predicted relevance (highest first).
        """
        if self.item_factors is None:
            raise ValueError("Model is not trained. Call .fit() first.")

        # Step 1: Get the user's latent factor vector
        if user_id in self.user_to_idx:
            # Known user — use their learned embedding directly
            user_idx = self.user_to_idx[user_id]
            user_vector = self.user_factors[user_idx]
        else:
            # Unknown user — approximate by averaging item embeddings from history.
            # This mirrors the ALS closed-form update: U_u ≈ mean of interacted V_i's.
            known_items = [self.item_to_idx[i] for i in user_history if i in self.item_to_idx]
            if not known_items:
                return []
            user_vector = self.item_factors[known_items].mean(axis=0)

        # Step 2: Score all items via dot product with the user vector
        # score(u, i) = U_u · V_i
        scores = self.item_factors @ user_vector

        # Step 3: Exclude items the user has already interacted with
        exclude_indices = [self.item_to_idx[i] for i in user_history if i in self.item_to_idx]
        scores[exclude_indices] = -np.inf

        # Step 4: Return top-N highest scoring item IDs
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [self.items[i] for i in top_indices]

    def save(self, path):
        """Serialize the trained model to disk using joblib.

        Saves user/item factors and all index mappings.  The implicit model
        object itself is excluded because it is not needed for inference —
        only the extracted numpy factor matrices are required.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a previously saved IALSImplicitRecommender from disk."""
        return joblib.load(path)