# Bayesian Personalized Ranking (BPR) Recommender
#
# BPR learns latent factor representations for users and items by optimizing
# a pairwise ranking loss. For each user, it samples a positive item (one the
# user interacted with) and a negative item (one they did not), then pushes the
# positive item's score above the negative item's score.
#
# Reference: Rendle et al., "BPR: Bayesian Personalized Ranking from Implicit Feedback" (2009)

import numpy as np
import torch
import torch.nn as nn



class BPRRecommender:
    def __init__(self, factors=64, lr=0.01, reg=0.01, epochs=10, batch_size=1024):
        """
        Initializes the BPR recommender with tunable hyperparameters.

        Args:
            factors:    Number of latent dimensions for user/item embeddings.
            lr:         Learning rate for the Adam optimizer.
            reg:        L2 regularization strength (weight decay).
            epochs:     Number of full passes over the training data.
            batch_size: Number of triplets per gradient update step.
        """
        self.factors = factors
        self.lr = lr
        self.reg = reg
        self.epochs = epochs
        self.batch_size = batch_size

        # These are populated after fit()
        self.user_factors = None
        self.item_factors = None
        self.items = None
        self.item_to_idx = None
        self.users = None
        self.user_to_idx = None

    def fit(self, matrix, item_map, items, user_map, users):
        """
        Trains user and item embeddings using BPR pairwise ranking loss.

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

        n_users, n_items = matrix.shape

        # Step 2: Build a per-user list of positive item indices for fast sampling
        # For each user, we store the column indices of items they interacted with
        user_positive_items = {}
        csr = matrix.tocsr()
        for u in range(n_users):
            user_positive_items[u] = set(csr.indices[csr.indptr[u]:csr.indptr[u + 1]])

        # Step 3: Select device — use Apple MPS GPU if available, otherwise CPU
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Training on Apple MPS (GPU)")
        else:
            device = torch.device("cpu")
            print("Training on CPU")

        # Step 4: Initialize user and item embedding matrices with small random values
        # Xavier-style initialization scaled by the number of factors
        user_embeddings = nn.Embedding(n_users, self.factors).to(device)
        item_embeddings = nn.Embedding(n_items, self.factors).to(device)
        nn.init.xavier_uniform_(user_embeddings.weight)
        nn.init.xavier_uniform_(item_embeddings.weight)

        # Step 5: Set up the Adam optimizer with L2 regularization (weight decay)
        optimizer = torch.optim.Adam(
            list(user_embeddings.parameters()) + list(item_embeddings.parameters()),
            lr=self.lr,
            weight_decay=self.reg
        )

        # Step 6: Training loop — iterate over epochs
        # Each epoch samples one triplet per observed interaction
        all_interactions = []
        for u in range(n_users):
            for i in user_positive_items[u]:
                all_interactions.append((u, i))
        all_interactions = np.array(all_interactions)

        for epoch in range(self.epochs):
            # Shuffle interactions at the start of each epoch
            np.random.shuffle(all_interactions)
            total_loss = 0.0
            n_batches = 0

            # Process interactions in mini-batches
            for start in range(0, len(all_interactions), self.batch_size):
                batch = all_interactions[start:start + self.batch_size]
                batch_users = batch[:, 0]
                batch_pos_items = batch[:, 1]

                # Step 6a: Sample negative items for each triplet
                # For each (user, positive_item) pair, randomly pick an item
                # the user has NOT interacted with
                batch_neg_items = np.zeros(len(batch), dtype=np.int64)
                for idx in range(len(batch)):
                    u = batch_users[idx]
                    neg = np.random.randint(0, n_items)
                    while neg in user_positive_items[u]:
                        neg = np.random.randint(0, n_items)
                    batch_neg_items[idx] = neg

                # Step 6b: Convert to tensors and move to device (GPU/CPU)
                u_tensor = torch.LongTensor(batch_users).to(device)
                pos_tensor = torch.LongTensor(batch_pos_items).to(device)
                neg_tensor = torch.LongTensor(batch_neg_items).to(device)

                # Step 6c: Look up embeddings for users, positive items, and negative items
                u_emb = user_embeddings(u_tensor)
                pos_emb = item_embeddings(pos_tensor)
                neg_emb = item_embeddings(neg_tensor)

                # Step 6d: Compute scores via dot product
                # Positive score = dot(user, positive_item)
                # Negative score = dot(user, negative_item)
                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)

                # Step 6e: Compute BPR loss
                # BPR loss = -log(sigmoid(pos_score - neg_score))
                # We want the positive item to score higher than the negative item
                loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

                # Step 6f: Backpropagate and update embeddings
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch + 1}/{self.epochs} — BPR loss: {avg_loss:.4f}")

        # Step 7: Extract trained embeddings as numpy arrays for fast inference
        # Move back to CPU before converting to numpy (required when training on GPU)
        self.user_factors = user_embeddings.weight.detach().cpu().numpy()
        self.item_factors = item_embeddings.weight.detach().cpu().numpy()

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
