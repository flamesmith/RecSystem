import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# PopularityRecommender
# ---------------------------------------------------------------------------

class PopularityRecommender:
    """Recommends globally popular items, trained once on historical data.

    Training pre-ranks all items by interaction count up to cutoff_time.
    Inference is a simple list filter — no raw data access at request time.
    """

    def __init__(self):
        self.ranked_items = None   # list of ASINs sorted by popularity (desc)
        self.cutoff_time = None

    def fit(self, df, cutoff_time):
        """Pre-compute global popularity ranking up to cutoff_time.

        Args:
            df: DataFrame with columns ['reviewerID', 'asin', 'unixReviewTime'].
            cutoff_time: Unix timestamp; only interactions before this are used.

        Returns:
            self
        """
        self.cutoff_time = cutoff_time
        df_train = df[df['unixReviewTime'] < cutoff_time]
        item_counts = df_train.groupby('asin').size().sort_values(ascending=False)
        self.ranked_items = item_counts.index.tolist()
        return self

    def recommend(self, user_history, n=10):
        """Return top-N popular items the user has not already interacted with.

        Args:
            user_history: List of ASINs the user has already interacted with.
            n: Number of recommendations.

        Returns:
            List of up to n ASINs.
        """
        if self.ranked_items is None:
            raise ValueError("Model is not trained. Call .fit() first.")
        seen = set(user_history)
        recs = [item for item in self.ranked_items if item not in seen]
        return recs[:n]

    def save(self, path):
        """Serialize model to disk using joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a previously saved PopularityRecommender from disk."""
        return joblib.load(path)


# ---------------------------------------------------------------------------
# TrendingRecommender
# ---------------------------------------------------------------------------

class TrendingRecommender:
    """Recommends items whose interaction velocity increased recently.

    Training pre-computes trend scores up to cutoff_time.
    Inference is a simple list filter — no raw data access at request time.
    """

    def __init__(self, n_days=7):
        self.n_days = n_days
        self.ranked_items = None   # list of ASINs sorted by trend score (desc)
        self.cutoff_time = None

    def fit(self, df, cutoff_time):
        """Pre-compute trend ranking up to cutoff_time.

        Trend score = recent_count * log(1 + recent_count / (past_count + 1))

        Args:
            df: DataFrame with columns ['reviewerID', 'asin', 'unixReviewTime'].
            cutoff_time: Unix timestamp; only interactions before this are used.

        Returns:
            self
        """
        self.cutoff_time = cutoff_time

        cutoff_ts = pd.to_datetime(cutoff_time, unit='s')
        recent_start = cutoff_ts - pd.Timedelta(days=self.n_days)

        df_hist = df[df['unixReviewTime'] < cutoff_time].copy()
        df_hist['dt'] = pd.to_datetime(df_hist['unixReviewTime'], unit='s')

        recent_counts = df_hist[df_hist['dt'] >= recent_start]['asin'].value_counts()
        past_counts = df_hist[df_hist['dt'] < recent_start]['asin'].value_counts()

        trend_df = pd.DataFrame({'recent': recent_counts, 'past': past_counts}).fillna(0)
        trend_df = trend_df[trend_df['past'] > 1]
        trend_df['score'] = (
            trend_df['recent'] * np.log1p(trend_df['recent'] / (trend_df['past'] + 1))
        )
        self.ranked_items = (
            trend_df.sort_values('score', ascending=False).index.tolist()
        )
        return self

    def recommend(self, user_history, n=10):
        """Return top-N trending items the user has not already interacted with.

        Args:
            user_history: List of ASINs the user has already interacted with.
            n: Number of recommendations.

        Returns:
            List of up to n ASINs.
        """
        if self.ranked_items is None:
            raise ValueError("Model is not trained. Call .fit() first.")
        seen = set(user_history)
        recs = [item for item in self.ranked_items if item not in seen]
        return recs[:n]

    def save(self, path):
        """Serialize model to disk using joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a previously saved TrendingRecommender from disk."""
        return joblib.load(path)


# ---------------------------------------------------------------------------
# CooccurrenceRecommender
# ---------------------------------------------------------------------------

class CooccurrenceRecommender:
    """Recommends items that co-occur with a user's history.

    Training builds the item-item co-occurrence matrix once.
    Inference is a fast sparse matrix-vector product — no raw data access.
    """

    def __init__(self):
        self.C = None            # sparse item-item co-occurrence matrix (csr)
        self.items = None        # ordered item index (pandas Index)
        self.item_to_idx = None  # dict: asin -> matrix column index
        self.cutoff_time = None

    def fit(self, df, cutoff_time):
        """Build item-item co-occurrence matrix up to cutoff_time.

        Args:
            df: DataFrame with columns ['reviewerID', 'asin', 'unixReviewTime'].
            cutoff_time: Unix timestamp; only interactions before this are used.

        Returns:
            self
        """
        self.cutoff_time = cutoff_time

        df_filtered = df.loc[df['unixReviewTime'] < cutoff_time, ['reviewerID', 'asin']]

        user_cat = df_filtered['reviewerID'].astype('category')
        item_cat = df_filtered['asin'].astype('category')

        self.items = item_cat.cat.categories
        self.item_to_idx = {item: idx for idx, item in enumerate(self.items)}

        X = csr_matrix(
            (np.ones(len(df_filtered)), (user_cat.cat.codes.values, item_cat.cat.codes.values)),
            shape=(user_cat.cat.categories.size, self.items.size),
        )
        self.C = (X.T @ X).tocsr()
        return self

    def recommend(self, user_history, n=10):
        """Return top-N co-occurrence-scored items not already in user history.

        Args:
            user_history: List of ASINs the user has already interacted with.
            n: Number of recommendations.

        Returns:
            List of up to n ASINs.
        """
        if self.C is None:
            raise ValueError("Model is not trained. Call .fit() first.")

        user_indices = [
            self.item_to_idx[item]
            for item in user_history
            if item in self.item_to_idx
        ]
        if not user_indices:
            return []

        user_vec = np.zeros(self.C.shape[0])
        user_vec[user_indices] = 1.0

        scores = np.asarray(user_vec @ self.C).flatten()
        scores[user_indices] = -1.0

        top_indices = np.argsort(scores)[-n:][::-1]
        return [self.items[i] for i in top_indices if scores[i] > 0]

    def save(self, path):
        """Serialize model to disk using joblib."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        """Load a previously saved CooccurrenceRecommender from disk."""
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Legacy standalone functions (deprecated)
#
# These thin wrappers preserve backward-compatibility with existing notebooks
# and scripts that call the old function-based API.  They will be removed in
# a future clean-up; prefer the class-based API above.
# ---------------------------------------------------------------------------

def get_topn_popular_items(df, user_id, timestamp, n):
    warnings.warn(
        "get_topn_popular_items is deprecated. Use PopularityRecommender.",
        DeprecationWarning, stacklevel=2,
    )
    model = PopularityRecommender().fit(df, timestamp)
    user_history = df.loc[
        (df['reviewerID'] == user_id) & (df['unixReviewTime'] < timestamp), 'asin'
    ].tolist()
    return model.recommend(user_history, n=n)


def get_items_purchased_after_cutoff(df, user_id, cutoff_date):
    """Returns items interacted with by user_id after cutoff_date."""
    return df.loc[
        (df['reviewerID'] == user_id) & (df['unixReviewTime'] > cutoff_date), 'asin'
    ].unique().tolist()


def get_topn_trending_items(df, user_id, timestamp, n, n_days=7):
    warnings.warn(
        "get_topn_trending_items is deprecated. Use TrendingRecommender.",
        DeprecationWarning, stacklevel=2,
    )
    model = TrendingRecommender(n_days=n_days).fit(df, timestamp)
    user_history = df.loc[
        (df['reviewerID'] == user_id) & (df['unixReviewTime'] < timestamp), 'asin'
    ].tolist()
    return model.recommend(user_history, n=n)


def cooccurrence_recommend_for_user_at_time(df, user_id, cutoff_time, n=5):
    warnings.warn(
        "cooccurrence_recommend_for_user_at_time is deprecated. Use CooccurrenceRecommender.",
        DeprecationWarning, stacklevel=2,
    )
    model = CooccurrenceRecommender().fit(df, cutoff_time)
    user_history = df.loc[
        (df['reviewerID'] == user_id) & (df['unixReviewTime'] < cutoff_time), 'asin'
    ].tolist()
    return model.recommend(user_history, n=n)


def compute_cooccurrence_before_time(df, cutoff_time):
    """Legacy helper. Prefer CooccurrenceRecommender.fit() directly."""
    warnings.warn(
        "compute_cooccurrence_before_time is deprecated. Use CooccurrenceRecommender.",
        DeprecationWarning, stacklevel=2,
    )
    model = CooccurrenceRecommender().fit(df, cutoff_time)
    return model.C, model.items
