import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.sparse import csr_matrix

from contextlib import redirect_stdout

def get_topn_popular_items(df, user_id, timestamp, n):
    """
    Returns the top-N most popular items up to a given timestamp
    that the user has NOT purchased yet.

    Parameters:
    - df: pandas DataFrame with columns ['reviewerID', 'asin', 'unixReviewTime']
    - user_id: the ID of the user (corresponds to 'reviewerID')
    - timestamp: cutoff time (recommend items purchased BEFORE this timestamp)
    - n: number of items to recommend

    Returns:
    - list of top-N item_ids (ASINs)
    """

    # 1️⃣ Filter data up to the given timestamp using 'unixReviewTime'
    df_up_to_t = df[df['unixReviewTime'] < timestamp]

    # 2️⃣ Get items the user has already purchased using 'reviewerID' and 'asin'
    user_purchased = set(df_up_to_t[df_up_to_t['reviewerID'] == user_id]['asin'])

    # 3️⃣ Count popularity of each item (number of purchases) using 'asin'
    item_counts = df_up_to_t.groupby('asin').size().sort_values(ascending=False)

    # 4️⃣ Filter out items the user has already purchased
    top_items = [item for item in item_counts.index if item not in user_purchased]

    # 5️⃣ Return top-N
    return top_items[:n]

def get_items_purchased_after_cutoff(df, user_id, cutoff_date):
    """
    Returns a list of unique items (ASINs) purchased by a given user
    after a specified cutoff date.

    Args:
        user_id (str): The ID of the user.
        cutoff_date (pd.Timestamp): The date after which to consider purchases.
        df (pd.DataFrame): The DataFrame containing 'reviewerID', 'asin', and 'unixReviewTime'.

    Returns:
        list: A list of unique ASINs purchased by the user after the cutoff date.
    """
    # Filter for the specific user and purchases after the cutoff date
    purchases_after_cutoff = df[
        (df['reviewerID'] == user_id) &
        (df['unixReviewTime'] > cutoff_date)
    ]

    # Return unique ASINs from these purchases
    return purchases_after_cutoff['asin'].unique().tolist()

def get_topn_reviewed_items(df, user_id, timestamp, review_sample, n):
    """
    Returns the top-N highest reviewed items with review_sample threshold up to a given timestamp
    that the user has NOT purchased yet.

    Parameters:
    - df: pandas DataFrame with columns ['reviewerID', 'asin', 'unixReviewTime']
    - user_id: the ID of the user (corresponds to 'reviewerID')
    - timestamp: cutoff time (recommend items purchased BEFORE this timestamp)
    - review_sample: cutoff on number of reviews to consider for ranking the reviews
    - n: number of items to recommend

    Returns:
    - list of top-N item_ids (ASINs)
    """

    # 1️⃣ Filter data up to the given timestamp using 'unixReviewTime'
    df_up_to_t = df[df['unixReviewTime'] < timestamp]

    # 2️⃣ Get items the user has already purchased using 'reviewerID' and 'asin'
    user_purchased = set(df_up_to_t[df_up_to_t['reviewerID'] == user_id]['asin'])

    # 3️⃣ Count popularity of each item (number of purchases) using 'asin'
    avg_ratings = df_subset.groupby('asin').agg({'overall' : 'mean', 'unixReviewTime' : 'count'}).reset_index()
    avg_ratings.columns = ['asin', 'avg_rating', 'num_ratings']
    avg_ratings = avg_ratings[avg_ratings['num_ratings'] > review_sample].copy()
    avg_ratings.sort_values(by = 'avg_rating', ascending = False, inplace = True)

    # 4️⃣ Filter out items the user has already purchased
    top_items = [item for item in list(avg_ratings['avg_rating']) if item not in user_purchased]

    # 5️⃣ Return top-N
    return top_items[:n]

## **Create function to recommend items based on co occurrence**

def compute_cooccurrence_before_time(df, cutoff_time):
    """
    Compute item–item co-occurrence matrix using implicit feedback.

    Any interaction (regardless of Review value) is treated as 1.
    Only interactions strictly before cutoff_time are used.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns required:
            - 'reviewerID'
            - 'asin'
            - 'unixReviewTime' (datetime)
            - 'overall' (ignored for implicit modeling)

    cutoff_time : datetime
        Only interactions before this timestamp are used.

    Returns
    -------
    C : scipy.sparse.csr_matrix
        Sparse item–item co-occurrence matrix.

    items : pandas.Index
        Mapping of matrix indices to original item IDs.
    """

    # 1 Filter before cutoff
    mask = df["unixReviewTime"] < cutoff_time
    df_filtered = df.loc[mask, ["reviewerID", "asin"]]

    # 2 Convert to categorical codes
    user_cat = df_filtered["reviewerID"].astype("category")
    item_cat = df_filtered["asin"].astype("category")

    user_codes = user_cat.cat.codes.values
    item_codes = item_cat.cat.codes.values

    n_users = user_cat.cat.categories.size
    n_items = item_cat.cat.categories.size

    # 3 Build sparse user-item matrix
    X = csr_matrix(
        (np.ones(len(user_codes)), (user_codes, item_codes)),
        shape=(n_users, n_items)
    )

    # 4 Compute co-occurrence
    C = X.T @ X

    return C, item_cat.cat.categories

def cooccurrence_recommend_for_user_at_time(df, user_id, cutoff_time, n=5):
    """
    Recommend items for a given user using co-occurrence,
    based only on interactions before cutoff_time.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain:
            - 'reviewerID'
            - 'asin'
            - 'unixReviewTime' (datetime)

    user_id : str or int
        User identifier.

    cutoff_time : datetime
        Only interactions before this timestamp are used.

    n : int
        Number of recommendations to return.

    Returns
    -------
    recommendations : list
        List of recommended item IDs (asin).
    """

    # 1️⃣ Compute co-occurrence matrix before cutoff
    C, items = compute_cooccurrence_before_time(df, cutoff_time)

    # 2️⃣ Get user's history before cutoff
    user_mask = (
        (df["reviewerID"] == user_id) &
        (df["unixReviewTime"] < cutoff_time)
    )

    user_items = df.loc[user_mask, "asin"].unique()

    if len(user_items) == 0:
        return []

    # 3️⃣ Map item IDs to matrix indices
    item_to_index = {item: idx for idx, item in enumerate(items)}

    user_indices = [
        item_to_index[item]
        for item in user_items
        if item in item_to_index
    ]

    if not user_indices:
        return []

    # 4️⃣ Build user interaction vector
    user_vector = np.zeros(C.shape[0])
    user_vector[user_indices] = 1

    # 5️⃣ Compute scores
    scores = user_vector @ C
    scores = np.array(scores).flatten()

    # 6️⃣ Remove already purchased items
    scores[user_indices] = -1

    # 7️⃣ Get top-n items
    top_indices = np.argsort(scores)[-n:][::-1]

    recommendations = [
        items[i]
        for i in top_indices
        if scores[i] > 0
    ]

    return recommendations

