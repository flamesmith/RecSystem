import math

def calculate_precision_at_k(recommended_items, actual_purchases, k):
    """
    Calculates Precision@K for a single user.

    Args:
        recommended_items (list): A list of recommended items.
        actual_purchases (list): A list of items actually purchased by the user.
        k (int): The number of top recommended items to consider.

    Returns:
        float: Precision@K score.
    """
    top_k_recommended = recommended_items[:k]
    if not top_k_recommended:
        return 0.0

    hits = len(set(top_k_recommended) & set(actual_purchases))
    precision = hits / k
    return precision

def calculate_recall_at_k(recommended_items, actual_purchases, k):
    """
    Calculates Recall@K for a single user.

    Args:
        recommended_items (list): A list of recommended items.
        actual_purchases (list): A list of items actually purchased by the user.
        k (int): The number of top recommended items to consider.

    Returns:
        float: Recall@K score.
    """
    if not actual_purchases:
        return 0.0  # or optionally skip this user when averaging

    top_k_recommended = recommended_items[:k]
    hits = len(set(top_k_recommended) & set(actual_purchases))
    recall = hits / len(actual_purchases)
    return recall

def calculate_hit_at_k(recommended_items, actual_purchases, k):
    """
    Calculates Hit@K for a single user.

    Args:
        recommended_items (list): A list of recommended item ASINs.
        actual_purchases (list): A list of item ASINs actually purchased by the user.
        k (int): The number of top recommended items to consider for evaluation.

    Returns:
        int: 1 if at least one recommended item in the top K is in actual purchases, 0 otherwise.
    """
    # Take only the top k items from the recommended list
    top_k_recommended_items = recommended_items[:k]

    if not top_k_recommended_items or not actual_purchases:
        return 0 # No hit if no recommendations or no actual purchases

    # Convert actual_purchases to a set for efficient lookup
    actual_purchases_set = set(actual_purchases)

    # Check if any of the top k recommended items are in the actual purchases
    for item in top_k_recommended_items:
        if item in actual_purchases_set:
            return 1 # Hit found

    return 0 # No hit

def calculate_dcg_at_k(recommended_items, actual_purchases, k):
    """
    Calculates Discounted Cumulative Gain (DCG@K) for a single user.

    Args:
        recommended_items (list): A list of recommended item ASINs, ordered by relevance.
        actual_purchases (list): A list of item ASINs actually purchased by the user.
        k (int): The number of top recommended items to consider for evaluation.

    Returns:
        float: The DCG@K score.
    """
    top_k_recommended = recommended_items[:k]
    if not top_k_recommended:
        return 0.0

    actual_purchases_set = set(actual_purchases)
    dcg = 0.0
    for i, item in enumerate(top_k_recommended):
        relevance = 1 if item in actual_purchases_set else 0
        # Discount factor: log2(position + 1)
        # Position is 0-indexed, so it's i+1. log2(i+1 + 1) = log2(i+2)
        dcg += relevance / math.log2(i + 2)
    return dcg

def calculate_idcg_at_k(actual_purchases, k):
    """
    Calculates Ideal Discounted Cumulative Gain (IDCG@K) for a single user.
    This represents the maximum possible DCG if all relevant items were ranked at the top.

    Args:
        actual_purchases (list): A list of item ASINs actually purchased by the user.
        k (int): The number of top positions to consider for the ideal ranking.

    Returns:
        float: The IDCG@K score.
    """
    if not actual_purchases:
        return 0.0

    # The ideal scenario is to place all relevant items (up to k) at the top
    # We assume relevance = 1 for all actual purchases
    num_relevant_items = min(len(actual_purchases), k)
    idcg = 0.0
    for i in range(num_relevant_items):
        # For ideal ranking, all relevant items have relevance = 1
        idcg += 1 / math.log2(i + 2)
    return idcg

def calculate_ndcg_at_k(recommended_items, actual_purchases, k):
    """
    Calculates Normalized Discounted Cumulative Gain (NDCG@K) for a single user.

    Args:
        recommended_items (list): A list of recommended item ASINs, ordered by relevance.
        actual_purchases (list): A list of item ASINs actually purchased by the user.
        k (int): The number of top recommended items to consider for evaluation.

    Returns:
        float: The NDCG@K score (value between 0 and 1).
    """
    dcg = calculate_dcg_at_k(recommended_items, actual_purchases, k)
    idcg = calculate_idcg_at_k(actual_purchases, k)

    if idcg == 0:
        return 0.0 # Cannot normalize if IDCG is zero (no actual purchases or k is 0)

    return dcg / idcg