
import numpy as np
import torch

def get_sorted_distances(features_DB, features_Q=None, k=None, metric="cosine"):
    """
    Computes the similarity or distance of up to two sets of embeddings and returns
    similarities/distances and indices of k nearest neighbours.
    
    Parameters
    ----------
    features_DB : array-like, shape = [n_samples, dimensionality]
        Database feature vectors
    features_Q : array-like, shape = [n_samples, dimensionality]
        Query feature vectors. If this parameter is not given, the database fv´s are used as queries.
    k : int
        k nearest neighbours
    metric : str
        Metric to use for distance calculation. Options are "cosine" or "L2".
    """
    if features_Q is None:
        features_Q = features_DB
    
    if k is None:
        k = len(features_DB)
    
    features_DB_torch = torch.tensor(features_DB, dtype=torch.float32)
    features_Q_torch = torch.tensor(features_Q, dtype=torch.float32)
    
    if metric == "cosine":
        similarities = features_Q_torch @ features_DB_torch.T  # Compute cosine similarities
        indices = torch.argsort(similarities, dim=1, descending=True)[:, :k]  # Get top-k indices
        sorted_distances = torch.gather(similarities, 1, indices)
    
    elif metric == "L2":
        
        dists = torch.cdist(features_Q_torch, features_DB_torch, p=2)  # Compute L2 distances using torch
        indices = torch.argsort(dists, dim=1)[:, :k]  # Get top-k indices
        sorted_distances = torch.gather(dists, 1, indices)
    
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'cosine' or 'L2'.")
    
    return sorted_distances.numpy(), indices.numpy()




def get_average_precision_score(y_true, k=None):
    """
    Average precision at rank k
    Modified to only work with sorted ground truth labels
    From: https://gist.github.com/mblondel/7337391
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Binary ground truth (True if relevant, False if irrelevant), sorted by the distances. 
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    if k is None:
        k = np.inf
    
    n_positive = np.sum(y_true.astype(np.int32) == 1)
    
    if n_positive == 0:
        # early return in cases where no positives are among the ranks
        return 0
    
    y_true = y_true[:min(y_true.shape[0], k)].astype(np.int32)
    
    score = 0
    n_positive_seen = 0
    pos_indices = np.where(y_true == 1)[0]
    
    for i in pos_indices:
        n_positive_seen += 1
        score += n_positive_seen / (i + 1.0)
    
    return score / n_positive


def compute_mean_average_precision(categories_DB, 
                                    features_DB=None, 
                                    features_Q=None, 
                                    categories_Q=None, 
                                    indices=None,
                                    metric="cosine", 
                                    k=None):
    """
    Performs a search for k neirest neighboors with the specified indexing method and computes the mean average precision@k 
    
    Parameters
    ----------
    features_DB : array-like, shape = [n_samples, dimensionality]
        Database feature vectors
    features_Q : array-like, shape = [n_samples, dimensionality]
        Query feature vectors. If this parameter is not given, the database fv´s are used as querries
    categories_DB : array-like, shape = [n_samples_DB]
        Database categories
    categories_Q : array-like, shape = [n_samples_Q]
        Query categories. If this parameter is not given, the database categories are used
    indices: array-lile, shape = [n_samples_Q, n_samples_DB]
        Nearest neighbours indices 
    k : int
        Mean average precision at @k value. If np.inf, this function computes the mean average precision score
    Returns
    -------
    Mean average precision @k : float
    """

    if (indices is None) & (features_DB is None):
        raise ValueError("Either indices or features_DB has to be provided ")
    
    if features_Q is None: features_Q = features_DB
    if categories_Q is None: categories_Q = categories_DB
    
    if (indices is None):
        _, indices = get_sorted_distances(features_DB, features_Q, k=k, metric=metric)
    
    aps = []
    for i in range(0, len(indices)):
        aps.append(get_average_precision_score((categories_DB[indices[i]] == categories_Q[i]), k))
    
    return aps
