
import numpy as np
from enum import Enum
import warnings

def _get_distance_fn():
    """
           Try to import the faiss modul to speed up evaluation.
           If faiss is not installed this will fall backt to significantly slower numpy code
    """

    try:

        import faiss

        def get_sorted_distances(features_DB, features_Q=None, k=np.inf):
            """
            Builds a Faiss index of desired type, indexes the given features as database features and performs a search.
            Faiss returns a matrix of sorted distances and the corresponding indices in the input features list. 
            These values are returned by this function
            
            Parameters
            ----------
            features_DB : array-like, shape = [n_samples_DB, dimensionality]
                Database feature vectors
            features_Q : array-like, shape = [n_samples_Q, dimensionality]
                Query feature vectors. If this parameter is not given, the database fv´s are used as querries
            k : int
                Number of neirest neighbors to search for.
            index_type : faiss_utils.INDEX_TYPES Enum
                Sets the distance metric used for this search
            gpu: boolean
                Enables GPU accelerated distance calculations and search.
                Note: If True, this parameter limits k to k=1024 due to Faiss GPU limitations
            Returns
            -------
            sorted_distances : array-like, shape = [n_features_A, k]
                A sorted distance matrix, where each row i represents distances of the k neirest neighbors to entity at features_DB[i]
            indices: array-like, shape = [n_features_A, k]
                Ranking order of indices sorted by the distances
            """
            if not features_DB.flags.contiguous:
                features_DB = np.ascontiguousarray(features_DB)
            
            if k == np.inf: k = features_DB.shape[0]
                
            if features_Q is None: 
                features_Q = features_DB
            else:
                if not features_Q.flags.contiguous:
                    features_Q = np.ascontiguousarray(features_Q)
                
            # create desired faiss index
            index_flat = faiss.IndexFlat(feature_dim)
            
            # add all feature vectors to as database entities
            
            index_flat.add(features_DB)
            # perform a search where the feature
            sorted_distances, indices = index_flat.search(features_Q, k)
            return sorted_distances, indices


    except:

        def get_sorted_distances(features_DB, features_Q=None, k=None):

            sorted_distances, indices = [], []

            if type(features_Q) == type(None):
                features_Q = features_DB

            if type(k) == type(None):
                k = len(features_DB)
            
            f_db_t = features_DB.T
            for f in features_Q:

                sims = f.dot(f_db_t)
                
                sorted_indx = np.argsort(sims)[::-1]
                
                indices.append(sorted_indx[:k])
                sorted_distances.append(sims[sorted_indx[:k]])
            
            sorted_distances, indices = np.array(sorted_distances), np.array(indices)

            return sorted_distances, indices

    return get_sorted_distances






def get_average_precision_score(y_true, k=np.inf):
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
    n_positive = np.sum(y_true.astype(np.int) == 1)
    
    if n_positive == 0:
        # early return in cases where no positives are among the ranks
        return 0
    
    y_true = y_true[:min(y_true.shape[0], k)].astype(np.int)
    
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
                                    k=np.inf):
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
        _, indices = _get_distance_fn()(features_DB, features_Q, k=k)
    
    meanAP = 0
    for i in range(0, len(indices)):
        meanAP += get_average_precision_score((categories_DB[indices[i]] == categories_Q[i]), k)
    return meanAP / len(indices)