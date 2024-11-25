import lap
import numpy as np
import scipy
import torch
from scipy.spatial.distance import cdist
# TODO
# from img_xtend.tracker.utils.iou import iou_batch
# END TODO
from img_xtend.utils import LOGGER

"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}

def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    if not data_is_normalized:
        
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    """Helper function for nearest neighbor distance metric (cosine).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.
    """
    if x[0].shape[0:2] != (1,1):
        print("problem")
    if y[0].shape[0:2] != (1,1):
        print("problem")
    
    x_ = torch.from_numpy(np.asarray(x)).squeeze(0,2)
    y_ = torch.from_numpy(np.asarray(y)).squeeze(0,2)
    distances = _cosine_distance(x_, y_)
    return distances.min(axis=0), distances.argmin(axis=0)
    
class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.
    """
    
    def  __init__(self, metric, matching_threshold, budget=None) -> None:
        if metric == 'euclidean':
            self._metric = _nn_euclidean_distance
        elif metric == 'cosine':
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric: must be either 'euclidean' or 'cosine'")

        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}
        
    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}
    
    def distance(self, features, targets, tracks):
        """Compute distance between features and targets.
        CALLED FROM gated_metric inside _match in tracker.py
        Parameters
        ----------
        features : ndarray  (features from the bboxes in the actual frames)
            An NxM matrix of N features of dimensionality M.
            here we are sending all the embeddings of the bboxes
        targets : List[int] (features from the track features)
            A list of targets to match the given `features` against.
            hee are all the tracks 
        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """
        
        cost_matrix = np.zeros((len(targets), len(features)))
        
        argmin_matrix = np.zeros((len(targets), len(features))) - 1
        
        for i, target in enumerate(targets):
            # logger.debug(f'{len(target.features)=} and {features.shape=}')
            # cost_matrix[i, :] = self._metric(self.samples[target], features)
            cost_matrix[i, :], argmin_matrix[i,:] = self._metric(target.features, features)
        return cost_matrix, argmin_matrix