import numpy as np
from permacache import permacache, stable_hash


@permacache(
    "modular_splicing/frame_alignment/utils/k_closest_index_array_5",
    key_function=dict(arr=stable_hash),
)
def k_closest_index_array(arr, k):
    """
    Get the k closest indices to each element in the array.

    :param arr: The array to get the closest indices for.
    :param k: The number of closest indices to get.

    :returns:
        closest_idxs: The indices of the k closest elements to each element in the array.
            Does not include the element itself or any duplicates.
    """
    closest_idxs = []
    for idx in range(arr.shape[0]):
        distances = np.abs(arr - arr[idx])
        distances[distances == 0] = np.inf
        closest = np.argpartition(distances, k)[:k]
        closest_idxs.append(closest)
    return np.array(closest_idxs)


def mean_quantile(actual, predicted, masks, *, k):
    """
    Compute the mean of the quantiles of the predicted values at the closest indices.
    """
    closest_index_array = k_closest_index_array(actual, k)
    quantile_by_position = (predicted[:, None] > predicted[closest_index_array]).mean(1)
    return (quantile_by_position * masks).sum(1) / masks.sum(1)
