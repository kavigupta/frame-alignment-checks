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


def mean_decrease_probability(actual, predicted, masks, *, k):
    """
    Compute the probability of a randomly selected masked element having a lower prediction than a
    randomly selected element overall, with similar actual values.

    :param actual: The actual values of the elements.
    :param predicted: The predicted values of the elements.
    :param masks: A boolean array indicating which elements are masked.
    :param k: The number of closest elements to consider for each element.
    :returns:
        A single float value representing the mean decrease in probability, for each masked element.
    """
    closest_index_array = k_closest_index_array(actual, k)
    decrease_probability = (predicted[:, None] < predicted[closest_index_array]).mean(1)
    return (decrease_probability * masks).sum(1) / masks.sum(1)
