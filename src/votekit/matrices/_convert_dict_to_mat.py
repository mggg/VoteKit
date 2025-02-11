import numpy as np


def _convert_dict_to_matrix(data_dict: dict[str, dict]) -> np.ndarray:
    """
    Convert a nested dictionary to a numpy matrix with float entries.
    Will respect the order of the dictionaries.

    Args:
      data_dict (dict[str, dict[str]]): Top level keys are rows, bottom level keys are columns.
        Values must be convertable to float.

    Returns
      np.ndarray: Matrix representing data in dictionary.

    """

    mat = np.zeros((len(data_dict), len(data_dict)))

    for i, row in enumerate(data_dict.values()):
        for j, val in enumerate(row.values()):
            mat[i, j] = float(val)

    return mat
