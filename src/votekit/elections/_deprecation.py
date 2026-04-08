import warnings
from typing import Any


def _handle_deprecated_kwargs(kwargs: dict, renames: dict[str, str]) -> dict[str, Any]:
    """
    Check **kwargs for deprecated parameter names, and emit DeprecationWarnings when necessary.

    Args:
        kwargs (dict): The **kwargs dict from the caller.
        renames (dict[str, str]): Mapping of old_name -> new_name.

    Returns:
        dict[str, Any]: Mapping of new_name -> value for each deprecated kwarg found.

    Raises:
        TypeError: If kwargs contains keys not in renames.
    """
    for old_name, new_name in renames.items():
        if old_name in kwargs:
            warnings.warn(
                f"The '{old_name}' parameter has been renamed to '{new_name}'. "
                f"'{old_name}' will be removed in a future version.",
                DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new_name] = kwargs.pop(old_name)

    return kwargs
