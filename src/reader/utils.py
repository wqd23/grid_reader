import functools
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np

from .types import ArrayDict

from .log import logger


def save_data(data: ArrayDict | tuple, group):
    if isinstance(data, tuple):
        group.attrs["type"] = "tuple"
        for idx, item in enumerate(data):
            element_group = group.create_group(f"element_{idx}")
            save_data(item, element_group)
    elif isinstance(data, dict):
        group.attrs["type"] = "ArrayDict"
        for k, v in data.items():
            group.create_dataset(k, data=v)
    else:
        raise ValueError(f"Unknown data type: {type(data)}")


def save_hdf5(data: ArrayDict, path):
    with h5py.File(path, "w") as f:
        save_data(data, f)


def load_data(group):
    data_type = group.attrs.get("type", None)
    if data_type == "tuple":
        elements = []
        idx = 0
        while f"element_{idx}" in group:
            element_group = group[f"element_{idx}"]
            elements.append(load_data(element_group))
            idx += 1
        return tuple(elements)
    elif data_type == "ArrayDict":
        data = {}
        for key in group.keys():
            data[key] = np.array(group[key][()])
        return data
    else:
        raise ValueError(f"Unknown data type: {data_type}")


def read_hdf5(path=None) -> ArrayDict:
    with h5py.File(path, "r") as f:
        data = load_data(f)
    return data


def disk_cache(cache_dir=".cache"):
    """
    cache func output to cache_dir with hdf5

    Parameters
    ----------
    cache_dir : str
        The directory where the cache files are stored.
        Default is '.cache'.

    Returns
    -------
    A decorator function, decorate a function whhich has a 'filename' argument
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory if not exists
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            # Create hash from args and kwargs
            args_str = str(args) + str(sorted(kwargs.items()))
            hash_key = hashlib.md5(args_str.encode()).hexdigest()

            # Full path for cached file
            cache_file = cache_path / f"{func.__name__}_{hash_key}.h5"
            file_name = (
                args[0]
                if len(args) > 0
                else kwargs.get("filename", "file name not found")
            )
            # Return cached result if exists
            if cache_file.exists() and not kwargs.get("nocache", False):
                result = read_hdf5(cache_file)
                logger.info(f"Cache hit: {cache_file} {file_name}")
            else:
                kwargs.pop("nocache", None)
                # Calculate result and cache it
                result = func(*args, **kwargs)
                logger.info(f"Cache miss: {cache_file} {file_name}")

                save_hdf5(result, cache_file)
            if len(result) == 0:
                logger.warning(f"Empty data: {cache_file}")
                sys.exit(1)
            return result

        return wrapper

    return decorator
