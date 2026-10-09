"""Shared fixed-grid data representation for sparse MultiRocket-HYDRA."""

try:
    from multirocket.data import (
        DataBundle,
        TimeSeriesSplit,
        channel_names,
        prepare_data,
        window_to_array,
    )
except ModuleNotFoundError as error:
    if error.name != "multirocket":
        raise
    from ..multirocket.data import (
        DataBundle,
        TimeSeriesSplit,
        channel_names,
        prepare_data,
        window_to_array,
    )

__all__ = [
    "DataBundle",
    "TimeSeriesSplit",
    "channel_names",
    "prepare_data",
    "window_to_array",
]
