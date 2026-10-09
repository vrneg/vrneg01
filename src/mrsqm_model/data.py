"""Shared fixed-grid data representation for MrSQM experiments."""

try:
    from minirocket.data import (
        DataBundle,
        TimeSeriesSplit,
        channel_names,
        prepare_data,
        window_to_array,
    )
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.data import (
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
