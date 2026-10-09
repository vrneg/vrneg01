"""Torch adapters over the fixed-grid representation shared with ROCKET."""

try:
    from inception_tcn.data import (
        DataBundle,
        FixedGridTorchDataset,
        TimeSeriesSplit,
        channel_names,
        prepare_data,
        window_to_array,
    )
except ModuleNotFoundError as error:
    if error.name != "inception_tcn":
        raise
    from ..inception_tcn.data import (
        DataBundle,
        FixedGridTorchDataset,
        TimeSeriesSplit,
        channel_names,
        prepare_data,
        window_to_array,
    )

__all__ = [
    "DataBundle",
    "FixedGridTorchDataset",
    "TimeSeriesSplit",
    "channel_names",
    "prepare_data",
    "window_to_array",
]
