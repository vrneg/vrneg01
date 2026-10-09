"""Fixed-grid data representation shared with the ROCKET baselines."""

try:
    from minirocket.data import (
        DataBundle,
        TimeSeriesSplit,
        apply_fixed_grid_masks,
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
        apply_fixed_grid_masks,
        channel_names,
        prepare_data,
        window_to_array,
    )

__all__ = [
    "DataBundle",
    "TimeSeriesSplit",
    "apply_fixed_grid_masks",
    "channel_names",
    "prepare_data",
    "window_to_array",
]
