"""Fixed-grid data representation adapted to aeon's HC2 numeric kernels."""

import numpy as np

try:
    from minirocket.data import (
        DataBundle,
        TimeSeriesSplit,
        channel_names,
        prepare_data as _prepare_shared_data,
        window_to_array,
    )
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.data import (
        DataBundle,
        TimeSeriesSplit,
        channel_names,
        prepare_data as _prepare_shared_data,
        window_to_array,
    )


def prepare_data(config) -> DataBundle:
    """Prepare the shared fixed grid as contiguous float64 HC2 input.

    aeon 1.5's random shapelet distance kernel normalizes candidate shapelets to
    float64 and cannot compile when the surrounding collection remains float32.
    The classical HC2 components also use double precision internally, so casting
    once here avoids repeated implicit conversions.
    """

    data = _prepare_shared_data(config)
    for split in (data.train, data.validation, data.test):
        split.values = np.ascontiguousarray(split.values, dtype=np.float64)
    return data


__all__ = [
    "DataBundle",
    "TimeSeriesSplit",
    "channel_names",
    "prepare_data",
    "window_to_array",
]
