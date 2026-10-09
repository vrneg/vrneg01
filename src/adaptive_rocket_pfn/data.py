"""Fixed-grid data preparation shared with the other ROCKET experiments."""

try:
    from minirocket.data import DataBundle, TimeSeriesSplit, prepare_data
except ModuleNotFoundError as error:
    if error.name != "minirocket":
        raise
    from ..minirocket.data import DataBundle, TimeSeriesSplit, prepare_data

__all__ = ["DataBundle", "TimeSeriesSplit", "prepare_data"]
