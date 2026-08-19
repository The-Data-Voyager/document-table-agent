"""CSV export helpers for transformed tables."""

from pathlib import Path

import pandas as pd


def export_dataframe_to_csv(
    df: pd.DataFrame,
    output_path: str | Path,
    *,
    index: bool = False,
    encoding: str = "utf-8-sig",
    overwrite: bool = False,
) -> Path:
    """Export ``df`` to CSV without modifying it and return the output path."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    path = Path(output_path)
    if path.suffix.lower() != ".csv":
        raise ValueError("output_path must use the .csv extension.")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {path}. Set overwrite=True to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, encoding=encoding)
    return path
