"""Analysis-ready postprocessors for the built-in electricity profile."""

from __future__ import annotations

from typing import Any

import pandas as pd


_DATE_FORMAT = "%d-%m-%Y"
_MAXIMUM_DEMAND_LABEL = "Max. Demand Met during the day"
_PEAK_SHORTAGE_LABEL = "Peak hr Shortage"


def _ensure_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")


def _parsed_date(value: Any) -> pd.Timestamp:
    try:
        return pd.to_datetime(value, format=_DATE_FORMAT, errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Expected a date column formatted as DD-MM-YYYY, got {value!r}."
        ) from error


def energy_consumption_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a cleaned Energy wide table to analysis-ready long format."""

    _ensure_dataframe(df)
    if isinstance(df.columns, pd.MultiIndex) or df.shape[1] < 3:
        raise ValueError(
            "Energy input must have flat Region/State columns and at least "
            "one dated measure column."
        )

    original = df.copy(deep=True)
    region_column = df.columns[0]
    state_column = df.columns[1]
    date_columns = list(df.columns[2:])
    date_lookup = {
        column: _parsed_date(column) for column in date_columns
    }

    long_table = df.melt(
        id_vars=[region_column, state_column],
        value_vars=date_columns,
        var_name="Date",
        value_name="Energy_Consumption_MU",
    ).rename(
        columns={region_column: "Region", state_column: "State"}
    )
    long_table["Date"] = long_table["Date"].map(date_lookup)
    long_table["Energy_Consumption_MU"] = pd.to_numeric(
        long_table["Energy_Consumption_MU"],
        errors="raise",
    )

    missing_state = long_table["State"].isna() | long_table["State"].map(
        lambda value: isinstance(value, str) and not value.strip()
    )
    total_rows = (
        long_table["Region"].astype(str).str.strip().str.upper()
        == "ALL INDIA"
    )
    long_table.loc[missing_state & total_rows, "State"] = "ALL INDIA"
    if bool(long_table["State"].isna().any()):
        raise ValueError("Energy long output contains an unresolved State.")

    result = long_table[
        ["Region", "State", "Date", "Energy_Consumption_MU"]
    ].reset_index(drop=True)
    pd.testing.assert_frame_equal(df, original, check_exact=True)
    return result


def maximum_demand_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a cleaned Maximum Demand MultiIndex table to long format."""

    _ensure_dataframe(df)
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
        raise ValueError(
            "Maximum Demand input must have a two-level column header."
        )
    if df.shape[1] < 3:
        raise ValueError(
            "Maximum Demand input must contain identity and measure columns."
        )

    original = df.copy(deep=True)
    measures_by_date: dict[pd.Timestamp, dict[str, int]] = {}
    for column_position in range(2, df.shape[1]):
        date_label, metric_label = df.columns[column_position]
        parsed_date = _parsed_date(date_label)
        if metric_label not in {
            _MAXIMUM_DEMAND_LABEL,
            _PEAK_SHORTAGE_LABEL,
        }:
            raise ValueError(
                f"Unexpected Maximum Demand metric label: {metric_label!r}."
            )
        date_metrics = measures_by_date.setdefault(parsed_date, {})
        if metric_label in date_metrics:
            raise ValueError(
                f"Duplicate metric {metric_label!r} for {date_label!r}."
            )
        date_metrics[metric_label] = column_position

    required_metrics = {
        _MAXIMUM_DEMAND_LABEL,
        _PEAK_SHORTAGE_LABEL,
    }
    frames = []
    for parsed_date, date_metrics in measures_by_date.items():
        if set(date_metrics) != required_metrics:
            missing = sorted(required_metrics - set(date_metrics))
            raise ValueError(
                f"Date {parsed_date.date()} is missing metrics: {missing!r}."
            )
        frame = pd.DataFrame(
            {
                "Region": df.iloc[:, 0].to_numpy(copy=True),
                "State": df.iloc[:, 1].to_numpy(copy=True),
                "Date": parsed_date,
                "Maximum_Demand_MW": pd.to_numeric(
                    df.iloc[:, date_metrics[_MAXIMUM_DEMAND_LABEL]],
                    errors="raise",
                ).to_numpy(copy=True),
                "Peak_Shortage_MW": pd.to_numeric(
                    df.iloc[:, date_metrics[_PEAK_SHORTAGE_LABEL]],
                    errors="raise",
                ).to_numpy(copy=True),
            }
        )
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)[
        [
            "Region",
            "State",
            "Date",
            "Maximum_Demand_MW",
            "Peak_Shortage_MW",
        ]
    ]
    pd.testing.assert_frame_equal(df, original, check_exact=True)
    return result


def reconcile_energy_consumption_totals(
    long_df: pd.DataFrame,
    *,
    tolerance: float = 1.0,
) -> pd.DataFrame:
    """Compare reported ALL INDIA values with summed component rows by date."""

    _ensure_dataframe(long_df)
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative.")
    required_columns = {
        "Region",
        "Date",
        "Energy_Consumption_MU",
    }
    missing_columns = required_columns - set(long_df.columns)
    if missing_columns:
        raise ValueError(
            f"Energy long table is missing columns: {sorted(missing_columns)!r}."
        )

    total_mask = (
        long_df["Region"].astype(str).str.strip().str.upper()
        == "ALL INDIA"
    )
    reported_counts = long_df.loc[total_mask].groupby("Date").size()
    if reported_counts.empty or bool((reported_counts != 1).any()):
        raise ValueError(
            "Expected exactly one ALL INDIA row for every reported total date."
        )

    component_sum = (
        long_df.loc[~total_mask]
        .groupby("Date")["Energy_Consumption_MU"]
        .sum()
        .rename("Component_Sum_MU")
    )
    reported_total = (
        long_df.loc[total_mask]
        .set_index("Date")["Energy_Consumption_MU"]
        .rename("Reported_ALL_INDIA_MU")
    )
    reconciliation = pd.concat(
        [component_sum, reported_total],
        axis=1,
        join="outer",
    ).reset_index()
    reconciliation["Difference_MU"] = (
        reconciliation["Reported_ALL_INDIA_MU"]
        - reconciliation["Component_Sum_MU"]
    )
    reconciliation["Within_Tolerance"] = (
        reconciliation["Difference_MU"].abs() <= tolerance
    )
    return reconciliation
