import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.transformation.electricity_analysis import (
    energy_consumption_to_long,
    maximum_demand_to_long,
    reconcile_energy_consumption_totals,
)


def test_energy_consumption_to_long_parses_dates_and_preserves_total():
    wide = pd.DataFrame(
        {
            "Region": ["NR", "ALL INDIA"],
            "States": ["Punjab", None],
            "30-03-2026": [10.0, 10.0],
            "31-03-2026": [12.0, 12.0],
        }
    )
    original = wide.copy(deep=True)

    long_table = energy_consumption_to_long(wide)

    assert long_table.shape == (4, 4)
    assert long_table.columns.tolist() == [
        "Region",
        "State",
        "Date",
        "Energy_Consumption_MU",
    ]
    assert pd.api.types.is_datetime64_any_dtype(long_table["Date"])
    assert long_table.loc[
        long_table["Region"] == "ALL INDIA", "State"
    ].tolist() == ["ALL INDIA", "ALL INDIA"]
    assert_frame_equal(wide, original, check_exact=True)


def test_maximum_demand_to_long_pairs_each_date_with_both_metrics():
    wide = pd.DataFrame(
        [
            ["NR", "Punjab", 100, 0, 110, 1],
            ["WR", "Gujarat", 90, 2, 95, 0],
        ]
    )
    wide.columns = pd.MultiIndex.from_tuples(
        [
            ("Region", ""),
            ("Date", "States"),
            ("30-03-2026", "Max. Demand Met during the day"),
            ("30-03-2026", "Peak hr Shortage"),
            ("31-03-2026", "Max. Demand Met during the day"),
            ("31-03-2026", "Peak hr Shortage"),
        ]
    )
    original = wide.copy(deep=True)

    long_table = maximum_demand_to_long(wide)

    assert long_table.shape == (4, 5)
    assert long_table.columns.tolist() == [
        "Region",
        "State",
        "Date",
        "Maximum_Demand_MW",
        "Peak_Shortage_MW",
    ]
    assert pd.api.types.is_datetime64_any_dtype(long_table["Date"])
    assert long_table["Maximum_Demand_MW"].tolist() == [100, 90, 110, 95]
    assert long_table["Peak_Shortage_MW"].tolist() == [0, 2, 1, 0]
    assert_frame_equal(wide, original, check_exact=True)


def test_maximum_demand_to_long_rejects_incomplete_metric_pair():
    wide = pd.DataFrame([["NR", "Punjab", 100]])
    wide.columns = pd.MultiIndex.from_tuples(
        [
            ("Region", ""),
            ("Date", "States"),
            ("30-03-2026", "Max. Demand Met during the day"),
        ]
    )

    with pytest.raises(ValueError, match="missing metrics"):
        maximum_demand_to_long(wide)


def test_energy_total_reconciliation_reports_difference_and_tolerance():
    long_table = pd.DataFrame(
        {
            "Region": ["NR", "WR", "ALL INDIA"],
            "State": ["Punjab", "Gujarat", "ALL INDIA"],
            "Date": pd.to_datetime(["2026-03-30"] * 3),
            "Energy_Consumption_MU": [10.0, 20.0, 30.4],
        }
    )

    reconciliation = reconcile_energy_consumption_totals(
        long_table,
        tolerance=0.5,
    )

    assert reconciliation.loc[0, "Component_Sum_MU"] == 30.0
    assert reconciliation.loc[0, "Reported_ALL_INDIA_MU"] == 30.4
    assert reconciliation.loc[0, "Difference_MU"] == pytest.approx(0.4)
    assert bool(reconciliation.loc[0, "Within_Tolerance"]) is True
