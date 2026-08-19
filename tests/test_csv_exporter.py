import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from app.export.csv_exporter import export_dataframe_to_csv


def test_csv_export_writes_file_without_mutating_dataframe(tmp_path):
    table = pd.DataFrame({"Region": ["NR"], "Value": [100]})
    original = table.copy(deep=True)
    output_path = tmp_path / "clean_table.csv"

    returned_path = export_dataframe_to_csv(table, output_path)

    assert returned_path == output_path
    assert output_path.exists()
    assert "Region,Value" in output_path.read_text(encoding="utf-8-sig")
    assert_frame_equal(table, original, check_exact=True)


def test_csv_export_requires_explicit_overwrite(tmp_path):
    output_path = tmp_path / "clean_table.csv"
    table = pd.DataFrame({"value": [1]})
    export_dataframe_to_csv(table, output_path)

    with pytest.raises(FileExistsError):
        export_dataframe_to_csv(table, output_path)

    export_dataframe_to_csv(table, output_path, overwrite=True)


def test_csv_export_requires_csv_suffix(tmp_path):
    with pytest.raises(ValueError, match=".csv"):
        export_dataframe_to_csv(
            pd.DataFrame({"value": [1]}),
            tmp_path / "table.txt",
        )
