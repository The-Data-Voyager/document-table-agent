import json
from pathlib import Path

import pandas as pd
import pytest

from app.cli import main


@pytest.fixture
def sample_pdf():
    return next(
        path
        for path in Path("sample_documents").glob("*.pdf")
        if path.name.startswith("Weekly ")
    )


@pytest.fixture
def outbreak_pdf():
    return Path("sample_documents/75788759701752062509.pdf")


def test_profiles_command_lists_builtin_profile(capsys):
    status = main(["profiles"])

    captured = capsys.readouterr()
    assert status == 0
    assert "grid_india_weekly_report" in captured.out
    assert "idsp_weekly_outbreak_report" in captured.out
    assert "energy_consumption" in captured.out
    assert captured.err == ""


def test_process_command_exports_all_configured_csvs(
    sample_pdf,
    tmp_path,
    capsys,
):
    status = main(
        [
            "process",
            str(sample_pdf),
            "--output-dir",
            str(tmp_path),
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "Profile: grid_india_weekly_report" in captured.out
    assert "raw 42x9, clean 40x9" in captured.out
    assert "raw 42x16, clean 39x16" in captured.out
    assert {path.name for path in tmp_path.glob("*.csv")} == {
        "clean_energy_consumption.csv",
        "clean_maximum_demand.csv",
        "energy_consumption_long.csv",
        "maximum_demand_long.csv",
    }
    assert captured.err == ""


def test_tables_command_prints_analysis_schemas(sample_pdf, capsys):
    status = main(["tables", str(sample_pdf)])

    captured = capsys.readouterr()
    assert status == 0
    assert "energy_consumption: 280 rows" in captured.out
    assert "maximum_demand: 273 rows" in captured.out
    assert "Date: datetime64[" in captured.out
    assert captured.err == ""


def test_query_command_reads_request_file_and_exports_json_answer(
    sample_pdf,
    tmp_path,
    capsys,
):
    answer_path = tmp_path / "top_energy.csv"
    status = main(
        [
            "query",
            str(sample_pdf),
            "--request-file",
            "examples/query_top_energy.json",
            "--format",
            "json",
            "--show-evidence",
            "--evidence-limit",
            "3",
            "--answer-output",
            str(answer_path),
        ]
    )

    captured = capsys.readouterr()
    response = json.loads(captured.out)
    exported = pd.read_csv(answer_path)
    assert status == 0
    assert response["matched_row_count"] == 273
    assert response["returned_row_count"] == 5
    assert response["evidence_rows_shown"] == 3
    assert response["answer"][0] == {
        "Region": "WR",
        "State": "Maharashtra",
        "Weekly_Energy_MU": 4279.9,
    }
    assert Path(response["answer_csv"]) == answer_path.resolve()
    assert exported.shape == (5, 3)
    assert captured.err == ""


def test_query_command_reports_invalid_request_without_traceback(
    sample_pdf,
    capsys,
):
    status = main(
        [
            "query",
            str(sample_pdf),
            "--request",
            '{"table": "energy_consumption", "unsafe": true}',
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert "unsupported fields" in captured.err
    assert "Traceback" not in captured.err


def test_ask_command_translates_question_and_returns_json(sample_pdf, capsys):
    status = main(
        [
            "ask",
            str(sample_pdf),
            "Which state had the highest energy consumption?",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert status == 0
    assert response["matched_metric"] == "energy consumption"
    assert response["request"]["table"] == "energy_consumption"
    assert response["request"]["limit"] == 1
    assert response["answer"] == [
        {
            "State": "Maharashtra",
            "Total_Energy_Consumption_MU": 4279.9,
        }
    ]
    assert response["matched_row_count"] == 273
    assert captured.err == ""


def test_outbreak_process_command_exports_both_continued_tables(
    outbreak_pdf,
    tmp_path,
    capsys,
):
    status = main(
        [
            "process",
            str(outbreak_pdf),
            "--output-dir",
            str(tmp_path),
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    assert status == 0
    assert "Profile: idsp_weekly_outbreak_report" in captured.out
    assert "Table current_outbreaks: pages 3-14" in captured.out
    assert "Table late_outbreaks: pages 15-16" in captured.out
    assert {path.name for path in tmp_path.glob("*.csv")} == {
        "clean_current_outbreaks.csv",
        "clean_late_outbreaks.csv",
        "current_outbreaks_analysis.csv",
        "late_outbreaks_analysis.csv",
    }


def test_outbreak_ask_command_uses_profile_specific_semantics(
    outbreak_pdf,
    capsys,
):
    status = main(
        [
            "ask",
            str(outbreak_pdf),
            "Which state had the most outbreak cases?",
            "--format",
            "json",
        ]
    )

    captured = capsys.readouterr()
    response = json.loads(captured.out)
    assert status == 0
    assert response["matched_metric"] == "current outbreak cases"
    assert response["matched_row_count"] == 34
    assert response["answer"] == [
        {"State_UT": "Maharashtra", "Total_Cases": 258}
    ]
    assert captured.err == ""
