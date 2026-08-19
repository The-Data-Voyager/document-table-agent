import sys  
CODE = '''"""
app/validation/table_validator.py
==================================
Generic, read-only validation for Pandas DataFrames extracted from PDF tables.

Architecture constraint
-----------------------
This module is the *Validation* layer.  It only inspects and reports
problems; it never modifies, cleans, or transforms the input DataFrame.

All public functions treat the DataFrame as immutable.  A deep-copy guard is
built into the main entry-point so callers can verify that the DataFrame is
returned unchanged.

Typical usage
-------------
>>> from app.validation.table_validator import validate_table
>>> report = validate_table(df)
>>> print(report)
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    re.compile(r"\\b\\d{1,2}[-/.]\\d{1,2}[-/.]\\d{2,4}\\b"),
    re.compile(r"\\b\\d{4}-\\d{2}-\\d{2}\\b"),
    re.compile(
        r"\\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
        r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|"
        r"nov(?:ember)?|dec(?:ember)?)\\b",
        re.IGNORECASE,
    ),
    re.compile(r"\\b20\\d{2}\\b"),
]


def _is_date_like(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    text = str(value).strip()
    if not text:
        return False
    return any(pat.search(text) for pat in _DATE_PATTERNS)


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _non_empty_count(row):
    return int(row.apply(lambda v: not _is_missing(v)).sum())


def check_dimensions(df):
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1])}


def count_missing_cells(df):
    return int(df.apply(lambda col: col.map(_is_missing)).values.sum())


def missing_by_column(df):
    result = {}
    for idx, col in enumerate(df.columns):
        result[idx] = int(df[col].map(_is_missing).sum())
    return result


def find_empty_rows(df):
    mask = df.apply(lambda col: col.map(_is_missing), axis=0)
    all_missing = mask.all(axis=1)
    return [int(i) for i in df.index[all_missing]]


def find_empty_columns(df):
    result = []
    for pos, col in enumerate(df.columns):
        if df[col].map(_is_missing).all():
            result.append(pos)
    return result


def count_duplicate_rows(df):
    str_df = df.map(lambda v: "" if _is_missing(v) else str(v))
    return int(str_df.duplicated(keep="first").sum())


def non_empty_per_row(df):
    return {int(i): _non_empty_count(df.loc[i]) for i in df.index}


def find_title_rows(df, max_populated_cells=2):
    """
    Identify rows that look like table titles.

    A title row is heuristically defined as a row where the number of
    non-missing cells is at or below *max_populated_cells*.  The assumption is
    that a title typically spans merged cells and pdfplumber surfaces only
    the first cell as populated while the rest are None.

    Parameters
    ----------
    df : pd.DataFrame
    max_populated_cells : int, default 2

    Returns
    -------
    list[int]
    """
    result = []
    for i in df.index:
        n = _non_empty_count(df.loc[i])
        if 1 <= n <= max_populated_cells:
            result.append(int(i))
    return result


def count_date_like_cells(df):
    return int(df.map(_is_date_like).values.sum())


def find_date_like_rows(df):
    mask = df.map(_is_date_like)
    return [int(i) for i in df.index[mask.any(axis=1)]]


def find_possible_header_rows(df, title_row_indices=None, max_header_rows=4):
    """
    Guess which rows are column-header rows.

    Skips title rows, then among the first max_header_rows non-title rows,
    keeps those containing date-like cells or mostly non-numeric text.

    Returns
    -------
    list[int]
    """
    skip = set(title_row_indices or [])
    candidates = [i for i in df.index if int(i) not in skip][:max_header_rows]
    header_rows = []
    date_mask = df.map(_is_date_like)
    for i in candidates:
        row = df.loc[i]
        if date_mask.loc[i].any():
            header_rows.append(int(i))
            continue
        non_missing = [v for v in row if not _is_missing(v)]
        if not non_missing:
            continue
        numeric_count = 0
        for v in non_missing:
            try:
                float(str(v).replace(",", ""))
                numeric_count += 1
            except ValueError:
                pass
        text_ratio = 1 - (numeric_count / len(non_missing))
        if text_ratio >= 0.6:
            header_rows.append(int(i))
    return header_rows


def find_merged_cell_columns(df, min_missing_ratio=0.25):
    """
    Identify columns with merged-cell-like missing-value patterns.

    pdfplumber surfaces merged cells by placing content only in the first
    row and leaving subsequent rows as None.

    Returns
    -------
    list[int]
    """
    result = []
    n_rows = len(df)
    if n_rows == 0:
        return result
    for pos, col in enumerate(df.columns):
        missing_ratio = df[col].map(_is_missing).mean()
        if missing_ratio >= min_missing_ratio:
            result.append(pos)
    return result


def detect_hierarchical_header(df, title_row_indices=None, header_row_indices=None):
    """
    Assess whether the table likely has a multi-row / hierarchical header.

    Uses cautious scoring heuristics.  Returns a dict with keys:
        possible_hierarchical_header (bool)
        confidence ("high" | "medium" | "low" | "none")
        evidence (list[str])
    """
    skip = set(title_row_indices or [])
    header_rows = list(header_row_indices or [])
    evidence = []
    score = 0

    header_rows_not_title = [r for r in header_rows if r not in skip]
    if len(header_rows_not_title) >= 2:
        evidence.append(
            f"{len(header_rows_not_title)} candidate header rows detected "
            f"(rows {header_rows_not_title})"
        )
        score += 2

    n_cols = df.shape[1]
    if n_cols >= 4 and n_cols % 2 == 0:
        evidence.append(
            f"Even column count ({n_cols}) is consistent with paired "
            "sub-columns (e.g. value + shortage per date)"
        )
        score += 1

    if header_rows_not_title:
        first_header_idx = header_rows_not_title[0]
        try:
            row_data = df.loc[first_header_idx]
            missing_in_header = row_data.map(_is_missing).mean()
            if missing_in_header >= 0.4:
                evidence.append(
                    f"Header row {first_header_idx} has "
                    f"{missing_in_header:.0%} missing values, "
                    "suggesting merged date header cells"
                )
                score += 1
        except KeyError:
            pass

    if df.shape[1] > 0:
        col0_missing = df.iloc[:, 0].map(_is_missing).mean()
        if col0_missing >= 0.5:
            evidence.append(
                f"First column has {col0_missing:.0%} missing values, "
                "consistent with merged region/group labels"
            )
            score += 1

    if score >= 4:
        confidence, possible = "high", True
    elif score >= 2:
        confidence, possible = "medium", True
    elif score == 1:
        confidence, possible = "low", False
    else:
        confidence, possible = "none", False

    return {
        "possible_hierarchical_header": possible,
        "confidence": confidence,
        "evidence": evidence,
    }


def generate_warnings(df, missing_by_col, empty_rows, empty_columns,
                      duplicate_count, merged_cols, hierarchical_info):
    """Produce human-readable warning strings without modifying the DataFrame."""
    warnings = []
    n_rows, n_cols = df.shape

    if n_cols > 0 and missing_by_col.get(0, 0) > 0:
        ratio = missing_by_col[0] / n_rows if n_rows else 0
        warnings.append(
            f"Missing values detected in leading column (column 0): "
            f"{missing_by_col[0]} of {n_rows} cells ({ratio:.0%}). "
            "May indicate merged region/group cells."
        )
    if empty_rows:
        warnings.append(
            f"Completely empty rows detected at index(es): {empty_rows}."
        )
    if empty_columns:
        warnings.append(
            f"Completely empty columns detected at position(s): {empty_columns}."
        )
    if duplicate_count > 0:
        warnings.append(f"{duplicate_count} duplicate row(s) detected.")
    if merged_cols:
        warnings.append(
            f"High missing-value ratio in column(s) {merged_cols}; "
            "possible merged-cell artefact from PDF extraction."
        )
    if hierarchical_info.get("possible_hierarchical_header"):
        conf = hierarchical_info.get("confidence", "unknown")
        warnings.append(
            f"Table may contain a multi-row / hierarchical header "
            f"(confidence: {conf}). Treat header rows with care before "
            "setting column names."
        )
    return warnings


def validate_table(df, title_max_populated=2, merged_min_missing_ratio=0.25):
    """
    Run a read-only structural validation of a raw extracted DataFrame.

    The input DataFrame is **never modified**.

    Parameters
    ----------
    df : pd.DataFrame
    title_max_populated : int, default 2
    merged_min_missing_ratio : float, default 0.25

    Returns
    -------
    dict with keys: rows, columns, missing_cells, missing_by_column,
        duplicate_rows, empty_rows, empty_columns, non_empty_per_row,
        possible_title_rows, possible_header_rows, date_like_cells,
        date_like_rows, merged_cell_columns, hierarchical_header, warnings
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"validate_table expects a pandas DataFrame, got {type(df).__name__}."
        )

    _df = copy.deepcopy(df)
    _df = _df.reset_index(drop=True)

    dims = check_dimensions(_df)
    missing_total = count_missing_cells(_df)
    missing_col = missing_by_column(_df)
    dup_count = count_duplicate_rows(_df)
    empty_r = find_empty_rows(_df)
    empty_c = find_empty_columns(_df)
    non_empty = non_empty_per_row(_df)
    title_rows = find_title_rows(_df, max_populated_cells=title_max_populated)
    date_cells = count_date_like_cells(_df)
    date_rows = find_date_like_rows(_df)
    header_rows = find_possible_header_rows(_df, title_row_indices=title_rows)
    merged_cols = find_merged_cell_columns(
        _df, min_missing_ratio=merged_min_missing_ratio
    )
    hier_info = detect_hierarchical_header(
        _df, title_row_indices=title_rows, header_row_indices=header_rows
    )
    warns = generate_warnings(
        _df,
        missing_by_col=missing_col,
        empty_rows=empty_r,
        empty_columns=empty_c,
        duplicate_count=dup_count,
        merged_cols=merged_cols,
        hierarchical_info=hier_info,
    )

    return {
        "rows": dims["rows"],
        "columns": dims["columns"],
        "missing_cells": missing_total,
        "missing_by_column": missing_col,
        "duplicate_rows": dup_count,
        "empty_rows": empty_r,
        "empty_columns": empty_c,
        "non_empty_per_row": non_empty,
        "possible_title_rows": title_rows,
        "possible_header_rows": header_rows,
        "date_like_cells": date_cells,
        "date_like_rows": date_rows,
        "merged_cell_columns": merged_cols,
        "hierarchical_header": hier_info,
        "warnings": warns,
    }


def assert_no_mutation(df_before, df_after):
    """
    Verify that df_after is identical to df_before.

    Returns True if identical, raises AssertionError otherwise.
    """
    try:
        pd.testing.assert_frame_equal(
            df_before, df_after, check_dtype=False, check_like=False
        )
    except AssertionError as exc:
        raise AssertionError(
            f"DataFrame was mutated by validate_table: {exc}"
        ) from exc
    return True
'''

with open('app/validation/table_validator.py', 'w', encoding='utf-8') as f:
    f.write(CODE)

print('Written', len(CODE), 'chars')
