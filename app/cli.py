"""Command-line interface for the profile-driven document table agent."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from app.agent import (
    QueryResult,
    TableQueryAgent,
    parse_query_request,
)
from app.document_service import analysis_tables, ask_document_question
from app.export.csv_exporter import export_dataframe_to_csv
from app.pipeline.builtin_profiles import BUILTIN_DOCUMENT_PROFILES
from app.pipeline.runner import DocumentPipelineResult, run_profiled_pipeline


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _validate_pdf_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {resolved}")
    if resolved.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {resolved}")
    return resolved


def _run_pipeline(
    pdf_path: Path,
    *,
    profile_name: str | None,
    output_dir: Path | None = None,
    overwrite: bool = False,
) -> DocumentPipelineResult:
    return run_profiled_pipeline(
        _validate_pdf_path(pdf_path),
        BUILTIN_DOCUMENT_PROFILES,
        profile_name=profile_name,
        output_dir=output_dir,
        overwrite=overwrite,
    )


def _print_dataframe(df: pd.DataFrame) -> None:
    print(df.to_string(index=False))


def _json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _command_profiles(_args: argparse.Namespace) -> int:
    for profile in BUILTIN_DOCUMENT_PROFILES:
        print(profile.name)
        print(f"  Detection terms: {', '.join(profile.detection_terms)}")
        for table in profile.tables:
            derived = ", ".join(item.name for item in table.postprocessors)
            suffix = f"; derived: {derived}" if derived else ""
            print(f"  Table: {table.name}{suffix}")
    return 0


def _command_process(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    result = _run_pipeline(
        args.pdf,
        profile_name=args.profile,
        output_dir=output_dir,
        overwrite=args.overwrite,
    )
    print(f"Profile: {result.profile_name}")
    for table_name, table_result in result.tables.items():
        page_label = (
            f"page {table_result.page_number}"
            if len(table_result.page_numbers) == 1
            else (
                f"pages {table_result.page_numbers[0]}-"
                f"{table_result.page_numbers[-1]}"
            )
        )
        print(
            f"Table {table_name}: {page_label}, "
            f"raw {table_result.raw_table.shape[0]}x"
            f"{table_result.raw_table.shape[1]}, clean "
            f"{table_result.transformed_table.shape[0]}x"
            f"{table_result.transformed_table.shape[1]}"
        )
        if table_result.output_path is not None:
            print(f"  CSV: {table_result.output_path.resolve()}")
        for output_name, table in table_result.postprocessed_tables.items():
            print(
                f"  Derived {output_name}: "
                f"{table.shape[0]}x{table.shape[1]}"
            )
            output_paths = table_result.postprocessed_output_paths or {}
            if output_name in output_paths:
                print(f"  CSV: {output_paths[output_name].resolve()}")
        for warning in table_result.warnings:
            print(f"  Warning: {warning}")
    return 0


def _load_query_payload(args: argparse.Namespace) -> str:
    if args.request is not None:
        return args.request
    request_path = args.request_file.expanduser().resolve()
    if not request_path.is_file():
        raise FileNotFoundError(f"Request file does not exist: {request_path}")
    return request_path.read_text(encoding="utf-8-sig")


def _agent_for_pdf(args: argparse.Namespace) -> TableQueryAgent:
    result = _run_pipeline(args.pdf, profile_name=args.profile)
    return TableQueryAgent(analysis_tables(result))


def _command_tables(args: argparse.Namespace) -> int:
    descriptions = _agent_for_pdf(args).describe_tables()
    for name, details in descriptions.items():
        print(f"{name}: {details['rows']} rows")
        for column in details["columns"]:
            print(f"  {column}: {details['dtypes'][str(column)]}")
    return 0


def _emit_query_result(
    args: argparse.Namespace,
    result: QueryResult,
    *,
    question: str | None = None,
    metric_name: str | None = None,
    notes: tuple[str, ...] = (),
) -> int:
    output_path = None
    if args.answer_output is not None:
        output_path = export_dataframe_to_csv(
            result.answer,
            args.answer_output.expanduser().resolve(),
            overwrite=args.overwrite,
        )

    if args.format == "json":
        response: dict[str, Any] = {
            "table": result.request.table,
            "request": asdict(result.request),
            "source_row_count": result.source_row_count,
            "matched_row_count": result.matched_row_count,
            "returned_row_count": result.returned_row_count,
            "answer": _json_records(result.answer),
        }
        if question is not None:
            response["question"] = question
            response["matched_metric"] = metric_name
            response["interpretation_notes"] = list(notes)
        if output_path is not None:
            response["answer_csv"] = str(output_path.resolve())
        if args.show_evidence:
            response["evidence"] = _json_records(
                result.evidence.head(args.evidence_limit)
            )
            response["evidence_rows_shown"] = min(
                len(result.evidence),
                args.evidence_limit,
            )
        print(json.dumps(response, indent=2))
    else:
        if question is not None:
            print(f"Question: {question}")
            print(f"Matched metric: {metric_name}")
            print("Interpreted request")
            print(json.dumps(asdict(result.request), indent=2))
            for note in notes:
                print(f"Note: {note}")
        print("Answer")
        _print_dataframe(result.answer)
        print(
            f"Rows: {result.returned_row_count} returned; "
            f"{result.matched_row_count} source rows matched."
        )
        if args.show_evidence:
            shown = result.evidence.head(args.evidence_limit)
            print(f"Evidence (showing {len(shown)} of {len(result.evidence)} rows)")
            _print_dataframe(shown)
        if output_path is not None:
            print(f"Answer CSV: {output_path.resolve()}")
    return 0


def _command_query(args: argparse.Namespace) -> int:
    request = parse_query_request(_load_query_payload(args))
    agent = _agent_for_pdf(args)
    return _emit_query_result(args, agent.execute(request))


def _command_ask(args: argparse.Namespace) -> int:
    pipeline_result = _run_pipeline(args.pdf, profile_name=args.profile)
    natural_result = ask_document_question(pipeline_result, args.question)
    interpretation = natural_result.interpretation
    return _emit_query_result(
        args,
        natural_result.query_result,
        question=interpretation.question,
        metric_name=interpretation.metric_name,
        notes=interpretation.notes,
    )


def _add_pdf_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pdf", type=Path, help="PDF document to process")
    parser.add_argument(
        "--profile",
        choices=[profile.name for profile in BUILTIN_DOCUMENT_PROFILES],
        help="select a profile explicitly instead of automatic detection",
    )


def _add_result_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="terminal output format (default: table)",
    )
    parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="include a preview of filtered source rows",
    )
    parser.add_argument(
        "--evidence-limit",
        type=_positive_int,
        default=20,
        help="maximum evidence rows to show (default: 20)",
    )
    parser.add_argument(
        "--answer-output",
        type=Path,
        help="optionally export answer rows to a CSV file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow an existing answer CSV to be replaced",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command parser used by ``main`` and CLI tests."""

    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Extract, transform, and query tables from profiled PDFs.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    profiles_parser = commands.add_parser(
        "profiles",
        help="list built-in document profiles",
    )
    profiles_parser.set_defaults(handler=_command_profiles)

    process_parser = commands.add_parser(
        "process",
        help="run the pipeline and export configured CSV files",
    )
    _add_pdf_arguments(process_parser)
    process_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="CSV destination directory (default: outputs)",
    )
    process_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="allow existing output CSV files to be replaced",
    )
    process_parser.set_defaults(handler=_command_process)

    tables_parser = commands.add_parser(
        "tables",
        help="show queryable analysis tables and their schemas",
    )
    _add_pdf_arguments(tables_parser)
    tables_parser.set_defaults(handler=_command_tables)

    query_parser = commands.add_parser(
        "query",
        help="execute a validated JSON query against a PDF",
    )
    _add_pdf_arguments(query_parser)
    request_group = query_parser.add_mutually_exclusive_group(required=True)
    request_group.add_argument(
        "--request",
        help="inline JSON request object",
    )
    request_group.add_argument(
        "--request-file",
        type=Path,
        help="path to a UTF-8 JSON request file",
    )
    _add_result_arguments(query_parser)
    query_parser.set_defaults(handler=_command_query)

    ask_parser = commands.add_parser(
        "ask",
        help="translate and execute a supported English question",
    )
    _add_pdf_arguments(ask_parser)
    ask_parser.add_argument(
        "question",
        help="quoted English question about one supported metric",
    )
    _add_result_arguments(ask_parser)
    ask_parser.set_defaults(handler=_command_ask)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit status."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
