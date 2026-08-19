# document-table-agent

A profile-driven Python pipeline for extracting, validating, transforming,
and exporting tables from PDFs.

## Architecture

```text
PDF parsing → table extraction → validation → profile transformation → export
```

The parser, validator, transformation utilities, exporter, and pipeline runner
are generic. Document-specific choices live in immutable profiles: search
markers, table selection, header structure, identity/measure columns, optional
text cleanup, optional mappings, and expected output schema.

This is deliberately not an "any PDF without configuration" system. Unknown
or ambiguous layouts stop with a clear profile error rather than being guessed.
Adding a new PDF family means adding a new `DocumentProfile`; core modules do
not need to be rewritten.

## Included profiles

`grid_india_weekly_report` handles the supplied sample's Energy Consumption and
Maximum Demand tables. Its layout and domain mapping are isolated in
`app/pipeline/builtin_profiles.py` and
`app/transformation/region_mapping.py`.

`idsp_weekly_outbreak_report` handles the supplied 16-page IDSP Weekly Outbreak
Report. It uses marker-bounded page spans to stitch the current-week table on
pages 3-14 and the late-report table on pages 15-16. Profile-specific row
reconstruction is isolated in `app/transformation/outbreak_analysis.py`; raw
page extraction remains unchanged. Use
`notebooks/08_second_document_profile.ipynb` for the complete walkthrough.

## Run checks

```powershell
Set-Location 'C:\Users\vinee\document-table-agent'
python -m pip install -r requirements-dev.txt
python -m pytest -q -p no:cacheprovider
```

## Local web interface

Run the Streamlit interface from the project directory:

```powershell
python -m streamlit run app\web_app.py
```

Your browser will open at `http://localhost:8501`. Upload a supported PDF and
the app will detect its profile, extract its configured tables, show the
analysis-ready data, answer supported English questions, display validation
observations, and provide individual CSV or combined ZIP downloads. Uploaded
files are processed through a temporary local file and are not saved to the
project's `outputs` directory.

Open the notebooks with:

```powershell
jupyter lab
```

Use `notebooks/05_profile_driven_pipeline.ipynb` for the complete configured
pipeline.

## Analysis-ready outputs

The built-in electricity table profiles also register optional long-format
postprocessors. The generic runner validates and exports:

- `outputs/energy_consumption_long.csv`
- `outputs/maximum_demand_long.csv`

Use `notebooks/06_analysis_ready_tables.ipynb` to inspect their schemas,
uniqueness checks, parsed dates, numeric measures, and Energy total
reconciliation. These postprocessors belong to the electricity profile; a
different document family can register different derived outputs.

The outbreak profile exports:

- `outputs/clean_current_outbreaks.csv`
- `outputs/current_outbreaks_analysis.csv`
- `outputs/clean_late_outbreaks.csv`
- `outputs/late_outbreaks_analysis.csv`

The analysis outputs contain parsed dates, numeric case/death measures, unique
outbreak IDs, and one logical record per outbreak.

## Query analysis-ready tables

`app/agent/` provides a deterministic query layer over flat Pandas tables. It
accepts validated JSON objects or Python dictionaries with allowlisted filters,
grouping, aggregations, ordering, column selection, and result limits. It never
uses arbitrary expression evaluation and returns the filtered source rows as
evidence alongside every answer.

Use `notebooks/07_table_query_agent.ipynb` for complete examples, including
weekly energy rankings, state/date demand filters, regional peak-shortage
aggregation, and a JSON request. Natural-language interpretation is deliberately
kept separate so it can later translate questions into the same typed request
contract without changing query execution.

## Command-line usage

The project can also run without Jupyter. From the project directory:

```powershell
$pdfPath = 'sample_documents\Weekly 300326 to 050426_544 (1).pdf'
python -m app.cli profiles
python -m app.cli process $pdfPath --overwrite
python -m app.cli tables $pdfPath
python -m app.cli query $pdfPath `
  --request-file examples\query_top_energy.json `
  --show-evidence
python -m app.cli ask $pdfPath `
  "Which state had the highest energy consumption?" `
  --show-evidence

$outbreakPdf = 'sample_documents\75788759701752062509.pdf'
python -m app.cli process $outbreakPdf --overwrite
python -m app.cli ask $outbreakPdf `
  "Which state had the most outbreak cases?"
```

Change `$pdfPath` when processing another document. Run
`python -m app.cli --help` or append `--help` to any subcommand for all options.
Existing CSVs are protected unless `--overwrite` is supplied.

## Supported English questions

The `ask` command uses a deterministic semantic catalog, not an LLM or API
key. Each detected document profile supplies its own vocabulary. The electricity
profile recognizes energy consumption, maximum demand, and peak shortage. The
outbreak profile recognizes current/late cases and deaths. Supported patterns
include:

- `Which state had the highest energy consumption?`
- `Show Punjab's demand from 2026-03-30 to 2026-04-05`
- `Calculate average peak shortage by region`
- `Compare maximum demand for Delhi and Haryana`
- `Show the top 5 states by energy consumption`
- `Which state had the most outbreak cases?`
- `Which disease had the most deaths?`
- `Show cases for Dengue`
- `Show late reported cases by state`

It supports total, average, median, minimum, and maximum aggregation;
profile-declared dimensions such as state, region, district, and disease;
top/bottom limits; recognized entity values; and exact dates or date ranges in
`YYYY-MM-DD` or `DD-MM-YYYY` form. The CLI prints the validated request it
generated before the answer. Multiple metrics, conflicting aggregations,
unclear comparisons, and unsupported wording fail explicitly instead of being
guessed.

## Add another PDF style

1. Create a `TableTransformationProfile` for each required table.
2. Define its search terms, table index, header rows, identity columns, measure
   columns, cleanup/mapping rules, and output schema.
3. Group the table profiles in a `DocumentProfile` with distinctive detection
   terms.
4. Register the document profile alongside the built-in profiles.
5. Add synthetic layout tests and one representative PDF integration test.

Profile detection requires exactly one match. A new or ambiguous document is
never silently processed using the wrong layout.
