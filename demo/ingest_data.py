import csv
import io
import re
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import DataFrameLoader

DATA_PATH = "../data/main-data/synthetic-resumes.csv"
FAISS_PATH = "../vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ROWS = 1500
MIN_RESUME_CHARS = 30
MAX_RESUME_CHARS = 80_000
EMPTY_TOKENS = {
  "", "nan", "none", "null", "n/a", "na", "nil", "-", "--", ".", "undefined", "#n/a",
}

ID_COLUMN_ALIASES = {
  "id", "ids", "applicant id", "applicant_id", "applicantid", "candidate id",
  "candidate_id", "candidateid", "resume id", "resume_id", "resumeid", "index",
  "name", "full name", "fullname", "candidate", "applicant", "email", "e mail",
  "employee id", "employee_id", "user id", "user_id", "uid", "pk",
}
RESUME_COLUMN_ALIASES = {
  "resume", "resumes", "cv", "cvs", "text", "content", "description",
  "resume_text", "resume text", "full_text", "full text", "profile", "bio",
  "summary", "details", "body", "document", "raw_text", "raw text",
  "cover_letter", "experience", "work experience",
}


class CsvImportError(ValueError):
  pass


def _normalize_column_name(name: str) -> str:
  cleaned = str(name or "").replace("\ufeff", "").strip().lower()
  cleaned = re.sub(r"[\s\-]+", "_", cleaned)
  cleaned = re.sub(r"[^a-z0-9_]+", "", cleaned)
  cleaned = re.sub(r"_+", "_", cleaned).strip("_")
  return cleaned


def _clean_cell(value) -> str:
  if value is None or (isinstance(value, float) and pd.isna(value)):
    return ""
  text = str(value).replace("\x00", " ").strip()
  if text.lower() in EMPTY_TOKENS:
    return ""
  if re.fullmatch(r"\d+\.0", text):
    text = text[:-2]
  return re.sub(r"[ \t]+", " ", text)


def _read_uploaded_bytes(uploaded_file) -> bytes:
  try:
    uploaded_file.seek(0)
  except Exception:
    pass
  raw = uploaded_file.read()
  if isinstance(raw, str):
    raw = raw.encode("utf-8")
  if not raw or not str(raw).strip():
    raise CsvImportError("The uploaded file is empty.")
  if len(raw) > MAX_UPLOAD_BYTES:
    raise CsvImportError("The CSV is larger than 25 MB. Please split it into smaller files.")
  if raw.startswith(b"PK"):
    raise CsvImportError("This looks like an Excel workbook (.xlsx). Please export it as CSV and try again.")
  if raw.startswith(b"\xd0\xcf\x11\xe0"):
    raise CsvImportError("This looks like a legacy Excel file (.xls). Please export it as CSV and try again.")
  raw = raw.replace(b"\x00", b"")
  return raw


def _decode_bytes(raw: bytes) -> str:
  if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
    return raw.decode("utf-16")
  encodings = ("utf-8-sig", "utf-8", "cp1252", "latin-1")
  for encoding in encodings:
    try:
      return raw.decode(encoding)
    except UnicodeDecodeError:
      continue
  return raw.decode("utf-8", errors="replace")


def _prepare_text(text: str) -> str:
  text = text.replace("\r\n", "\n").replace("\r", "\n")
  if text.lower().startswith("sep="):
    first_line, _, remainder = text.partition("\n")
    text = remainder
  lines = [line for line in text.split("\n") if line.strip() != ""]
  return "\n".join(lines)


def _sniff_delimiter(sample: str) -> str:
  header = sample.split("\n", 1)[0] if sample else ""
  try:
    dialect = csv.Sniffer().sniff(sample[:8192], delimiters=",;\t|")
    if dialect.delimiter:
      return dialect.delimiter
  except csv.Error:
    pass
  counts = {delimiter: header.count(delimiter) for delimiter in [",", ";", "\t", "|"]}
  best = max(counts, key=counts.get)
  return best if counts[best] > 0 else ","


def _looks_like_header(values) -> bool:
  cells = [_clean_cell(value) for value in values]
  nonempty = [cell for cell in cells if cell]
  if not nonempty:
    return False
  short_cells = sum(1 for cell in nonempty if len(cell) <= 40)
  return short_cells / len(nonempty) >= 0.6 and all(len(cell) <= 80 for cell in nonempty)


def _dedupe_columns(columns):
  seen = {}
  unique = []
  for column in columns:
    base = column or "column"
    count = seen.get(base, 0)
    seen[base] = count + 1
    unique.append(base if count == 0 else f"{base}_{count + 1}")
  return unique


def _load_dataframe(text: str, delimiter: str) -> pd.DataFrame:
  options = dict(
    sep=delimiter,
    dtype=str,
    engine="python",
    quoting=csv.QUOTE_MINIMAL,
    skip_blank_lines=True,
    on_bad_lines="skip",
    keep_default_na=True,
  )
  buffer = io.StringIO(text)
  dataframe = pd.read_csv(buffer, **options)
  if dataframe.shape[1] == 1 and not _looks_like_header(dataframe.columns):
    buffer = io.StringIO(text)
    dataframe = pd.read_csv(buffer, header=None, names=["Resume"], **options)
  return dataframe


def _find_column(columns, aliases):
  normalized_aliases = {_normalize_column_name(alias) for alias in aliases}
  for column in columns:
    if _normalize_column_name(column) in normalized_aliases:
      return column
  return None


def _is_metadata_column(column: str) -> bool:
  name = _normalize_column_name(column)
  return name.startswith("unnamed") or name in {"row_number", "row", "unnamed_0"}


def _build_resume_text(row: pd.Series, resume_column: str, id_column: str) -> str:
  if resume_column:
    return _clean_cell(row.get(resume_column, ""))
  parts = []
  for column, value in row.items():
    if column in {"ID", "Resume"} or column == id_column or _is_metadata_column(column):
      continue
    cleaned = _clean_cell(value)
    if not cleaned:
      continue
    parts.append(f"{column}: {cleaned}")
  return "\n".join(parts)


def _unique_ids(values: pd.Series) -> pd.Series:
  seen = {}
  unique = []
  for value in values:
    base = _clean_cell(value) or "candidate"
    count = seen.get(base, 0) + 1
    seen[base] = count
    unique.append(base if count == 1 else f"{base}-{count}")
  return pd.Series(unique, index=values.index)


def read_resume_csv(uploaded_file):
  filename = str(getattr(uploaded_file, "name", "upload.csv"))
  if not filename.lower().endswith(".csv"):
    raise CsvImportError("Please upload a .csv file.")

  raw = _read_uploaded_bytes(uploaded_file)
  text = _prepare_text(_decode_bytes(raw))
  if not text.strip():
    raise CsvImportError("The CSV file contains no readable text.")

  delimiter = _sniff_delimiter(text)
  try:
    dataframe = _load_dataframe(text, delimiter)
  except Exception as error:
    raise CsvImportError(f"The CSV could not be parsed. {error}") from error

  dataframe.columns = _dedupe_columns([str(column).replace("\ufeff", "").strip() or f"column_{index + 1}" for index, column in enumerate(dataframe.columns)])
  dataframe = dataframe.dropna(how="all")
  dataframe = dataframe.loc[:, ~dataframe.columns.to_series().map(_is_metadata_column)]
  if dataframe.empty:
    raise CsvImportError("The CSV has a header but no data rows.")

  original_rows = len(dataframe)
  id_column = _find_column(dataframe.columns, ID_COLUMN_ALIASES)
  resume_column = _find_column(dataframe.columns, RESUME_COLUMN_ALIASES)

  resumes = dataframe.apply(lambda row: _build_resume_text(row, resume_column, id_column), axis=1)
  if id_column is None:
    ids = pd.Series([str(index) for index in range(len(dataframe))], index=dataframe.index)
    id_source = "generated row numbers"
  else:
    ids = dataframe[id_column].map(_clean_cell).replace("", pd.NA)
    missing_ids = ids.isna()
    if missing_ids.any():
      ids.loc[missing_ids] = [f"candidate-{index + 1}" for index in range(missing_ids.sum())]
    id_source = id_column

  prepared = pd.DataFrame({"ID": ids.astype(str), "Resume": resumes.astype(str)})
  prepared["Resume"] = prepared["Resume"].str.slice(0, MAX_RESUME_CHARS)
  dropped_short = int((prepared["Resume"].str.len() < MIN_RESUME_CHARS).sum())
  prepared = prepared[prepared["Resume"].str.len() >= MIN_RESUME_CHARS].copy()
  if prepared.empty:
    raise CsvImportError(
      "No resume text was found. Add a Resume/CV/text column, or include enough text in each row."
    )

  before_dedupe = len(prepared)
  prepared["ID"] = _unique_ids(prepared["ID"])
  truncated = False
  if len(prepared) > MAX_ROWS:
    prepared = prepared.head(MAX_ROWS).copy()
    truncated = True

  prepared = prepared.reset_index(drop=True)
  report = {
    "filename": filename,
    "rows_in_file": original_rows,
    "rows_indexed": len(prepared),
    "rows_dropped_short": dropped_short,
    "duplicate_ids_renamed": max(0, before_dedupe - len(prepared["ID"].unique()) + (before_dedupe - before_dedupe)),
    "id_column": id_source,
    "resume_column": resume_column or "combined text columns",
    "delimiter": "tab" if delimiter == "\t" else delimiter,
    "truncated": truncated,
  }
  prepared.attrs["import_report"] = report
  return prepared


def format_import_report(report: dict) -> str:
  message = (
    f"Indexed {report['rows_indexed']} resume(s) from `{report['filename']}`. "
    f"ID source: {report['id_column']}. Text source: {report['resume_column']}."
  )
  if report["rows_dropped_short"]:
    message += f" Skipped {report['rows_dropped_short']} row(s) with too little text."
  if report["truncated"]:
    message += f" Only the first {MAX_ROWS} valid rows were indexed."
  return message


def ingest(df: pd.DataFrame, content_column: str, embedding_model):
  if df is None or df.empty:
    raise CsvImportError("There are no resumes to index.")
  loader = DataFrameLoader(df, page_content_column=content_column)
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=500,
  )
  documents = loader.load()
  document_chunks = text_splitter.split_documents(documents)
  document_chunks = [chunk for chunk in document_chunks if chunk.page_content and chunk.page_content.strip()]
  if not document_chunks:
    raise CsvImportError("The CSV was read, but no searchable text chunks could be created.")
  return FAISS.from_documents(document_chunks, embedding_model, distance_strategy=DistanceStrategy.COSINE)
