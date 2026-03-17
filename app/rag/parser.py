from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree


class DocumentParseError(RuntimeError):
    pass


def parse_document(path: Path, mime_type: str | None = None) -> tuple[str, str]:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8"), "text"
    if suffix == ".docx":
        return _parse_docx(path), "docx"
    if suffix == ".xlsx":
        return _parse_xlsx(path), "xlsx"
    if suffix == ".pdf":
        return _parse_pdf_fallback(path), "pdf"
    raise DocumentParseError(f"Unsupported document type: {suffix or mime_type or 'unknown'}")


def _parse_docx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            xml_data = archive.read("word/document.xml")
    except (FileNotFoundError, KeyError, zipfile.BadZipFile) as exc:
        raise DocumentParseError(f"Failed to parse DOCX file: {path.name}") from exc
    root = ElementTree.fromstring(xml_data)
    texts = [node.text for node in root.iter() if node.tag.endswith("}t") and node.text]
    return "\n".join(part.strip() for part in texts if part.strip())


def _parse_xlsx(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            shared_strings = _read_shared_strings(archive)
            sheet_names = sorted(name for name in archive.namelist() if name.startswith("xl/worksheets/sheet"))
            rows: list[str] = []
            for sheet_name in sheet_names:
                xml_data = archive.read(sheet_name)
                rows.extend(_parse_sheet_rows(xml_data, shared_strings))
    except (FileNotFoundError, KeyError, zipfile.BadZipFile) as exc:
        raise DocumentParseError(f"Failed to parse XLSX file: {path.name}") from exc
    return "\n".join(item for item in rows if item)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return [node.text or "" for node in root.iter() if node.tag.endswith("}t")]


def _parse_sheet_rows(xml_data: bytes, shared_strings: list[str]) -> list[str]:
    root = ElementTree.fromstring(xml_data)
    namespace = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[str] = []
    for row in root.findall(".//main:row", namespace):
        values: list[str] = []
        for cell in row.findall("main:c", namespace):
            cell_type = cell.attrib.get("t")
            value_node = cell.find("main:v", namespace)
            if value_node is None or value_node.text is None:
                continue
            raw_value = value_node.text
            if cell_type == "s":
                try:
                    values.append(shared_strings[int(raw_value)])
                except (IndexError, ValueError):
                    values.append(raw_value)
            else:
                values.append(raw_value)
        if values:
            rows.append("\t".join(values))
    return rows


def _parse_pdf_fallback(path: Path) -> str:
    data = path.read_bytes()
    text = data.decode("latin-1", errors="ignore")
    strings = re.findall(r"\(([^()]*)\)", text)
    cleaned = [item.strip() for item in strings if _looks_like_text(item)]
    if cleaned:
        return "\n".join(cleaned)
    ascii_blocks = re.findall(r"[A-Za-z0-9\u4e00-\u9fff][A-Za-z0-9\u4e00-\u9fff\s,:;._/\-]{8,}", text)
    cleaned = [item.strip() for item in ascii_blocks if _looks_like_text(item)]
    if cleaned:
        return "\n".join(cleaned)
    raise DocumentParseError(f"Failed to extract text from PDF file: {path.name}")


def _looks_like_text(value: str) -> bool:
    compact = re.sub(r"\s+", "", value)
    return len(compact) >= 4 and any(char.isalpha() or "\u4e00" <= char <= "\u9fff" for char in compact)
