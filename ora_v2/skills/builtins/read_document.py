"""Skill: read text from PDF, TXT, DOCX files."""
from __future__ import annotations
from pathlib import Path

DESCRIPTION = "Read and extract text from a local file (PDF, TXT, DOCX, MD). Returns the text content."
ARGS = {"path": "absolute or relative path to the file"}


def run(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"ERROR: file not found: {path}"

    suffix = p.suffix.lower()

    if suffix in (".txt", ".md"):
        return p.read_text(encoding="utf-8", errors="replace")[:20000]

    if suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(p))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)[:20000]
        except ImportError:
            return "ERROR: pypdf not installed. Run: pip install pypdf"
        except Exception as e:
            return f"ERROR reading PDF: {e}"

    if suffix in (".docx",):
        try:
            import docx
            doc = docx.Document(str(p))
            return "\n".join(para.text for para in doc.paragraphs)[:20000]
        except ImportError:
            return "ERROR: python-docx not installed. Run: pip install python-docx"
        except Exception as e:
            return f"ERROR reading DOCX: {e}"

    return f"ERROR: unsupported file type: {suffix}"
