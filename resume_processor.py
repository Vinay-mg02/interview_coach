"""
resume_processor.py
-------------------
Local, zero-cost PDF text extractor using pypdf.
Reads raw bytes in-memory and returns a clean string block.
"""

import logging
from io import BytesIO

logger = logging.getLogger(__name__)


def extract_resume_text(file_bytes: bytes) -> str:
    """
    Extracts all text from a PDF supplied as raw bytes.

    Args:
        file_bytes: Raw bytes of the uploaded PDF file.

    Returns:
        A single clean string containing all extracted text.

    Raises:
        ValueError: If the file is not a valid PDF or text cannot be extracted.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))

        if len(reader.pages) == 0:
            raise ValueError("PDF file appears to be empty (0 pages found).")

        page_texts = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                page_texts.append(text.strip())
            else:
                logger.warning(f"[ResumeProcessor] Page {i + 1} yielded no text — may be image-based.")

        if not page_texts:
            raise ValueError(
                "No readable text found in PDF. The file may be image-based or scanned. "
                "Please upload a text-based PDF resume."
            )

        full_text = "\n\n".join(page_texts)

        # Normalise whitespace
        import re
        full_text = re.sub(r"\n{3,}", "\n\n", full_text)  # collapse excess blank lines
        full_text = re.sub(r"[ \t]+", " ", full_text)      # collapse multiple spaces/tabs

        logger.info(
            f"[ResumeProcessor] ✅ Successfully extracted {len(page_texts)} page(s), "
            f"{len(full_text.split())} words total."
        )
        return full_text.strip()

    except ImportError:
        raise RuntimeError("pypdf is not installed. Run: pip install pypdf")
    except Exception as e:
        logger.error(f"[ResumeProcessor] ❌ Failed to extract PDF text: {e}")
        raise ValueError(f"Could not read the uploaded PDF: {str(e)}")
