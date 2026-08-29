"""
Render the user guide to PDF.

Run from the project root:

    python docs/build_pdf.py

Uses WeasyPrint, which supports CSS Paged Media - so the footer page numbers and the
contents-page references (via target-counter) are generated automatically rather than
being typed in by hand and going stale.
"""

from pathlib import Path

from weasyprint import HTML

DOCS = Path(__file__).resolve().parent
SOURCE = DOCS / "user_guide.html"
OUTPUT = DOCS / "DataCo_User_Guide.pdf"


def main() -> int:
    if not SOURCE.exists():
        print(f"ERROR: {SOURCE} not found")
        return 1

    # base_url lets the relative screenshot paths resolve against docs/.
    HTML(filename=str(SOURCE), base_url=str(DOCS)).write_pdf(str(OUTPUT))

    size_mb = OUTPUT.stat().st_size / 1e6
    print(f"Written {OUTPUT}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
