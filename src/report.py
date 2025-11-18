# src/report.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PyPDF2 import PdfMerger


class HermesReport:
    """
    Collect plots and PDFs and merge into a single HERMES_<version>.pdf.
    """

    def __init__(self, version: str, out_dir: str | Path = "results"):
        self.version = str(version)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._parts: List[Path] = []

    def add_pngs_as_pdf(
        self,
        paths: Iterable[str | Path],
        out_name: str = "HERMES_panels.pdf",
    ) -> Path:
        """
        Take a list of plot files and wrap them into a single multi-page PDF.

        - If an input file is already a PDF, it is appended as-is (vector).
        - Otherwise (PNG/SVG/etc.), it is drawn into a Matplotlib figure and
          that page is written as PDF (raster content inside, but the page
          itself is vector).
        """
        paths = [Path(p) for p in paths]
        pdf_path = self.out_dir / out_name
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

        merger = PdfMerger()

        for p in paths:
            suffix = p.suffix.lower()
            if suffix == ".pdf":
                # already vector: append directly
                merger.append(str(p))
            else:
                # fallback: read image and embed into a single-page PDF
                img = mpimg.imread(p)
                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axis("off")
                # save temporary one-page pdf and append
                tmp_pdf = pdf_path.parent / f"tmp_{p.stem}.pdf"
                fig.savefig(tmp_pdf, bbox_inches="tight")
                plt.close(fig)
                merger.append(str(tmp_pdf))
                # you can uncomment the next line if you want to clean temp files
                # tmp_pdf.unlink(missing_ok=True)

        with open(pdf_path, "wb") as f:
            merger.write(f)
        merger.close()

        self._parts.append(pdf_path)
        return pdf_path

    def add_pdf(self, pdf_path: str | Path) -> Path:
        pdf_path = Path(pdf_path)
        self._parts.append(pdf_path)
        return pdf_path

    def build(self) -> Path:
        """
        Merge all added PDFs into final HERMES_<version>.pdf
        and return its Path.
        """
        final_path = self.out_dir / f"HERMES_{self.version}.pdf"
        merger = PdfMerger()
        for p in self._parts:
            merger.append(str(p))
        with open(final_path, "wb") as f:
            merger.write(f)
        merger.close()
        return final_path
