"""Report generation for motion evaluation results.

Generates publication-ready reports in multiple formats:
    - HTML: Interactive report with Plotly charts
    - LaTeX: Booktabs table for copy-paste into papers
    - PDF: Rendered version of the HTML report
"""

from .report_generator import EvaluationReportGenerator

__all__ = ["EvaluationReportGenerator"]
