import os
from pathlib import Path
from weasyprint import HTML, CSS
from datetime import datetime
import logging
from typing import Dict

class PDFGenerator:
    """Generate PDF reports from HTML content."""
    
    def __init__(self, output_dir: str = 'results/reports/pdf'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_pdf(self, html_path: str, output_name: str = None) -> str:
        """Generate a PDF from an HTML file."""
        try:
            # Read HTML content
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Create PDF filename
            if output_name is None:
                output_name = Path(html_path).stem
            
            pdf_path = self.output_dir / f"{output_name}_{self.timestamp}.pdf"
            
            # Generate PDF
            HTML(string=html_content).write_pdf(
                pdf_path,
                stylesheets=[
                    CSS(string='''
                        @page {
                            margin: 2.5cm;
                            @top-right {
                                content: "Page " counter(page) " of " counter(pages);
                            }
                        }
                        body {
                            font-family: Arial, sans-serif;
                            line-height: 1.6;
                        }
                        h1, h2, h3 {
                            color: #2c3e50;
                        }
                        table {
                            width: 100%;
                            border-collapse: collapse;
                            margin: 1em 0;
                        }
                        th, td {
                            border: 1px solid #ddd;
                            padding: 8px;
                            text-align: left;
                        }
                        th {
                            background-color: #f2f2f2;
                        }
                        img {
                            max-width: 100%;
                            height: auto;
                        }
                        .section {
                            margin: 20px 0;
                        }
                        .metric {
                            margin: 10px 0;
                        }
                    ''')
                ]
            )
            
            logging.info(f"PDF generated successfully: {pdf_path}")
            return str(pdf_path)
            
        except Exception as e:
            logging.error(f"Error generating PDF: {str(e)}")
            raise
    
    def generate_all_pdfs(self, report_dir: str) -> Dict[str, str]:
        """Generate PDFs for all HTML reports in a directory."""
        report_dir = Path(report_dir)
        pdf_paths = {}
        
        for html_file in report_dir.glob('*.html'):
            try:
                pdf_path = self.generate_pdf(
                    str(html_file),
                    output_name=html_file.stem
                )
                pdf_paths[html_file.stem] = pdf_path
            except Exception as e:
                logging.error(f"Error generating PDF for {html_file}: {str(e)}")
        
        return pdf_paths 