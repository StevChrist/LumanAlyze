import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64

logger = logging.getLogger(__name__)

class ReportService:
    """Service untuk generate comprehensive reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.HexColor('#014F39')
        )
    
    def generate_pdf_report(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive PDF report"""
        logger.info("Generating PDF report")
        
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            
            # Title
            title = Paragraph("LumenALYZE Analysis Report", self.title_style)
            story.append(title)
            story.append(Spacer(1, 12))
            
            # Report metadata
            metadata = [
                ['Report Generated:', datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                ['Analysis Type:', analysis_results.get('model_type', 'Unknown')],
                ['Status:', analysis_results.get('status', 'Unknown')]
            ]
            
            metadata_table = Table(metadata)
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            
            story.append(metadata_table)
            story.append(Spacer(1, 20))
            
            # Performance metrics
            if 'metrics' in analysis_results:
                story.append(Paragraph("Performance Metrics", self.styles['Heading2']))
                metrics_data = self._format_metrics_for_table(analysis_results['metrics'])
                metrics_table = Table(metrics_data)
                metrics_table.setStyle(self._get_table_style())
                story.append(metrics_table)
                story.append(Spacer(1, 20))
            
            # Build PDF
            doc.build(story)
            pdf_content = buffer.getvalue()
            buffer.close()
            
            # Encode to base64
            pdf_base64 = base64.b64encode(pdf_content).decode()
            
            return {
                'status': 'success',
                'filename': f'lumenalyze_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
                'content': pdf_base64,
                'content_type': 'application/pdf',
                'size_kb': len(pdf_content) / 1024
            }
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise
    
    def _format_metrics_for_table(self, metrics: Dict[str, Any]) -> List[List[str]]:
        """Format metrics for table display"""
        table_data = [['Metric', 'Value']]
        
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            table_data.append([key.replace('_', ' ').title(), formatted_value])
        
        return table_data
    
    def _get_table_style(self):
        """Get standard table style"""
        return TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
